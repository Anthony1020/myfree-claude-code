"""API key pool manager backed by Supabase.

Keys are stored in the `api_keys` table.  On first use for a provider the
pool fetches the full key list (api_key, rate_limit_per_minute, reset_period,
reset_day) via the ``get_active_keys`` RPC and builds an in-process
``_KeyState`` for each key.

Per request, ``get_client()`` works as follows:

1. Find the first key whose **per-minute rolling window** has capacity
   (in-process check — no DB round-trip).
2. Call ``claim_api_request(key)`` on Supabase to atomically increment the
   daily/monthly counter.
   - TRUE  → use that key's cached AsyncOpenAI client.
   - FALSE → key's daily/monthly quota is exhausted; mark it exhausted until
             the period resets and try the next key.
3. If **all** keys are per-minute limited, wait until the soonest slot opens,
   then retry once.
4. If no key has remaining quota → return ``None`` (caller falls back to the
   env-var key).
"""

import time
from collections import deque
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx
from loguru import logger
from openai import AsyncOpenAI

from providers.base import ProviderConfig

# Short timeout for Supabase RPC calls.
_SUPABASE_TIMEOUT = httpx.Timeout(5.0)
_MINUTE = 60.0


class _KeyState:
    """In-process state for a single API key.

    Tracks:
    - Per-minute rolling window (proactive throttle, never hits the provider limit).
    - Period-exhaustion deadline (cleared automatically when the billing period resets).
    """

    def __init__(
        self,
        api_key: str,
        rate_limit_per_minute: int,
        reset_period: str,
        reset_day: int,
    ) -> None:
        self.api_key = api_key
        self.rate_limit_per_minute = max(1, rate_limit_per_minute)
        self._reset_period = reset_period
        self._reset_day = reset_day
        # Monotonic timestamps of the last `rate_limit_per_minute` acquisitions.
        self._request_times: deque[float] = deque()
        # Monotonic clock time until which this key's daily/monthly quota is exhausted.
        self._exhausted_until: float = 0.0

    # ------------------------------------------------------------------
    # Exhaustion (daily / monthly quota)
    # ------------------------------------------------------------------

    def is_period_exhausted(self) -> bool:
        return time.monotonic() < self._exhausted_until

    def mark_period_exhausted(self) -> None:
        """Set exhaustion deadline to the start of the next billing period."""
        now_utc = datetime.now(UTC)
        if self._reset_period == "daily":
            # Next period starts at UTC midnight tonight.
            next_reset = datetime(
                now_utc.year, now_utc.month, now_utc.day, tzinfo=UTC
            ) + timedelta(days=1)
        else:
            # Monthly: next reset_day in the future.
            try:
                candidate = date(now_utc.year, now_utc.month, self._reset_day)
            except ValueError:
                # reset_day > days in month — clamp to last day.
                next_month = (now_utc.replace(day=1) + timedelta(days=32)).replace(day=1)
                candidate = (next_month - timedelta(days=1))
            if candidate <= now_utc.date():
                # Already past this month's reset; advance to next month.
                year = now_utc.year + (1 if now_utc.month == 12 else 0)
                month = 1 if now_utc.month == 12 else now_utc.month + 1
                try:
                    candidate = date(year, month, self._reset_day)
                except ValueError:
                    candidate = date(year, month, 1) + timedelta(days=32)
                    candidate = candidate.replace(day=1) - timedelta(days=1)
            next_reset = datetime(
                candidate.year, candidate.month, candidate.day, tzinfo=UTC
            )

        delta = (next_reset - now_utc).total_seconds()
        self._exhausted_until = time.monotonic() + max(0.0, delta)
        logger.info(
            "KeyPool: key={}... quota exhausted until {} ({})",
            self.api_key[:8],
            next_reset.strftime("%Y-%m-%d %H:%M UTC"),
            self._reset_period,
        )

    # ------------------------------------------------------------------
    # Per-minute rolling window
    # ------------------------------------------------------------------

    def _prune(self, now: float) -> None:
        cutoff = now - _MINUTE
        while self._request_times and self._request_times[0] <= cutoff:
            self._request_times.popleft()

    def try_acquire_minute_slot(self) -> bool:
        """Non-blocking. Acquire a per-minute slot and return True, or False if full."""
        now = time.monotonic()
        self._prune(now)
        if len(self._request_times) < self.rate_limit_per_minute:
            self._request_times.append(now)
            return True
        return False

    def release_minute_slot(self) -> None:
        """Give back the last acquired slot (called when Supabase claim fails)."""
        if self._request_times:
            self._request_times.pop()

    def seconds_until_next_minute_slot(self) -> float:
        """How many seconds until a per-minute slot opens (0 if one is free now)."""
        now = time.monotonic()
        self._prune(now)
        if len(self._request_times) < self.rate_limit_per_minute:
            return 0.0
        # The oldest slot expires after 60 s.
        return max(0.0, self._request_times[0] + _MINUTE - now)


class KeyPoolManager:
    """Manages a pool of provider API keys stored in Supabase.

    Per-minute rate limiting is enforced in-process via ``_KeyState``.
    Daily/monthly quota is tracked atomically in Supabase via
    ``claim_api_request``.

    Usage::

        pool = KeyPoolManager(supabase_url, supabase_service_key)
        client = await pool.get_client("nvidia_nim", base_url, provider_config)
        if client is None:
            # all keys exhausted — caller falls back to env-var key
    """

    def __init__(self, supabase_url: str, supabase_service_key: str) -> None:
        self._url = supabase_url.rstrip("/")
        self._key = supabase_service_key
        # provider → ordered list of _KeyState (loaded once per provider on first use)
        self._provider_states: dict[str, list[_KeyState]] = {}
        # api_key → cached AsyncOpenAI client
        self._client_cache: dict[str, AsyncOpenAI] = {}

    # ------------------------------------------------------------------
    # Supabase helpers
    # ------------------------------------------------------------------

    async def _rpc(self, function_name: str, params: dict[str, Any]) -> Any:
        """POST to the Supabase REST RPC endpoint and return the decoded body."""
        url = f"{self._url}/rest/v1/rpc/{function_name}"
        headers = {
            "apikey": self._key,
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=_SUPABASE_TIMEOUT) as http:
            response = await http.post(url, json=params, headers=headers)
            response.raise_for_status()
            return response.json()

    async def _fetch_keys(self, provider: str) -> None:
        """Load active keys for *provider* from Supabase and build _KeyState objects."""
        try:
            rows = await self._rpc("get_active_keys", {"p_provider": provider})
            if not isinstance(rows, list):
                rows = []
            states = [
                _KeyState(
                    api_key=row["api_key"],
                    rate_limit_per_minute=int(row.get("rate_limit_per_minute", 20)),
                    reset_period=row.get("reset_period", "monthly"),
                    reset_day=int(row.get("reset_day", 1)),
                )
                for row in rows
                if isinstance(row, dict) and row.get("api_key")
            ]
            self._provider_states[provider] = states
            logger.info(
                "KeyPool: loaded {} active key(s) for provider={} "
                "(per-minute limits: {})",
                len(states),
                provider,
                [s.rate_limit_per_minute for s in states],
            )
        except Exception as e:
            logger.warning(
                "KeyPool: failed to fetch keys for provider={}: {}", provider, e
            )
            self._provider_states[provider] = []

    async def _get_states(self, provider: str) -> list[_KeyState]:
        if provider not in self._provider_states:
            await self._fetch_keys(provider)
        return self._provider_states.get(provider, [])

    async def _claim_specific(self, api_key: str) -> bool:
        """Call claim_api_request for a specific key. Returns True if quota available."""
        try:
            result = await self._rpc("claim_api_request", {"p_api_key": api_key})
            return result is True
        except Exception as e:
            logger.warning(
                "KeyPool: claim_api_request failed for key={}...: {}",
                api_key[:8],
                e,
            )
            return False

    def _get_or_create_client(
        self, api_key: str, base_url: str, config: ProviderConfig
    ) -> AsyncOpenAI:
        """Return the cached AsyncOpenAI client for *api_key*, creating it if needed."""
        if api_key not in self._client_cache:
            self._client_cache[api_key] = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                max_retries=0,
                timeout=httpx.Timeout(
                    config.http_read_timeout,
                    connect=config.http_connect_timeout,
                    read=config.http_read_timeout,
                    write=config.http_write_timeout,
                ),
            )
        return self._client_cache[api_key]

    # ------------------------------------------------------------------
    # Core selection logic
    # ------------------------------------------------------------------

    async def _try_claim(
        self,
        states: list[_KeyState],
        base_url: str,
        config: ProviderConfig,
    ) -> AsyncOpenAI | None:
        """One pass: find the first key with per-minute capacity and Supabase quota."""
        for state in states:
            if state.is_period_exhausted():
                continue
            if not state.try_acquire_minute_slot():
                continue
            # Per-minute slot acquired — now verify daily/monthly quota in Supabase.
            if await self._claim_specific(state.api_key):
                return self._get_or_create_client(state.api_key, base_url, config)
            # Supabase says quota exhausted; release the minute slot and mark key.
            state.release_minute_slot()
            state.mark_period_exhausted()
        return None

    async def get_client(
        self, provider: str, base_url: str, config: ProviderConfig
    ) -> AsyncOpenAI | None:
        """Return an AsyncOpenAI client for the best available key.

        Selection order:
        1. First pass — find a key with per-minute capacity AND Supabase quota.
        2. If all non-exhausted keys are at their per-minute limit, wait for the
           soonest slot to open (≤ 60 s) then retry once.
        3. Return ``None`` if all keys are period-exhausted or no key is found
           after waiting (caller falls back to the env-var key).
        """
        states = await self._get_states(provider)
        if not states:
            logger.warning("KeyPool: no active keys found for provider={}", provider)
            return None

        # First pass
        client = await self._try_claim(states, base_url, config)
        if client is not None:
            return client

        # All available keys are at per-minute limit — compute wait time.
        import asyncio

        active = [s for s in states if not s.is_period_exhausted()]
        if not active:
            logger.warning("KeyPool: all {} keys are period-exhausted", provider)
            return None

        wait = min(s.seconds_until_next_minute_slot() for s in active)
        if wait > 0:
            logger.info(
                "KeyPool: all {} keys at per-minute limit, waiting {:.1f}s",
                provider,
                wait,
            )
            await asyncio.sleep(wait)

        # Retry after wait
        return await self._try_claim(states, base_url, config)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def cleanup(self) -> None:
        """Close all cached AsyncOpenAI clients."""
        for client in self._client_cache.values():
            await client.aclose()
        self._client_cache.clear()
        self._provider_states.clear()
