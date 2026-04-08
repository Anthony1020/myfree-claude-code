"""API key pool manager backed by Supabase.

Keys are stored in the `api_keys` table and claimed atomically via
the `claim_next_available_key` Supabase RPC before each provider request.
A per-key AsyncOpenAI client cache avoids recreating HTTP clients on every call.
"""

from typing import Any

import httpx
from loguru import logger
from openai import AsyncOpenAI

from providers.base import ProviderConfig

# Short timeout for Supabase RPC calls — these should be fast.
_SUPABASE_TIMEOUT = httpx.Timeout(5.0)


class KeyPoolManager:
    """Manages a pool of provider API keys stored in Supabase.

    Usage:
        pool = KeyPoolManager(supabase_url, supabase_service_key)
        client = await pool.get_client("nvidia_nim", base_url, provider_config)
        if client is None:
            # all keys exhausted for this provider
    """

    def __init__(self, supabase_url: str, supabase_service_key: str) -> None:
        self._url = supabase_url.rstrip("/")
        self._key = supabase_service_key
        # Cache one AsyncOpenAI client per API key string to avoid rebuilding
        # HTTP connections on every request.
        self._client_cache: dict[str, AsyncOpenAI] = {}

    # ------------------------------------------------------------------
    # Internal helpers
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def claim_next_key(self, provider: str) -> str | None:
        """Atomically claim the next available API key for *provider*.

        Calls the Supabase ``claim_next_available_key(p_provider)`` function
        which increments the usage counter and returns the key string, or
        ``null`` / empty if all keys for the provider are exhausted.

        Returns the key string on success, ``None`` on exhaustion or error.
        """
        try:
            result = await self._rpc(
                "claim_next_available_key", {"p_provider": provider}
            )
            # Supabase returns the TEXT value directly or JSON null.
            if isinstance(result, str) and result:
                return result
            return None
        except Exception as e:
            logger.warning(
                "KeyPool: Supabase RPC failed for provider={}: {}", provider, e
            )
            return None

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

    async def get_client(
        self, provider: str, base_url: str, config: ProviderConfig
    ) -> AsyncOpenAI | None:
        """Claim the next available key and return its AsyncOpenAI client.

        Returns ``None`` if the pool is exhausted for *provider*.
        """
        api_key = await self.claim_next_key(provider)
        if api_key is None:
            return None
        return self._get_or_create_client(api_key, base_url, config)

    async def cleanup(self) -> None:
        """Close all cached AsyncOpenAI clients."""
        for client in self._client_cache.values():
            await client.aclose()
        self._client_cache.clear()
