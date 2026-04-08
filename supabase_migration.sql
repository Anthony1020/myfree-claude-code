-- ============================================================
-- API Key Pool Migration
-- Adds reset_period ('daily' | 'monthly') to api_keys and
-- updates all functions to support both period types.
--
-- Run this in the Supabase SQL Editor (public schema).
-- Safe to run on a fresh DB or on top of the original script.
-- ============================================================

-- 1. API Keys Configuration Table
CREATE TABLE IF NOT EXISTS public.api_keys (
  api_key       TEXT PRIMARY KEY,
  provider      TEXT        NOT NULL DEFAULT 'default',
  label         TEXT,
  reset_day     INT         NOT NULL DEFAULT 1,
  reset_period  TEXT        NOT NULL DEFAULT 'monthly'
                            CHECK (reset_period IN ('daily', 'monthly')),
  max_requests  INT         NOT NULL DEFAULT 200,
  is_active     BOOLEAN     NOT NULL DEFAULT TRUE,
  created_at    TIMESTAMPTZ          DEFAULT NOW()
);

-- Add reset_period to existing tables (idempotent)
ALTER TABLE public.api_keys
  ADD COLUMN IF NOT EXISTS reset_period TEXT NOT NULL DEFAULT 'monthly'
  CHECK (reset_period IN ('daily', 'monthly'));

-- 2. API Usage Tracking Table
CREATE TABLE IF NOT EXISTS public.api_usage (
  id             BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  api_key        TEXT        NOT NULL REFERENCES public.api_keys(api_key),
  period_start   DATE        NOT NULL,
  period_end     DATE        NOT NULL,
  request_count  INT         NOT NULL DEFAULT 0,
  created_at     TIMESTAMPTZ          DEFAULT NOW(),
  updated_at     TIMESTAMPTZ          DEFAULT NOW(),
  UNIQUE (api_key, period_start)
);

CREATE INDEX IF NOT EXISTS idx_api_usage_key_period
  ON public.api_usage (api_key, period_start);

-- ============================================================
-- Helper: compute period bounds for a key
-- Returns (period_start, period_end) based on reset_period.
--   'daily'   -> (CURRENT_DATE, CURRENT_DATE)
--   'monthly' -> billing window based on reset_day
-- ============================================================
CREATE OR REPLACE FUNCTION public._get_period_bounds(
  p_reset_period TEXT,
  p_reset_day    INT,
  OUT v_period_start DATE,
  OUT v_period_end   DATE
)
LANGUAGE plpgsql AS $$
BEGIN
  IF p_reset_period = 'daily' THEN
    v_period_start := CURRENT_DATE;
    v_period_end   := CURRENT_DATE;
  ELSE
    -- Monthly: find the most recent reset_day on or before today
    IF EXTRACT(DAY FROM CURRENT_DATE)::INT >= p_reset_day THEN
      v_period_start := DATE_TRUNC('month', CURRENT_DATE)::DATE + (p_reset_day - 1);
    ELSE
      v_period_start := (DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '1 month')::DATE
                        + (p_reset_day - 1);
    END IF;
    v_period_end := (v_period_start + INTERVAL '1 month' - INTERVAL '1 day')::DATE;
  END IF;
END;
$$;

-- ============================================================
-- 3. claim_api_request
-- Atomically increments the counter for a key if under the limit.
-- Returns TRUE if allowed, FALSE if limit reached or key inactive.
-- ============================================================
CREATE OR REPLACE FUNCTION public.claim_api_request(p_api_key TEXT)
RETURNS BOOLEAN
LANGUAGE plpgsql AS $$
DECLARE
  v_reset_day    INT;
  v_reset_period TEXT;
  v_max_requests INT;
  v_is_active    BOOLEAN;
  v_period_start DATE;
  v_period_end   DATE;
  v_rows_updated INT;
BEGIN
  SELECT reset_day, reset_period, max_requests, is_active
    INTO v_reset_day, v_reset_period, v_max_requests, v_is_active
    FROM public.api_keys
   WHERE api_key = p_api_key;

  IF NOT FOUND THEN
    RAISE EXCEPTION 'API key not found: %', p_api_key;
  END IF;

  IF NOT v_is_active THEN
    RETURN FALSE;
  END IF;

  SELECT * INTO v_period_start, v_period_end
    FROM public._get_period_bounds(v_reset_period, v_reset_day);

  INSERT INTO public.api_usage (api_key, period_start, period_end, request_count)
  VALUES (p_api_key, v_period_start, v_period_end, 0)
  ON CONFLICT (api_key, period_start) DO NOTHING;

  UPDATE public.api_usage
     SET request_count = request_count + 1,
         updated_at    = NOW()
   WHERE api_key      = p_api_key
     AND period_start = v_period_start
     AND request_count < v_max_requests;

  GET DIAGNOSTICS v_rows_updated = ROW_COUNT;
  RETURN v_rows_updated > 0;
END;
$$;

-- ============================================================
-- 4. get_api_usage
-- Returns current usage info for a key (dashboards / debugging).
-- ============================================================
CREATE OR REPLACE FUNCTION public.get_api_usage(p_api_key TEXT)
RETURNS TABLE (
  api_key        TEXT,
  label          TEXT,
  provider       TEXT,
  reset_day      INT,
  reset_period   TEXT,
  max_requests   INT,
  current_count  INT,
  remaining      INT,
  period_start   DATE,
  period_end     DATE,
  is_active      BOOLEAN
)
LANGUAGE plpgsql AS $$
DECLARE
  v_reset_day    INT;
  v_reset_period TEXT;
  v_period_start DATE;
  v_period_end   DATE;
BEGIN
  SELECT ak.reset_day, ak.reset_period
    INTO v_reset_day, v_reset_period
    FROM public.api_keys ak
   WHERE ak.api_key = p_api_key;

  IF NOT FOUND THEN
    RAISE EXCEPTION 'API key not found: %', p_api_key;
  END IF;

  SELECT * INTO v_period_start, v_period_end
    FROM public._get_period_bounds(v_reset_period, v_reset_day);

  RETURN QUERY
  SELECT
    ak.api_key,
    ak.label,
    ak.provider,
    ak.reset_day,
    ak.reset_period,
    ak.max_requests,
    COALESCE(au.request_count, 0)                       AS current_count,
    ak.max_requests - COALESCE(au.request_count, 0)     AS remaining,
    v_period_start                                       AS period_start,
    v_period_end                                         AS period_end,
    ak.is_active
  FROM public.api_keys ak
  LEFT JOIN public.api_usage au
    ON au.api_key     = ak.api_key
   AND au.period_start = v_period_start
  WHERE ak.api_key = p_api_key;
END;
$$;

-- ============================================================
-- 5. claim_next_available_key
-- Tries each active key for a provider in order; returns the
-- first one with remaining capacity, or NULL if all exhausted.
-- ============================================================
CREATE OR REPLACE FUNCTION public.claim_next_available_key(p_provider TEXT)
RETURNS TEXT
LANGUAGE plpgsql AS $$
DECLARE
  v_key     RECORD;
  v_claimed BOOLEAN;
BEGIN
  FOR v_key IN
    SELECT ak.api_key
      FROM public.api_keys ak
     WHERE ak.provider  = p_provider
       AND ak.is_active = TRUE
     ORDER BY ak.api_key   -- deterministic; swap for RANDOM() to load-balance
  LOOP
    v_claimed := public.claim_api_request(v_key.api_key);
    IF v_claimed THEN
      RETURN v_key.api_key;
    END IF;
  END LOOP;

  RETURN NULL;
END;
$$;
