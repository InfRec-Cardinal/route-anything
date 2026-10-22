"""Configurable OpenAI-compatible client for the archived collection scripts.

Every experiment talks to the relay ONLY through this module.  It provides

* ``PackyClient.chat``      -- POST /chat/completions (Chat Completions API)
* ``PackyClient.responses`` -- POST /responses (Responses API; gpt-5.6-sol, gpt-6-astra)
* ``PackyClient.map``       -- batch of chat calls under a per-model concurrency cap
* disk cache               -- sqlite ``<cache_dir>/packy_cache.sqlite``; a hit never re-calls
* usage ledger             -- ``<cache_dir>/usage_ledger.jsonl`` (one row per call, cached or live)
* budget guard             -- before every live call the ledger's summed ``units_est`` is compared
                              with ``cap_units`` in ``common/budget.json``; ``BudgetExceeded`` stops
                              the client when the cap is exceeded
* relay accounting         -- ``relay_usage()`` returns the relay's ``total_usage`` (GET
                              /dashboard/billing/usage); ``set_units_per_token`` stores a
                              calibrated per-token rate in ``budget.json``.

Conventions enforced here (see CONVENTIONS.md):

* Claude models are excluded by the archived collection profile; their outcomes are replayed.
* ``gpt-5.4`` is not offered.  It is mapped to ``gpt-5.6-sol`` only when the caller passes
  ``allow_substitute=True``; the result then carries ``substituted_for='gpt-5.4'``.
* ``gpt-5.6-sol`` / ``gpt-6-astra`` only support the Responses API; ``chat()`` transparently routes
  them through ``/responses`` (messages are passed as Responses-API input items).
* deepseek-flash returns visible content only when ``max_tokens >= 512`` (reasoning tokens are
  billed first); a floor of 512 is enforced for it.  deepseek-v4-pro also reasons by default;
  pass ``extra={"thinking": {"type": "disabled"}}`` to turn reasoning off (the judge does this).
* Timeouts: 300 s default, glm-5 900 s, qwen3.5-flash / qwen3.8-flash 600 s.  Timeouts, HTTP 429,
  HTTP 5xx and empty content are retried up to ``max_retries`` (default 3) times with exponential
  backoff.  Empty content with ``finish_reason == 'length'`` is a deterministic truncation and is
  returned as-is (retrying the same body cannot help).  Nothing is stripped from responses.
The endpoint and authentication are supplied through ``ROUTER_API_BASE_URL`` and
``ROUTER_API_KEY``. This module supplies no provider address or local credential file.

Units and calibration
---------------------
``units_est`` per call = ``total_tokens * units_per_token_by_model[model]`` (default
``DEFAULT_UNITS_PER_TOKEN`` = 0.05 units/token when a model has no calibrated rate).  Cached hits
cost 0 units.  The relay's ``total_usage`` counter is account-wide and moves while other consumers
are active, so a calibration from a before/after delta is only as good as the background drift
allows; ``set_units_per_token`` therefore stores the measurement together with the metadata the
caller passes (drift, tokens, quality flag).

The module has no side effects at import time; files are created lazily by ``PackyClient``.
"""
from __future__ import annotations

import json
import os
import random
import re
import sqlite3
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx

try:  # tqdm is in the venv, but keep the client importable without it
    from tqdm import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None

# --------------------------------------------------------------------------------------- constants

EX_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMON_DIR = os.path.join(EX_DIR, "common")
DEFAULT_CACHE_DIR = os.path.join(COMMON_DIR, "cache")
DEFAULT_BUDGET_PATH = os.path.join(COMMON_DIR, "budget.json")

CONCURRENCY: Dict[str, int] = {
    "deepseek-v4-pro": 8,
    "deepseek-flash": 8,
    "qwen3-max": 4,
    "qwen3.5-plus": 4,
    "kimi-k2.5": 2,
    "MiniMax-M2.7": 2,
    "glm-5": 1,
    "grok-4.6": 10,
    "qwen3-coder-next": 2,
    "gpt-5.6-sol": 4,
}
DEFAULT_CONCURRENCY = 2

DEFAULT_TIMEOUT_S = 300.0
TIMEOUTS_S: Dict[str, float] = {"glm-5": 900.0, "qwen3.5-flash": 600.0, "qwen3.8-flash": 600.0}

MIN_MAX_TOKENS: Dict[str, int] = {"deepseek-flash": 512}

RESPONSES_ONLY_MODELS = {"gpt-5.6-sol", "gpt-6-astra", "codex-auto-review"}
SUBSTITUTES: Dict[str, str] = {"gpt-5.4": "gpt-5.6-sol"}

DEFAULT_UNITS_PER_TOKEN = 0.05
MIN_SANE_UNITS_PER_TOKEN = 1e-5  # calibration results outside this range are reported but not stored
MAX_SANE_UNITS_PER_TOKEN = 2.0
DEFAULT_CAP_UNITS = 60000.0
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
LEDGER_RESCAN_S = 5.0  # re-read the ledger file (other processes may append) at most this often


# -------------------------------------------------------------------------------------- exceptions


class PackyError(Exception):
    """Base class for all errors raised by this module."""


class BudgetExceeded(PackyError):
    """Raised before a live call when the ledger's summed ``units_est`` exceeds ``cap_units``."""


class ModelUnavailable(PackyError):
    """Raised for models unavailable in the configured collection profile."""


class RelayError(PackyError):
    """Raised when the relay returns a non-retryable error or retries are exhausted."""


class _EmptyContent(PackyError):
    """Internal: the relay answered 200 but with empty content (retried like a transient error)."""


# ----------------------------------------------------------------------------------------- helpers


def _read_api_key() -> str:
    value = os.environ.get("ROUTER_API_KEY", "").strip()
    if not value:
        raise PackyError("Set ROUTER_API_KEY before collecting new responses")
    return value


def _make_redactor(secret: str) -> Callable[[Any], str]:
    """Return a function that stringifies its argument with ``secret`` replaced by a placeholder."""

    def redact(text: Any) -> str:
        return str(text).replace(secret, "<REDACTED>")

    return redact


def canonical_json(obj: Any) -> str:
    """Canonical JSON serialisation used for cache keys (sorted keys, compact separators)."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def cache_key(model: str, body: Dict[str, Any]) -> str:
    """Return the stable identifier used by the archived response cache."""
    return canonical_json({"model": model, "body": body})


def _now() -> float:
    """Wall-clock seconds (monotonic enough for latency measurements)."""
    return time.time()


def _int(v: Any) -> int:
    """Coerce a possibly-missing usage counter to int."""
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def _content_text(content: Any) -> str:
    """Text of a chat message ``content`` (string or list of text parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return "" if content is None else str(content)


def flatten_messages(messages: Sequence[Dict[str, Any]]) -> str:
    """Render chat messages as ONE Responses-API input string.

    The relay's gpt-5.6-sol endpoint accepted a plain string ``input`` but answered HTTP 403
    ("client anomaly") to a list of role/content items, so Chat-style requests to Responses-only
    models are flattened: a single user message is passed verbatim; otherwise turns are labelled
    ``System:`` / ``User:`` / ``Assistant:`` and separated by blank lines.
    """
    msgs = [m for m in messages if isinstance(m, dict)]
    if len(msgs) == 1 and str(msgs[0].get("role", "user")) == "user":
        return _content_text(msgs[0].get("content"))
    parts = []
    for m in msgs:
        role = str(m.get("role", "user")).capitalize()
        parts.append(f"{role}:\n{_content_text(m.get('content'))}")
    return "\n\n".join(parts)


def parse_chat_response(js: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a Chat Completions JSON body into the client's result dict (without cache metadata).

    Returns ``dict(text, reasoning, usage, finish_reason)`` where ``usage`` has
    ``prompt_tokens, completion_tokens, reasoning_tokens, total_tokens``.
    """
    choices = js.get("choices") or []
    msg = (choices[0].get("message") or {}) if choices else {}
    content = msg.get("content")
    if isinstance(content, list):  # some relays return content parts
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or None
    usage = js.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    parsed_usage = {
        "prompt_tokens": _int(usage.get("prompt_tokens")),
        "completion_tokens": _int(usage.get("completion_tokens")),
        "reasoning_tokens": _int(details.get("reasoning_tokens")),
        "total_tokens": _int(usage.get("total_tokens"))
        or _int(usage.get("prompt_tokens")) + _int(usage.get("completion_tokens")),
    }
    return {
        "text": content if isinstance(content, str) else ("" if content is None else str(content)),
        "reasoning": reasoning,
        "usage": parsed_usage,
        "finish_reason": (choices[0].get("finish_reason") if choices else None),
    }


def parse_responses_response(js: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a Responses-API JSON body: concatenates ``output[*].content[*]`` of type ``output_text``.

    ``finish_reason`` is the response ``status`` ('completed', 'incomplete', ...) with the
    ``incomplete_details.reason`` appended when present.  Reasoning summaries (output items of type
    ``reasoning``) are returned in ``reasoning`` when the relay includes them.
    """
    texts: List[str] = []
    reasoning_parts: List[str] = []
    for item in js.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "reasoning":
            for s in item.get("summary") or []:
                if isinstance(s, dict) and s.get("text"):
                    reasoning_parts.append(s["text"])
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "output_text":
                texts.append(part.get("text") or "")
    if not texts and isinstance(js.get("output_text"), str):  # SDK-style convenience field
        texts.append(js["output_text"])
    usage = js.get("usage") or {}
    details = usage.get("output_tokens_details") or {}
    parsed_usage = {
        "prompt_tokens": _int(usage.get("input_tokens")),
        "completion_tokens": _int(usage.get("output_tokens")),
        "reasoning_tokens": _int(details.get("reasoning_tokens")),
        "total_tokens": _int(usage.get("total_tokens"))
        or _int(usage.get("input_tokens")) + _int(usage.get("output_tokens")),
    }
    status = js.get("status")
    inc = js.get("incomplete_details") or {}
    finish = status if not inc.get("reason") else f"{status}:{inc.get('reason')}"
    return {
        "text": "".join(texts),
        "reasoning": "\n".join(reasoning_parts) or None,
        "usage": parsed_usage,
        "finish_reason": finish,
    }


# ------------------------------------------------------------------------------------------ client


class PackyClient:
    """Cached, budgeted, rate-capped client for the Packy relay.

    Parameters
    ----------
    cache_dir : str
        Directory for ``packy_cache.sqlite`` and ``usage_ledger.jsonl`` (created if missing).
    budget_path : str
        ``budget.json`` with ``cap_units`` and ``units_per_token_by_model``.
    base_url : str
        API base URL; defaults to ``ROUTER_API_BASE_URL``.
    max_retries : int
        Retries after the first attempt for timeouts / 429 / 5xx / empty content.
    quiet : bool
        Disable the tqdm bar in ``map``.
    """

    def __init__(
        self,
        cache_dir: str = DEFAULT_CACHE_DIR,
        *,
        budget_path: str = DEFAULT_BUDGET_PATH,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        quiet: bool = False,
        offline: bool = False,
    ) -> None:
        """Open the cache and prepare the configured HTTP session."""
        configured_url = (base_url or os.environ.get("ROUTER_API_BASE_URL", "")).strip()
        if not configured_url and not offline:
            raise PackyError("Set ROUTER_API_BASE_URL before collecting new responses")
        api_key = _read_api_key() if not offline else ""
        self.offline = offline
        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.budget_path = budget_path
        self.base_url = configured_url.rstrip("/")
        self.max_retries = int(max_retries)
        self.quiet = quiet
        self._redact = _make_redactor(api_key) if api_key else str
        self._http = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=httpx.Timeout(DEFAULT_TIMEOUT_S, connect=30.0),
        ) if not offline else None
        del api_key
        self.db_path = os.path.join(self.cache_dir, "packy_cache.sqlite")
        self.ledger_path = os.path.join(self.cache_dir, "usage_ledger.jsonl")
        self._db = sqlite3.connect(self.db_path, check_same_thread=False, timeout=60.0)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, model TEXT, endpoint TEXT, "
            "ts REAL, latency_s REAL, request TEXT, response TEXT, usage TEXT)"
        )
        self._db.commit()
        self._db_lock = threading.Lock()
        self._ledger_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._sem_lock = threading.Lock()
        self._semaphores: Dict[str, threading.Semaphore] = {}
        self._inflight: Dict[str, int] = defaultdict(int)
        self._max_inflight: Dict[str, int] = defaultdict(int)
        self._session: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {
                "calls": 0,
                "cached_hits": 0,
                "live_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
                "units_est": 0.0,
                "latency_s_total": 0.0,
            }
        )
        self._ledger_total_units = 0.0
        self._ledger_scanned_at = 0.0
        self._rescan_ledger(force=True)

    # ------------------------------------------------------------------ housekeeping / dunder

    def __repr__(self) -> str:
        """Representation without secrets."""
        return f"PackyClient(base_url={self.base_url!r}, cache_dir={self.cache_dir!r})"

    def __enter__(self) -> "PackyClient":
        """Context-manager entry."""
        return self

    def __exit__(self, *exc: Any) -> None:
        """Context-manager exit: close network and database handles."""
        self.close()

    def close(self) -> None:
        """Close the HTTP client and the sqlite connection."""
        try:
            if self._http is not None:
                self._http.close()
        finally:
            with self._db_lock:
                self._db.close()

    # ------------------------------------------------------------------ budget / ledger

    def load_budget(self) -> Dict[str, Any]:
        """Read ``budget.json`` (returns defaults when the file is missing or malformed)."""
        try:
            with open(self.budget_path, "r", encoding="utf-8") as fh:
                b = json.load(fh)
        except (OSError, json.JSONDecodeError):
            b = {}
        b.setdefault("cap_units", DEFAULT_CAP_UNITS)
        b.setdefault("units_per_token_by_model", {})
        return b

    def _write_budget(self, budget: Dict[str, Any]) -> None:
        """Atomically rewrite ``budget.json``."""
        tmp = self.budget_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(budget, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp, self.budget_path)

    def units_per_token(self, model: str) -> float:
        """Calibrated units/token for ``model`` (``DEFAULT_UNITS_PER_TOKEN`` when unknown)."""
        rates = self.load_budget().get("units_per_token_by_model") or {}
        try:
            return float(rates.get(model, DEFAULT_UNITS_PER_TOKEN))
        except (TypeError, ValueError):
            return DEFAULT_UNITS_PER_TOKEN

    def set_units_per_token(self, model: str, rate: float, meta: Optional[Dict[str, Any]] = None) -> None:
        """Store a calibrated per-token rate (and its measurement metadata) in ``budget.json``.

        ``meta`` is saved under ``calibration[model]`` (e.g. tokens, usage delta, background drift,
        quality flag) so later readers can judge how trustworthy the rate is.
        """
        budget = self.load_budget()
        budget.setdefault("units_per_token_by_model", {})[model] = float(rate)
        if meta is not None:
            budget.setdefault("calibration", {})[model] = dict(meta, ts=_now())
        self._write_budget(budget)

    def _rescan_ledger(self, force: bool = False) -> float:
        """Re-read the ledger file (other processes append too) and return the summed units_est."""
        if not force and (_now() - self._ledger_scanned_at) < LEDGER_RESCAN_S:
            return self._ledger_total_units
        total = 0.0
        if os.path.exists(self.ledger_path):
            with open(self.ledger_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        total += float(json.loads(line).get("units_est") or 0.0)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
        self._ledger_total_units = total
        self._ledger_scanned_at = _now()
        return total

    def ledger_units(self) -> float:
        """Sum of ``units_est`` over the whole ledger (cached hits contribute 0)."""
        with self._ledger_lock:
            return self._rescan_ledger(force=True)

    def _check_budget(self) -> None:
        """Raise ``BudgetExceeded`` when the ledger already exceeds ``cap_units``."""
        cap = float(self.load_budget().get("cap_units", DEFAULT_CAP_UNITS))
        with self._ledger_lock:
            total = self._rescan_ledger()
        if total > cap:
            raise BudgetExceeded(f"ledger units_est {total:.1f} exceeds cap_units {cap:.1f} ({self.budget_path})")

    def _ledger_append(self, row: Dict[str, Any]) -> None:
        """Append one JSON row to the ledger and update the in-memory running total."""
        line = json.dumps(row, ensure_ascii=False)
        with self._ledger_lock:
            with open(self.ledger_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            self._ledger_total_units += float(row.get("units_est") or 0.0)

    # ------------------------------------------------------------------ cache

    def _cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        """Fetch a cached response row by key (``None`` on miss)."""
        with self._db_lock:
            cur = self._db.execute("SELECT model, endpoint, ts, latency_s, response, usage FROM cache WHERE key=?", (key,))
            row = cur.fetchone()
        if row is None:
            return None
        model, endpoint, ts, latency_s, response, usage = row
        return {
            "model": model,
            "endpoint": endpoint,
            "ts": ts,
            "latency_s": latency_s,
            "response": json.loads(response),
            "usage": json.loads(usage) if usage else None,
        }

    def _cache_put(self, key: str, model: str, endpoint: str, body: Dict[str, Any], response: Dict[str, Any], usage: Dict[str, Any], latency_s: float) -> None:
        """Store the full response JSON plus usage and timestamp under ``key``."""
        with self._db_lock:
            self._db.execute(
                "INSERT OR REPLACE INTO cache (key, model, endpoint, ts, latency_s, request, response, usage) VALUES (?,?,?,?,?,?,?,?)",
                (key, model, endpoint, _now(), latency_s, canonical_json(body), json.dumps(response, ensure_ascii=False), json.dumps(usage)),
            )
            self._db.commit()

    def cache_stats(self) -> Dict[str, Any]:
        """Number of cached responses per model."""
        with self._db_lock:
            rows = self._db.execute("SELECT model, COUNT(*) FROM cache GROUP BY model").fetchall()
        return {m: n for m, n in rows}

    # ------------------------------------------------------------------ concurrency

    def _semaphore(self, model: str) -> threading.Semaphore:
        """Per-model semaphore sized by ``CONCURRENCY`` (``DEFAULT_CONCURRENCY`` for unknown models)."""
        with self._sem_lock:
            sem = self._semaphores.get(model)
            if sem is None:
                sem = threading.Semaphore(CONCURRENCY.get(model, DEFAULT_CONCURRENCY))
                self._semaphores[model] = sem
            return sem

    def concurrency_stats(self) -> Dict[str, Dict[str, int]]:
        """Per-model ``{'cap': n, 'max_inflight': m}`` observed in this session (live calls only)."""
        with self._sem_lock:
            models = set(self._semaphores) | set(self._max_inflight)
            return {m: {"cap": CONCURRENCY.get(m, DEFAULT_CONCURRENCY), "max_inflight": int(self._max_inflight.get(m, 0))} for m in sorted(models)}

    # ------------------------------------------------------------------ model resolution

    def resolve_model(self, model: str, allow_substitute: bool = False) -> Tuple[str, Optional[str]]:
        """Return ``(model_to_call, substituted_for)`` or raise ``ModelUnavailable``.

        Claude models raise immediately.  ``gpt-5.4`` maps to ``gpt-5.6-sol`` only when
        ``allow_substitute`` is true; the second element then names the requested model.
        """
        if model.lower().startswith("claude"):
            raise ModelUnavailable(f"{model}: this model is excluded by the archived collection profile")
        if model in SUBSTITUTES:
            sub = SUBSTITUTES[model]
            if not allow_substitute:
                raise ModelUnavailable(f"{model} is not offered by the relay; pass allow_substitute=True to use {sub} (always labelled as a substitute)")
            return sub, model
        return model, None

    @staticmethod
    def timeout_for(model: str) -> float:
        """Default request timeout in seconds for ``model``."""
        return TIMEOUTS_S.get(model, DEFAULT_TIMEOUT_S)

    # ------------------------------------------------------------------ public API

    def chat(
        self,
        model: str,
        messages: Sequence[Dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float = 0,
        tag: str,
        timeout: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
        allow_substitute: bool = False,
    ) -> Dict[str, Any]:
        """One Chat Completions call (cached, budgeted, retried).

        Returns ``dict(text, reasoning, usage, latency_s, cached, model, finish_reason)`` plus
        ``endpoint``, ``cache_key``, ``retries`` and -- when gpt-5.4 was substituted --
        ``substituted_for``.  ``extra`` is merged into the request body (e.g.
        ``{"thinking": {"type": "disabled"}}`` for DeepSeek, ``{"response_format": ...}``).
        Models that only support the Responses API are routed there with ``messages`` as input.
        """
        model_used, substituted_for = self.resolve_model(model, allow_substitute)
        if model_used in RESPONSES_ONLY_MODELS:
            body: Dict[str, Any] = {"model": model_used, "input": flatten_messages(messages), "max_output_tokens": int(max_tokens)}
            body.update(extra or {})
            return self._execute("responses", model_used, body, tag=tag, timeout=timeout, parse=parse_responses_response, substituted_for=substituted_for)
        max_tokens = max(int(max_tokens), MIN_MAX_TOKENS.get(model_used, 0))
        body = {"model": model_used, "messages": list(messages), "max_tokens": max_tokens, "temperature": temperature}
        body.update(extra or {})
        return self._execute("chat/completions", model_used, body, tag=tag, timeout=timeout, parse=parse_chat_response, substituted_for=substituted_for)

    def responses(
        self,
        model: str,
        input_text: Any,
        *,
        max_output_tokens: int,
        tag: str,
        timeout: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
        allow_substitute: bool = False,
    ) -> Dict[str, Any]:
        """One Responses-API call (POST /responses) for gpt-5.6-sol / gpt-6-astra.

        ``input_text`` may be a string or a list of Responses-API input items.  The text is the
        concatenation of ``output[*].content[*].output_text``.  Same result dict as ``chat``.
        """
        model_used, substituted_for = self.resolve_model(model, allow_substitute)
        body: Dict[str, Any] = {"model": model_used, "input": input_text, "max_output_tokens": int(max_output_tokens)}
        body.update(extra or {})
        return self._execute("responses", model_used, body, tag=tag, timeout=timeout, parse=parse_responses_response, substituted_for=substituted_for)

    def map(
        self,
        model: str,
        list_of_message_lists: Sequence[Sequence[Dict[str, Any]]],
        *,
        workers: Optional[int] = None,
        tag: str,
        **kw: Any,
    ) -> List[Dict[str, Any]]:
        """Run ``chat`` over many message lists in parallel, never exceeding the model's cap.

        ``workers`` defaults to the model's concurrency cap and is clipped to it.  Results are
        returned in input order.  Per-item ``RelayError`` s become result dicts with
        ``finish_reason='error'`` and an ``error`` field; ``BudgetExceeded`` and ``ModelUnavailable``
        cancel the remaining work and propagate.  ``kw`` is forwarded to ``chat`` (``max_tokens`` is
        required).  A tqdm bar is shown unless the client was created with ``quiet=True``.
        """
        model_used, _ = self.resolve_model(model, kw.get("allow_substitute", False))
        cap = CONCURRENCY.get(model_used, DEFAULT_CONCURRENCY)
        n_workers = cap if workers is None else max(1, min(int(workers), cap))
        items = list(list_of_message_lists)
        results: List[Optional[Dict[str, Any]]] = [None] * len(items)
        if not items:
            return []
        bar = _tqdm(total=len(items), desc=f"packy:{model_used}", disable=self.quiet or _tqdm is None) if _tqdm else None
        fatal: Optional[BaseException] = None
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(self.chat, model, msgs, tag=tag, **kw): i for i, msgs in enumerate(items)}
            try:
                for fut in as_completed(futures):
                    i = futures[fut]
                    try:
                        results[i] = fut.result()
                    except (BudgetExceeded, ModelUnavailable) as e:
                        fatal = e
                        for f in futures:
                            f.cancel()
                        break
                    except RelayError as e:
                        results[i] = self._error_result(model_used, e)
                    finally:
                        if bar is not None:
                            bar.update(1)
            finally:
                if bar is not None:
                    bar.close()
        if fatal is not None:
            raise fatal
        for i, r in enumerate(results):  # cancelled futures (only after a fatal error) -> error rows
            if r is None:
                results[i] = self._error_result(model_used, RelayError("cancelled"))
        return results  # type: ignore[return-value]

    def usage_summary(self) -> Dict[str, Any]:
        """Session and ledger usage: per-model token counts, units, cache hits, cap and remainder."""
        with self._stats_lock:
            session = {m: dict(v) for m, v in self._session.items()}
        session_units = sum(v["units_est"] for v in session.values())
        budget = self.load_budget()
        cap = float(budget.get("cap_units", DEFAULT_CAP_UNITS))
        ledger_total = self.ledger_units()
        return {
            "session": session,
            "session_units_est": session_units,
            "ledger_units_total": ledger_total,
            "cap_units": cap,
            "remaining_units": cap - ledger_total,
            "units_per_token_by_model": budget.get("units_per_token_by_model", {}),
            "concurrency": self.concurrency_stats(),
        }

    def relay_usage(self, retries: int = 2) -> float:
        """The relay's account-wide ``total_usage`` (GET /dashboard/billing/usage)."""
        if self.offline:
            return float("nan")
        last: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                r = self._http.get("/dashboard/billing/usage", timeout=60.0)
                if r.status_code == 200:
                    return float(r.json()["total_usage"])
                last = RelayError(f"usage endpoint HTTP {r.status_code}: {self._redact(r.text[:200])}")
            except (httpx.HTTPError, ValueError, KeyError) as e:
                last = RelayError(f"usage endpoint failed: {self._redact(e)}")
            time.sleep(1.0 + attempt)
        raise last if last else RelayError("usage endpoint failed")

    def wait_relay_usage_change(self, before: float, max_wait_s: float = 12.0, poll_s: float = 2.0) -> Tuple[float, float]:
        """Poll ``relay_usage`` until it differs from ``before`` (the counter lags a call by a few s).

        Returns ``(value, seconds_waited)``; the last polled value is returned when nothing moved.
        """
        t0 = _now()
        value = before
        while _now() - t0 < max_wait_s:
            time.sleep(poll_s)
            value = self.relay_usage()
            if value != before:
                break
        return value, _now() - t0

    def calibrate_units_per_token(
        self,
        model: str,
        message_lists: Sequence[Sequence[Dict[str, Any]]],
        *,
        tag: str,
        max_tokens: int = 256,
        extra: Optional[Dict[str, Any]] = None,
        settle_s: float = 15.0,
        poll_s: float = 1.0,
        confirm_s: float = 3.0,
        store: bool = True,
    ) -> Dict[str, Any]:
        """Estimate units/token for ``model`` from sequential live calls, robust to other consumers.

        For each message list (must be a cache miss to count): read the relay counter, make the call,
        poll the counter every ``poll_s`` until it moves (at most ``settle_s``), keep polling for
        ``confirm_s`` to see whether a second jump follows, and record ``delta / total_tokens``.
        Usage by concurrent consumers can only ADD to a window's delta, so the smallest
        windows are the least-contaminated; the estimate is the median of the "low cluster" (windows
        within 2x of the minimum, clean windows preferred).  Windows with a single jump are flagged
        ``clean``.  When ``store`` is true and the estimate is sane (``MIN_SANE_UNITS_PER_TOKEN`` ..
        ``MAX_SANE_UNITS_PER_TOKEN``) it is written to ``budget.json`` together with the per-window
        measurements and a ``note_calibration``.  Returns
        ``dict(estimate, n_windows, n_clean, per_call, stored, results, ...)`` where ``results`` are
        the ``chat`` result dicts (one per message list, cached ones included) so the calls can double
        as useful work (e.g. judge calls).
        """
        per_call: List[Dict[str, Any]] = []
        results: List[Dict[str, Any]] = []
        for msgs in message_lists:
            u0 = self.relay_usage()
            t0 = _now()
            res = self.chat(model, msgs, max_tokens=max_tokens, temperature=0, tag=tag, extra=extra)
            results.append(res)
            if res.get("cached"):
                per_call.append({"cached": True})
                continue
            tokens = int((res.get("usage") or {}).get("total_tokens") or 0)
            first: Optional[float] = None
            jumps = 0
            last = u0
            deadline = _now() + settle_s
            while _now() < deadline:
                time.sleep(poll_s)
                v = self.relay_usage()
                if v != last:
                    jumps += 1
                    last = v
                    if first is None:
                        first = v
                        deadline = min(deadline, _now() + confirm_s)  # confirm window after the first jump
            delta = (last - u0) if first is not None else None
            per_call.append({
                "tokens": tokens, "prompt_tokens": (res.get("usage") or {}).get("prompt_tokens"), "completion_tokens": (res.get("usage") or {}).get("completion_tokens"),
                "delta_units": delta, "first_jump_units": (first - u0) if first is not None else None, "jumps": jumps, "clean": jumps == 1,
                "rate": (delta / tokens) if (delta is not None and tokens) else None, "first_jump_rate": ((first - u0) / tokens) if (first is not None and tokens) else None,
                "latency_s": res.get("latency_s"), "window_s": round(_now() - t0, 1), "cached": False,
            })
        rates = [c["first_jump_rate"] for c in per_call if c.get("first_jump_rate") is not None and c["first_jump_rate"] > 0]
        clean_rates = [c["first_jump_rate"] for c in per_call if c.get("clean") and c.get("first_jump_rate")]
        pool = clean_rates or rates
        est = None
        low: List[float] = []
        if pool:
            # Contamination is additive and, on this relay, comes in jumps of tens to thousands of units, so windows within 2x of the
            # minimum form the uncontaminated cluster; its median is more stable than the raw minimum (which mixes in the
            # prompt/completion price difference of the cheapest call).
            lo = min(pool)
            low = sorted(r for r in pool if r <= 2.0 * lo)
            est = low[len(low) // 2]
        out: Dict[str, Any] = {"model": model, "estimate": est, "estimate_min": (min(pool) if pool else None), "low_cluster": low,
                               "estimate_basis": ("median of the low cluster of clean windows" if clean_rates else ("median of the low cluster of all windows (none clean)" if rates else "no live windows")),
                               "n_windows": sum(1 for c in per_call if not c.get("cached")), "n_clean": len(clean_rates), "per_call": per_call, "stored": False,
                               "results": results}
        if store and est is not None and MIN_SANE_UNITS_PER_TOKEN <= est <= MAX_SANE_UNITS_PER_TOKEN:
            self.set_units_per_token(model, est, meta={"method": "sequential windows; median of the low cluster (within 2x of the minimum)", "estimate_basis": out["estimate_basis"],
                                                        "estimate_min": out["estimate_min"], "low_cluster": low, "n_windows": out["n_windows"], "n_clean": out["n_clean"],
                                                        "per_call": [{k: v for k, v in c.items() if k != "cached"} for c in per_call if not c.get("cached")], "tag": tag})
            budget = self.load_budget()
            budget["note_calibration"] = ("units_per_token_by_model measured by PackyClient.calibrate_units_per_token (relay total_usage polled after sequential calls; "
                                          "windows carrying concurrent consumers' usage are excluded via the low cluster).")
            self._write_budget(budget)
            out["stored"] = True
        return out

    # ------------------------------------------------------------------ internals

    def _error_result(self, model: str, err: Exception) -> Dict[str, Any]:
        """Result dict describing a failed item in ``map``."""
        return {
            "text": None,
            "reasoning": None,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0},
            "latency_s": None,
            "cached": False,
            "model": model,
            "finish_reason": "error",
            "error": self._redact(err),
        }

    def _record(self, model: str, tag: str, endpoint: str, usage: Dict[str, int], latency_s: float, cached: bool, units_est: float) -> None:
        """Append a ledger row and update the in-memory session statistics."""
        self._ledger_append(
            {
                "ts": _now(),
                "tag": tag,
                "model": model,
                "endpoint": endpoint,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "reasoning_tokens": usage.get("reasoning_tokens", 0),
                "latency_s": round(float(latency_s), 4),
                "cached": bool(cached),
                "units_est": round(float(units_est), 6),
            }
        )
        with self._stats_lock:
            s = self._session[model]
            s["calls"] += 1
            s["cached_hits" if cached else "live_calls"] += 1
            if not cached:
                s["prompt_tokens"] += usage.get("prompt_tokens", 0)
                s["completion_tokens"] += usage.get("completion_tokens", 0)
                s["reasoning_tokens"] += usage.get("reasoning_tokens", 0)
                s["units_est"] += units_est
                s["latency_s_total"] += float(latency_s)

    def _execute(
        self,
        endpoint: str,
        model: str,
        body: Dict[str, Any],
        *,
        tag: str,
        timeout: Optional[float],
        parse: Callable[[Dict[str, Any]], Dict[str, Any]],
        substituted_for: Optional[str],
    ) -> Dict[str, Any]:
        """Cache lookup -> budget check -> rate-capped live call -> cache store -> ledger row."""
        key = cache_key(model, body)
        hit = self._cache_get(key)
        if hit is not None:
            result = parse(hit["response"])
            result.update(cached=True, latency_s=hit["latency_s"], model=model, endpoint=endpoint, cache_key=key, retries=0)
            if substituted_for:
                result["substituted_for"] = substituted_for
            self._record(model, tag, endpoint, result["usage"], hit["latency_s"] or 0.0, cached=True, units_est=0.0)
            return result

        if self.offline:
            raise PackyError("Response is absent from the local cache; use explicit collection mode for new calls")
        self._check_budget()
        timeout_s = float(timeout) if timeout is not None else self.timeout_for(model)
        sem = self._semaphore(model)
        with sem:
            with self._sem_lock:
                self._inflight[model] += 1
                self._max_inflight[model] = max(self._max_inflight[model], self._inflight[model])
            try:
                t0 = _now()
                js, retries = self._call_with_retries(endpoint, model, body, timeout_s, parse)
                latency_s = _now() - t0
            finally:
                with self._sem_lock:
                    self._inflight[model] -= 1
        result = parse(js)
        self._cache_put(key, model, endpoint, body, js, result["usage"], latency_s)
        units_est = result["usage"].get("total_tokens", 0) * self.units_per_token(model)
        result.update(cached=False, latency_s=latency_s, model=model, endpoint=endpoint, cache_key=key, retries=retries)
        if substituted_for:
            result["substituted_for"] = substituted_for
        self._record(model, tag, endpoint, result["usage"], latency_s, cached=False, units_est=units_est)
        return result

    def _call_with_retries(
        self,
        endpoint: str,
        model: str,
        body: Dict[str, Any],
        timeout_s: float,
        parse: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], int]:
        """POST with retries on timeouts / 429 / 5xx / empty content; returns ``(json, retries)``."""
        last_err: Optional[Exception] = None
        last_js: Optional[Dict[str, Any]] = None
        for attempt in range(self.max_retries + 1):
            if attempt:
                delay = min(60.0, 2.0 * (2 ** (attempt - 1))) + random.uniform(0.0, 1.0)
                time.sleep(delay)
            try:
                r = self._http.post("/" + endpoint, json=body, timeout=httpx.Timeout(timeout_s, connect=30.0))
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_err = RelayError(f"{model}: {type(e).__name__} after {timeout_s:.0f}s (attempt {attempt + 1}/{self.max_retries + 1})")
                continue
            if r.status_code in RETRY_STATUS:
                last_err = RelayError(f"{model}: HTTP {r.status_code} (attempt {attempt + 1}): {self._redact(r.text[:300])}")
                continue
            if r.status_code == 403:
                raise ModelUnavailable(f"{model}: HTTP 403 -- {self._redact(r.text[:300])}")
            if r.status_code == 400 and "incorrect api key" in r.text.lower():
                last_err = RelayError(f"{model}: transient HTTP 400 auth flake (attempt {attempt + 1}): {self._redact(r.text[:200])}")
                continue
            if r.status_code >= 400:
                raise RelayError(f"{model}: HTTP {r.status_code} -- {self._redact(r.text[:500])}")
            try:
                js = r.json()
            except ValueError:
                last_err = RelayError(f"{model}: non-JSON body (attempt {attempt + 1}): {self._redact(r.text[:200])}")
                continue
            if isinstance(js, dict) and js.get("error") and not js.get("choices") and not js.get("output"):
                err = js["error"]
                msg = self._redact(err.get("message") if isinstance(err, dict) else err)
                code = str((err.get("code") if isinstance(err, dict) else "") or "")
                if "rate" in msg.lower() or code in {"429", "rate_limit_exceeded", "server_error", "overloaded"}:
                    last_err = RelayError(f"{model}: relay error (attempt {attempt + 1}): {msg[:300]}")
                    continue
                raise RelayError(f"{model}: relay error: {msg[:500]}")
            parsed = parse(js)
            last_js = js
            if not (parsed["text"] or "").strip():
                if parsed.get("finish_reason") in ("length", "incomplete:max_output_tokens"):
                    return js, attempt  # deterministic truncation: content budget consumed by reasoning
                last_err = _EmptyContent(f"{model}: empty content (finish_reason={parsed.get('finish_reason')!r}, attempt {attempt + 1})")
                continue
            return js, attempt
        if last_js is not None:  # exhausted retries on empty content: return the last body as-is
            return last_js, self.max_retries
        raise RelayError(f"{model}: giving up after {self.max_retries + 1} attempts: {self._redact(last_err)}")


__all__ = [
    "PackyClient",
    "PackyError",
    "BudgetExceeded",
    "ModelUnavailable",
    "RelayError",
    "CONCURRENCY",
    "TIMEOUTS_S",
    "DEFAULT_TIMEOUT_S",
    "MIN_MAX_TOKENS",
    "RESPONSES_ONLY_MODELS",
    "SUBSTITUTES",
    "DEFAULT_UNITS_PER_TOKEN",
    "cache_key",
    "canonical_json",
    "parse_chat_response",
    "parse_responses_response",
]

