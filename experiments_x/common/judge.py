"""LLM-as-judge for code responses through ``common.packy``.

* ``judge_prompt(task_prompt, response_text, language)`` -> strict grader prompt asking for JSON
  ``{"verdict": "correct"|"incorrect", "confidence": 0-1, "reason": "<=30 words"}``.
* ``judge(client, task, response_text, model='deepseek-v4-pro', tag=...)`` -> ``dict(p_correct,
  verdict, confidence, reason, parse_ok, raw, usage, latency_s, cached, model)`` where
  ``p_correct = confidence`` if the verdict is ``correct`` else ``1 - confidence`` (0.5 when the
  reply could not be parsed).
* ``judge_many(client, items, model, workers, tag)`` -> list of the same dicts, run through
  ``client.map`` under the model's concurrency cap.
* ``parse_judge_json(text)`` -> robust JSON extraction (whole text, first ``{...}`` containing
  ``verdict``, then regex fallbacks for the individual fields).

The default judge is deepseek-v4-pro with reasoning disabled (``extra={"thinking": {"type":
"disabled"}}``; without it the model spends its whole ``max_tokens`` budget on reasoning and the
visible content is empty at small budgets).  Pass ``thinking=True`` to keep reasoning on (the
``max_tokens`` floor is then raised to 1024).  Prompts and responses longer than ``max_chars`` are
truncated in the middle (with a marker) to bound the cost.  Nothing happens at import time.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

SYSTEM_PROMPT = (
    "You are a strict, meticulous automated grader for programming tasks. You judge functional "
    "correctness only, you mentally execute edge cases, and you never give credit for explanations, "
    "partial solutions or code that solves a different problem. You answer with a single JSON object."
)

_USER_TEMPLATE = """Grade the candidate response to the task below.

Language: {language}

### Task
{task_prompt}

### Candidate response
{response_text}

### Grading conventions (apply strictly, in this order)
1. The response is graded by running it as a standalone program against hidden tests. It must contain the COMPLETE definition of the required function or class with exactly the name and signature given in the task (the `def`/`class` line, the whole body, and any imports it needs). A bare function body without its `def` line, an indented fragment, pseudo-code, or prose without code is INCORRECT even when the logic would be right.
2. Code that stops before the function is finished, that ends mid-statement, that has a code fence glued to code, stray characters, or anything else that would not parse is INCORRECT.
3. Mentally execute the examples given in the task and consider edge cases (empty input, one element, negative numbers, duplicates, ties, off-by-one, large values) before deciding. Solving a different problem, a wrong return type or output format, or reading input differently from what the task specifies is INCORRECT.
4. Ignore code style, comments, efficiency and extra helper code unless the task requires them. An empty response is INCORRECT.

Reply with ONLY this JSON object and nothing else (no markdown, no code fence):
{{"verdict": "correct" | "incorrect", "confidence": <number between 0 and 1>, "reason": "<at most 30 words>"}}"""

DEFAULT_JUDGE_MODEL = "deepseek-v4-pro"
DEFAULT_MAX_TOKENS = 320
DEFAULT_MAX_CHARS = 12000
THINKING_OFF_EXTRA: Dict[str, Any] = {"thinking": {"type": "disabled"}}
_THINKING_MODELS_PREFIX = ("deepseek",)

_JSON_OBJ_RE = re.compile(r"\{[^{}]*\"verdict\"[^{}]*\}", re.S)
_VERDICT_RE = re.compile(r"[\"']?verdict[\"']?\s*[:=]\s*[\"']?\s*(correct|incorrect)", re.I)
_CONF_RE = re.compile(r"[\"']?confidence[\"']?\s*[:=]\s*([0-9]*\.?[0-9]+)\s*(%?)", re.I)
_REASON_RE = re.compile(r"[\"']?reason[\"']?\s*[:=]\s*[\"']([^\"']{0,400})", re.I)


def _truncate(s: str, max_chars: int) -> str:
    """Middle-truncate ``s`` to ``max_chars`` with an explicit marker."""
    s = s or ""
    if len(s) <= max_chars:
        return s
    half = max_chars // 2
    return s[:half] + f"\n... [{len(s) - max_chars} characters omitted] ...\n" + s[-half:]


def judge_prompt(task_prompt: str, response_text: str, language: str = "python", max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Build the strict grader prompt (the user turn; ``SYSTEM_PROMPT`` is the system turn)."""
    return _USER_TEMPLATE.format(
        language=language or "python",
        task_prompt=_truncate(task_prompt or "(no task text)", max_chars),
        response_text=_truncate(response_text if (response_text or "").strip() else "(empty response)", max_chars),
    )


def judge_messages(task: Dict[str, Any], response_text: str, max_chars: int = DEFAULT_MAX_CHARS) -> List[Dict[str, str]]:
    """Chat messages (system + user) for judging ``response_text`` against ``task``."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": judge_prompt(task.get("prompt") or task.get("raw_prompt") or "", response_text, task.get("language") or "python", max_chars)},
    ]


def _clamp01(x: float) -> float:
    """Clamp to [0, 1]."""
    return max(0.0, min(1.0, float(x)))


def parse_judge_json(text: Optional[str]) -> Dict[str, Any]:
    """Extract ``verdict`` / ``confidence`` / ``reason`` from a grader reply; ``parse_ok`` flags success.

    Order: ``json.loads`` of the whole text (also after stripping a code fence), the first JSON
    object containing ``"verdict"``, then regexes for the individual fields.  Confidence given as a
    percentage (``85%`` or a bare number > 1) is rescaled; missing confidence defaults to 0.5.
    """
    out: Dict[str, Any] = {"verdict": None, "confidence": None, "reason": None, "parse_ok": False}
    if not text or not str(text).strip():
        return out
    s = str(text).strip()
    candidates: List[str] = [s]
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    if fence:
        candidates.append(fence.group(1).strip())
    candidates += _JSON_OBJ_RE.findall(s)
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict) and "verdict" in obj:
            v = str(obj.get("verdict", "")).strip().lower()
            if v in ("correct", "incorrect"):
                out["verdict"] = v
                try:
                    c = float(obj.get("confidence", 0.5))
                except (TypeError, ValueError):
                    c = 0.5
                out["confidence"] = _clamp01(c / 100.0 if c > 1.0 else c)
                out["reason"] = (str(obj.get("reason")) if obj.get("reason") is not None else None)
                out["parse_ok"] = True
                return out
    m = _VERDICT_RE.search(s)
    if m:
        out["verdict"] = m.group(1).lower()
        mc = _CONF_RE.search(s)
        c = 0.5
        if mc:
            try:
                c = float(mc.group(1))
                if mc.group(2) == "%" or c > 1.0:
                    c = c / 100.0
            except ValueError:
                c = 0.5
        out["confidence"] = _clamp01(c)
        mr = _REASON_RE.search(s)
        out["reason"] = mr.group(1).strip() if mr else None
        out["parse_ok"] = True
        return out
    low = s.lower()
    if re.search(r"\bincorrect\b", low) and not re.search(r"\bcorrect\b(?!\s*\|)", low.replace("incorrect", "")):
        out.update(verdict="incorrect", confidence=0.5, parse_ok=True, reason="(regex fallback)")
    elif re.search(r"\bcorrect\b", low) and "incorrect" not in low:
        out.update(verdict="correct", confidence=0.5, parse_ok=True, reason="(regex fallback)")
    return out


def _to_result(parsed: Dict[str, Any], call: Dict[str, Any]) -> Dict[str, Any]:
    """Combine a parsed verdict with the packy call record into the judge result dict."""
    if parsed.get("parse_ok") and parsed.get("verdict") in ("correct", "incorrect"):
        conf = float(parsed.get("confidence") if parsed.get("confidence") is not None else 0.5)
        p = conf if parsed["verdict"] == "correct" else 1.0 - conf
    else:
        conf = None
        p = 0.5
    return {
        "p_correct": _clamp01(p),
        "verdict": parsed.get("verdict") or "unknown",
        "confidence": conf,
        "reason": parsed.get("reason"),
        "parse_ok": bool(parsed.get("parse_ok")),
        "raw": call.get("text"),
        "reasoning": call.get("reasoning"),
        "usage": call.get("usage"),
        "latency_s": call.get("latency_s"),
        "cached": call.get("cached"),
        "model": call.get("model"),
        "finish_reason": call.get("finish_reason"),
        "error": call.get("error"),
    }


def _call_kwargs(model: str, max_tokens: int, thinking: bool, extra: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Keyword arguments for ``client.chat`` / ``client.map`` implementing the thinking policy."""
    kw: Dict[str, Any] = {"max_tokens": int(max_tokens), "temperature": 0}
    ex: Dict[str, Any] = {}
    if not thinking and model.lower().startswith(_THINKING_MODELS_PREFIX):
        ex.update(THINKING_OFF_EXTRA)
    if thinking:
        kw["max_tokens"] = max(kw["max_tokens"], 1024)
    if extra:
        ex.update(extra)
    if ex:
        kw["extra"] = ex
    return kw


def judge_call_kwargs(model: str = DEFAULT_JUDGE_MODEL, *, max_tokens: int = DEFAULT_MAX_TOKENS, thinking: bool = False, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Public form of the keyword arguments ``judge`` passes to ``client.chat`` (for callers that batch the calls themselves)."""
    return _call_kwargs(model, max_tokens, thinking, extra)


def judge_result_from_call(call: Dict[str, Any]) -> Dict[str, Any]:
    """Turn a raw ``client.chat`` result for a judge prompt into the ``judge`` result dict."""
    return _to_result(parse_judge_json(call.get("text")), call)


def judge(client: Any, task: Dict[str, Any], response_text: str, model: str = DEFAULT_JUDGE_MODEL, tag: str = "judge", *,
          max_tokens: int = DEFAULT_MAX_TOKENS, thinking: bool = False, extra: Optional[Dict[str, Any]] = None, max_chars: int = DEFAULT_MAX_CHARS) -> Dict[str, Any]:
    """Judge one response; returns ``dict(p_correct, verdict, confidence, reason, parse_ok, raw, ...)``.

    ``p_correct`` is the judge's confidence when the verdict is ``correct`` and ``1 - confidence``
    otherwise (0.5 if the reply could not be parsed).  ``client`` is a ``common.packy.PackyClient``.
    """
    msgs = judge_messages(task, response_text, max_chars)
    call = client.chat(model, msgs, tag=tag, **_call_kwargs(model, max_tokens, thinking, extra))
    return _to_result(parse_judge_json(call.get("text")), call)


def judge_many(client: Any, items: Sequence[Union[Tuple[Dict[str, Any], str], Dict[str, Any]]], model: str = DEFAULT_JUDGE_MODEL, workers: Optional[int] = None,
               tag: str = "judge", *, max_tokens: int = DEFAULT_MAX_TOKENS, thinking: bool = False, extra: Optional[Dict[str, Any]] = None, max_chars: int = DEFAULT_MAX_CHARS) -> List[Dict[str, Any]]:
    """Judge many ``(task, response_text)`` pairs (or dicts with those keys) via ``client.map``.

    Results keep the input order.  Items whose relay call failed carry ``verdict='unknown'``,
    ``p_correct=0.5`` and the ``error`` text.
    """
    pairs: List[Tuple[Dict[str, Any], str]] = []
    for it in items:
        if isinstance(it, dict):
            pairs.append((it["task"], it.get("response_text", it.get("response", ""))))
        else:
            pairs.append((it[0], it[1]))
    msg_lists = [judge_messages(t, r, max_chars) for t, r in pairs]
    calls = client.map(model, msg_lists, workers=workers, tag=tag, **_call_kwargs(model, max_tokens, thinking, extra))
    return [_to_result(parse_judge_json(c.get("text")), c) for c in calls]


__all__ = ["SYSTEM_PROMPT", "DEFAULT_JUDGE_MODEL", "DEFAULT_MAX_TOKENS", "THINKING_OFF_EXTRA", "judge_prompt", "judge_messages", "parse_judge_json",
           "judge_call_kwargs", "judge_result_from_call", "judge", "judge_many"]

