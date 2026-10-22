"""Sandboxed execution of Python solutions against the tests reconstructed by ``common.sources``.

Public API
----------
* ``run_python_tests(code, tests, *, entry_point=None, timeout_s=10, mem_mb=1024)`` ->
  ``dict(passed, frac, stdout, stderr, error, elapsed_s, n_cases, n_passed, timed_out, returncode)``
  - ``tests`` is a string (HumanEval / MBPP / BigCodeBench / HumanEval-X style): the sandbox runs
    ``code + "\\n" + tests``; if ``entry_point`` is given and the tests define ``check`` without
    calling it, ``check(<entry_point>)`` is appended; ``unittest.TestCase`` suites without a
    ``unittest.main()`` call get a runner appended and ``frac`` = share of passing test methods.
  - ``tests`` is a dict ``{type: 'stdin', public: [...], private: [...]}`` (LiveCodeBench): each
    case is run as a separate process with the case's ``input`` on stdin and the stdout is compared
    with ``output`` (line-wise after stripping trailing whitespace, then token-wise with a 1e-6
    float tolerance); ``frac`` = pass fraction, ``passed`` = all cases pass.
  - ``tests`` is a dict ``{type: 'functional', func_name, public, private}`` (LeetCode-style):
    one harness process calls ``Solution().<func_name>(*args)`` (or a bare function of that name)
    per case, with the newline-separated JSON arguments of ``input`` and the JSON ``output``.
* ``extract_code(response_text, language='python', entry_point=None)`` -> code string: the last
  fenced block of the language (preferring blocks that define ``entry_point``, else blocks with a
  definition), else the whole text with leading explanations stripped and trailing prose trimmed
  until it compiles.  DS-1000 markers (``<code>``, ``BEGIN/END SOLUTION``) are removed.
* ``extract_prediction(response_text)`` -> literal predicted by an output-prediction response
  (CRUXEval): ``Output: `x``` phrases, the last non-code fenced block, or the last back-ticked span.
* ``verify_response(task, response_text, **kw)`` -> ``run_python_tests`` result plus ``code``,
  ``kind`` (harness type), ``failure_kind`` (``passed`` / ``empty_response`` / ``timeout`` /
  ``extraction_failure`` / ``missing_lib`` / ``harness_mismatch`` / ``assertion`` /
  ``runtime_error`` / ``not_runnable``).  An empty response fails without running anything.
  Source-specific assembly: BigCodeBench / HumanEval imports from the prompt are prepended to the
  extracted code; CRUXEval compares the predicted literal with ``f(<input>)``; DS-1000 runs the
  official ``code_context`` (``test_execution(solution)`` and ``test_string(solution)``);
  LiveCodeBench prepends the usual ``typing`` / ``collections`` / ``itertools`` imports.

Isolation
---------
Every run happens in a fresh temporary directory (cwd) with ``python -c`` executing a bootstrap
that sets ``resource`` limits inside the child (``RLIMIT_CPU`` = timeout + 1 s, ``RLIMIT_AS`` /
``RLIMIT_DATA`` = ``mem_mb``, ``RLIMIT_FSIZE`` = 64 MB) and then runs ``main.py``.  Setting the
limits in the child (instead of ``preexec_fn``) keeps the launcher thread-safe.  The child gets a
minimal environment: ``PATH``, ``PYTHONHASHSEED=0`` plus, so that plotting tasks run headless and
write nothing outside the sandbox, ``MPLBACKEND=Agg``, ``HOME`` / ``TMPDIR`` = the temp dir,
``MPLCONFIGDIR`` = the persistent ``common/cache/mplconfig`` (so matplotlib's font cache is built
once, not per run), ``PYTHONDONTWRITEBYTECODE=1``, ``PYTHONIOENCODING=utf-8`` and single-threaded
BLAS variables.  The process group is killed on wall-clock timeout.  Note: macOS does not enforce
``RLIMIT_AS`` / ``RLIMIT_DATA`` for Python allocations (verified: a 2 GB allocation succeeds under
a 1 GB limit); the CPU limit is enforced.  No network isolation is attempted.
"""
from __future__ import annotations

import ast
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

PYTHON = sys.executable
MAX_CAPTURE = 20000
EX_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Persistent matplotlib config/font cache shared by all sandboxes (a fresh MPLCONFIGDIR per run makes
# matplotlib rebuild its font cache, ~10 s per plotting task).  Lives under common/cache, the only
# shared directory experiments may write to.
MPL_CONFIG_DIR = os.environ.get("EX_SANDBOX_MPLCONFIGDIR") or os.path.join(EX_DIR, "common", "cache", "mplconfig")
DEFAULT_TIMEOUTS: Dict[str, float] = {"humaneval": 10, "he": 10, "mbpp": 10, "hex": 10, "cruxeval": 10, "bcb": 30, "ds1000": 30, "lcb": 6}
DEFAULT_LCB_MAX_CASES = 100  # LiveCodeBench has at most 100 cases per task, so this runs them all

_BOOTSTRAP = (
    "import resource, sys, runpy\n"
    "def _lim(name, val):\n"
    "    r = getattr(resource, name, None)\n"
    "    if r is None: return\n"
    "    try: resource.setrlimit(r, (val, val))\n"
    "    except (ValueError, OSError): pass\n"
    "_lim('RLIMIT_CPU', {cpu})\n_lim('RLIMIT_AS', {mem})\n_lim('RLIMIT_DATA', {mem})\n_lim('RLIMIT_FSIZE', {fsize})\n"
    "sys.argv = ['main.py']\n"
    "runpy.run_path('main.py', run_name='__main__')\n"
)

_UNITTEST_RUNNER = '''

if __name__ == "__main__":
    import unittest as _ut, sys as _sys, json as _json
    _suite = _ut.defaultTestLoader.loadTestsFromModule(_sys.modules["__main__"])
    _res = _ut.TextTestRunner(stream=_sys.stderr, verbosity=1).run(_suite)
    print("__SANDBOX_UNITTEST__ " + _json.dumps({"run": _res.testsRun, "failed": len(_res.failures) + len(_res.errors)}))
    _sys.exit(0 if _res.wasSuccessful() else 1)
'''

LCB_HEADER = (
    "import sys\nsys.setrecursionlimit(1000000)\n"
    "from typing import *\nimport math, collections, itertools, heapq, bisect, functools, string, re, random\n"
    "from collections import *\nfrom itertools import *\nfrom heapq import *\nfrom bisect import *\nfrom functools import *\n\n"
)

_FUNCTIONAL_HARNESS = '''

import json as _json, sys as _sys, traceback as _tb, math as _math, ast as _ast
_FUNC = {func!r}
_CASES = _json.loads({cases_json!r})
def _norm(x):
    if isinstance(x, (list, tuple)): return [_norm(v) for v in x]
    if isinstance(x, dict): return {{str(k): _norm(v) for k, v in x.items()}}
    if isinstance(x, set): return sorted((_norm(v) for v in x), key=repr)
    return x
def _same(a, b):
    a, b = _norm(a), _norm(b)
    if a == b: return True
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return _math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6)
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return all(_same(x, y) for x, y in zip(a, b))
    return False
def _loads(s):
    try: return _json.loads(s)
    except Exception:
        try: return _ast.literal_eval(s)
        except Exception: return s.strip()
def _parse_args(s):
    return [_loads(l) for l in s.strip().split("\\n") if l.strip() != ""]
def _resolve():
    g = globals()
    if "Solution" in g:
        obj = g["Solution"]()
        if _FUNC and hasattr(obj, _FUNC): return getattr(obj, _FUNC)
        meths = [m for m in dir(obj) if not m.startswith("_") and callable(getattr(obj, m))]
        if len(meths) == 1: return getattr(obj, meths[0])
    if _FUNC and _FUNC in g and callable(g[_FUNC]): return g[_FUNC]
    raise NameError("no Solution.%s / function %s defined" % (_FUNC, _FUNC))
_passed = 0
for _c in _CASES:
    try:
        _args = _parse_args(_c["input"])
        _exp = _loads(_c["output"])
        _got = _resolve()(*_args)
        _ok = _same(_got, _exp) or (str(_got).strip() == str(_c["output"]).strip()) or (_json.dumps(_norm(_got)) == str(_c["output"]).strip())
    except BaseException:
        _ok = False
        _tb.print_exc(limit=2, file=_sys.stderr)
    _passed += bool(_ok)
print("__SANDBOX_CASES__ " + _json.dumps({{"run": len(_CASES), "passed": _passed}}))
_sys.exit(0 if _passed == len(_CASES) else 1)
'''

_CRUXEVAL_HARNESS = '''

import ast as _ast, sys as _sys
_pred_src = {pred!r}
_got = f({inp})
_SENTINEL = object()
_pred = _SENTINEL
try:
    _pred = _ast.literal_eval(_pred_src)
except Exception:
    try:  # simple expressions such as 'abc'.upper() or [1, 2] + [3]; no builtins, inside the sandbox
        _pred = eval(_pred_src, {{"__builtins__": {{}}}}, {{}})
    except Exception:
        _pred = _SENTINEL
if _pred is not _SENTINEL:
    _ok = (_pred == _got)
else:
    _s = _pred_src.strip()
    _ok = _s == repr(_got) or _s == str(_got) or _s.strip("'\\"") == str(_got)
if not _ok:
    print("PRED:", _pred_src[:300])
    print("GOT:", repr(_got)[:300])
    raise AssertionError("prediction mismatch")
print("OK")
'''

_FENCE_RE = re.compile(r"```[ \t]*([\w+#.\-]*)[^\n]*\n(.*?)(?:```|\Z)", re.S)
_LANG_TAGS = {"python": {"python", "py", "python3", "", "text", "plaintext"}}
_CODE_LINE_RE = re.compile(
    r"^(from\s+\S+\s+import\b|import\s+\S+|def\s|class\s|async\s+def\s|@\w|#|[A-Za-z_][\w\.\[\]\", ']*\s*(=|\+=|-=|\*=|/=)[^=]|if\s|for\s|while\s|try:|with\s|print\(|return\b|result\s*=|plt\.|df\b|np\.)"
)
_TRACEBACK_LAST_RE = re.compile(r"^(\w+(?:\.\w+)*(?:Error|Exception|Exit|Interrupt|Warning)?)(?::\s*(.*))?$", re.M)


# ---------------------------------------------------------------------------------------- helpers


def _decode(b: Optional[bytes]) -> str:
    """Decode captured bytes (utf-8, replacement) in full; comparisons use the full text."""
    if not b:
        return ""
    return b.decode("utf-8", errors="replace")


def _cap(s: str) -> str:
    """Middle-truncate captured text to ``MAX_CAPTURE`` for storage in result dicts."""
    if len(s) > MAX_CAPTURE:
        return s[: MAX_CAPTURE // 2] + f"\n... [{len(s) - MAX_CAPTURE} chars truncated] ...\n" + s[-MAX_CAPTURE // 2:]
    return s


def _child_env(tmp: str) -> Dict[str, str]:
    """Minimal environment for the sandboxed interpreter (see module docstring)."""
    try:
        os.makedirs(MPL_CONFIG_DIR, exist_ok=True)
        mpl_dir = MPL_CONFIG_DIR
    except OSError:
        mpl_dir = tmp
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": mpl_dir,
        "HOME": tmp,
        "TMPDIR": tmp,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }


def _kill_group(p: subprocess.Popen) -> None:
    """Kill the child's whole process group (fallback: the child alone)."""
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            p.kill()
        except OSError:
            pass


def _exec(tmp: str, stdin_text: Optional[str], timeout_s: float, mem_mb: int) -> Dict[str, Any]:
    """Run ``main.py`` in ``tmp`` under the bootstrap with limits; returns the raw process result."""
    boot = _BOOTSTRAP.format(cpu=int(math.ceil(timeout_s)) + 1, mem=int(mem_mb) * 1024 * 1024, fsize=64 * 1024 * 1024)
    t0 = time.time()
    p = subprocess.Popen(
        [PYTHON, "-c", boot], cwd=tmp, env=_child_env(tmp), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, start_new_session=True,
    )
    timed_out = False
    try:
        out, err = p.communicate(input=(stdin_text or "").encode("utf-8", errors="replace"), timeout=timeout_s + 1.0)
    except subprocess.TimeoutExpired:
        _kill_group(p)
        try:
            out, err = p.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            out, err = b"", b""
        timed_out = True
    return {"returncode": p.returncode, "stdout": _decode(out), "stderr": _decode(err), "timed_out": timed_out, "elapsed_s": round(time.time() - t0, 4)}


def _write_main(tmp: str, script: str) -> None:
    """Write the script (NUL bytes removed) to ``main.py`` in the sandbox dir."""
    with open(os.path.join(tmp, "main.py"), "w", encoding="utf-8") as fh:
        fh.write(script.replace("\x00", ""))


def _hoist_future(code: str) -> str:
    """Move ``from __future__ import ...`` lines to the top so a prepended header stays legal."""
    lines = code.split("\n")
    fut = [l for l in lines if re.match(r"^\s*from\s+__future__\s+import", l)]
    if not fut:
        return code
    rest = [l for l in lines if not re.match(r"^\s*from\s+__future__\s+import", l)]
    return "\n".join(fut + rest)


def classify_error(stderr: str, timed_out: bool = False, returncode: Optional[int] = None) -> Optional[str]:
    """Short label for the failure visible in ``stderr`` (``None`` when nothing is wrong)."""
    if timed_out:
        return "timeout"
    if returncode is not None and returncode < 0:
        sig = -returncode
        return f"killed by signal {sig}" + (" (CPU limit)" if sig in (signal.SIGXCPU, signal.SIGKILL) else "")
    lines = [l.strip() for l in (stderr or "").strip().split("\n") if l.strip()]
    for line in reversed(lines):
        m = _TRACEBACK_LAST_RE.match(line)
        if m and (m.group(1).endswith(("Error", "Exception", "Exit", "Interrupt")) or m.group(1) in ("AssertionError", "SystemExit")):
            msg = (m.group(2) or "").strip()
            return f"{m.group(1)}: {msg[:160]}" if msg else m.group(1)
    if any(l.startswith("FAILED") for l in lines):
        return next(l for l in lines if l.startswith("FAILED"))[:200]
    if returncode not in (None, 0):
        return f"exit code {returncode}"
    return None


def _stdout_matches(got: str, expected: str) -> bool:
    """LiveCodeBench-style stdout comparison: exact lines, else whitespace tokens with float tolerance."""

    def norm_lines(s: str) -> List[str]:
        lines = [l.rstrip() for l in s.replace("\r\n", "\n").split("\n")]
        while lines and lines[-1] == "":
            lines.pop()
        while lines and lines[0] == "":
            lines.pop(0)
        return lines

    if norm_lines(got) == norm_lines(expected):
        return True
    gt, et = got.split(), expected.split()
    if len(gt) != len(et):
        return False
    for a, b in zip(gt, et):
        if a == b:
            continue
        try:
            if math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6):
                continue
        except ValueError:
            pass
        return False
    return True


def _result(passed: bool, frac: float, r: Dict[str, Any], *, n_cases: int = 1, n_passed: Optional[int] = None, error: Optional[str] = None) -> Dict[str, Any]:
    """Assemble the public result dict from a raw process result."""
    return {
        "passed": bool(passed), "frac": float(frac), "stdout": _cap(r.get("stdout", "")), "stderr": _cap(r.get("stderr", "")),
        "error": error if error is not None else (None if passed else classify_error(r.get("stderr", ""), r.get("timed_out", False), r.get("returncode"))),
        "elapsed_s": round(float(r.get("elapsed_s", 0.0)), 4), "n_cases": n_cases,
        "n_passed": int(n_passed if n_passed is not None else (n_cases if passed else 0)),
        "timed_out": bool(r.get("timed_out", False)), "returncode": r.get("returncode"),
    }


# --------------------------------------------------------------------------------------- runners


def _run_script(code: str, tests: str, entry_point: Optional[str], timeout_s: float, mem_mb: int, tmp: str) -> Dict[str, Any]:
    """HumanEval / MBPP / BigCodeBench / HumanEval-X style: run ``code + tests`` once."""
    script = _hoist_future(code).rstrip() + "\n\n" + (tests or "").rstrip() + "\n"
    if entry_point and re.search(r"^\s*def\s+check\s*\(", tests or "", re.M) and not re.search(r"^check\s*\(", tests or "", re.M):
        script += f"\ncheck({entry_point})\n"
    unittest_suite = bool(re.search(r"class\s+\w+\s*\(\s*(?:unittest\.)?TestCase\s*\)", tests or "")) and "unittest.main" not in (tests or "")
    if unittest_suite:
        script += _UNITTEST_RUNNER
    _write_main(tmp, script)
    r = _exec(tmp, None, timeout_s, mem_mb)
    ok = (not r["timed_out"]) and r["returncode"] == 0
    n_cases, n_passed = 1, int(ok)
    frac = 1.0 if ok else 0.0
    m = re.search(r"__SANDBOX_UNITTEST__ (\{.*\})", r["stdout"])
    if m:
        try:
            st = json.loads(m.group(1))
            n_cases = max(int(st.get("run", 0)), 1)
            n_passed = max(n_cases - int(st.get("failed", 0)), 0) if not r["timed_out"] else 0
            frac = n_passed / n_cases
        except (ValueError, TypeError):
            pass
    return _result(ok, frac, r, n_cases=n_cases, n_passed=n_passed)


def _run_stdin_cases(code: str, cases: Sequence[Dict[str, Any]], timeout_s: float, mem_mb: int, tmp: str) -> Dict[str, Any]:
    """LiveCodeBench stdin/stdout: one process per case, pass fraction over cases."""
    _write_main(tmp, LCB_HEADER + _hoist_future(code))
    n_passed = 0
    first_fail: Optional[Dict[str, Any]] = None
    last: Dict[str, Any] = {"stdout": "", "stderr": "", "timed_out": False, "returncode": 0, "elapsed_s": 0.0}
    elapsed = 0.0
    for c in cases:
        r = _exec(tmp, str(c.get("input", "")), timeout_s, mem_mb)
        elapsed += r["elapsed_s"]
        ok = (not r["timed_out"]) and r["returncode"] == 0 and _stdout_matches(r["stdout"], str(c.get("output", "")))
        n_passed += int(ok)
        last = r
        if not ok and first_fail is None:
            first_fail = dict(r, expected=str(c.get("output", ""))[:500])
    n = len(cases)
    show = first_fail or last
    err = None
    if first_fail is not None:
        err = classify_error(first_fail["stderr"], first_fail["timed_out"], first_fail["returncode"]) or "wrong answer"
    res = _result(n_passed == n and n > 0, (n_passed / n) if n else 0.0, dict(show, elapsed_s=elapsed), n_cases=n, n_passed=n_passed, error=err)
    if first_fail is not None:
        res["expected"] = first_fail["expected"]
    return res


def _run_functional_cases(code: str, func_name: Optional[str], cases: Sequence[Dict[str, Any]], timeout_s: float, mem_mb: int, tmp: str) -> Dict[str, Any]:
    """LeetCode-style functional tests: one harness process calls the solution per case."""
    cases_json = json.dumps([{"input": str(c.get("input", "")), "output": str(c.get("output", ""))} for c in cases])
    script = LCB_HEADER + _hoist_future(code).rstrip() + _FUNCTIONAL_HARNESS.format(func=func_name or "", cases_json=cases_json)
    _write_main(tmp, script)
    r = _exec(tmp, None, timeout_s, mem_mb)
    n = len(cases)
    n_passed = 0
    m = re.search(r"__SANDBOX_CASES__ (\{.*\})", r["stdout"])
    if m:
        try:
            n_passed = int(json.loads(m.group(1)).get("passed", 0))
        except (ValueError, TypeError):
            n_passed = 0
    ok = (not r["timed_out"]) and r["returncode"] == 0 and n_passed == n and n > 0
    err = None if ok else (classify_error(r["stderr"], r["timed_out"], r["returncode"]) or "wrong answer")
    return _result(ok, (n_passed / n) if n else 0.0, r, n_cases=n, n_passed=n_passed, error=err)


def run_python_tests(code: str, tests: Any, *, entry_point: Optional[str] = None, timeout_s: float = 10, mem_mb: int = 1024, max_cases: Optional[int] = None) -> Dict[str, Any]:
    """Execute ``code`` against ``tests`` in a fresh sandbox directory (see module docstring).

    ``tests``: a test-script string, or a LiveCodeBench dict (``type`` 'stdin' | 'functional' with
    ``public`` / ``private`` case lists; ``max_cases`` caps the number of cases, public first).
    Returns ``dict(passed, frac, stdout, stderr, error, elapsed_s, n_cases, n_passed, timed_out,
    returncode)``.
    """
    tmp = tempfile.mkdtemp(prefix="sbx_")
    try:
        if isinstance(tests, dict):
            cases = list(tests.get("public") or []) + list(tests.get("private") or [])
            if max_cases is not None:
                cases = cases[: int(max_cases)]
            if not cases:
                return _result(False, 0.0, {"stdout": "", "stderr": "", "timed_out": False, "returncode": None, "elapsed_s": 0.0}, n_cases=0, n_passed=0, error="no test cases")
            if tests.get("type") == "functional":
                return _run_functional_cases(code, tests.get("func_name") or entry_point, cases, timeout_s, mem_mb, tmp)
            return _run_stdin_cases(code, cases, timeout_s, mem_mb, tmp)
        return _run_script(code, str(tests or ""), entry_point, timeout_s, mem_mb, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------------------------ extraction


def _strip_markers(s: str) -> str:
    """Remove DS-1000 / HTML code markers and solution delimiters."""
    s = re.sub(r"</?code>", "", s)
    s = re.sub(r"^\s*(BEGIN|END) SOLUTION\s*$", "", s, flags=re.M)
    return s


def _compiles(code: str) -> bool:
    """True when ``code`` parses as Python (SyntaxWarnings from model code are silenced)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ast.parse(code)
        return True
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False


def _trim_to_compilable(code: str, max_iter: int = 60) -> str:
    """Drop trailing (then leading) prose lines until the text parses; best effort."""
    if _compiles(code):
        return code
    lines = code.rstrip().split("\n")
    for _ in range(min(max_iter, len(lines) - 1)):
        lines.pop()
        cand = "\n".join(lines)
        if _compiles(cand):
            return cand
    lines = code.rstrip().split("\n")
    for _ in range(min(max_iter, len(lines) - 1)):
        lines.pop(0)
        cand = "\n".join(lines)
        if _compiles(cand):
            return cand
    return code


def _strip_leading_prose(text: str) -> str:
    """Cut everything before the first line that looks like Python code."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if _CODE_LINE_RE.match(line):
            return "\n".join(lines[i:])
    return text


def _has_definition(block: str, entry_point: Optional[str] = None) -> bool:
    """True when the block defines ``entry_point`` (if given) or any def / class."""
    if entry_point and re.search(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", block, re.M):
        return True
    if entry_point and entry_point == "Solution" and re.search(r"^\s*class\s+Solution\b", block, re.M):
        return True
    return bool(re.search(r"^\s*(def|class)\s+\w+", block, re.M))


def extract_code(response_text: Optional[str], language: str = "python", entry_point: Optional[str] = None) -> str:
    """Pull the solution code out of a model response (see module docstring for the rules)."""
    text = (response_text or "").replace("\r\n", "\n")
    if not text.strip():
        return ""
    blocks = [(tag.lower(), body) for tag, body in _FENCE_RE.findall(text)]
    if blocks:
        tags = _LANG_TAGS.get(language.lower(), {language.lower(), ""})
        cands = [b for t, b in blocks if t in tags] or [b for _, b in blocks]
        chosen = None
        if entry_point:
            for b in reversed(cands):
                if _has_definition(b, entry_point):
                    chosen = b
                    break
        if chosen is None:
            for b in reversed(cands):
                if _has_definition(b):
                    chosen = b
                    break
        if chosen is None:
            chosen = cands[-1]
        code = _strip_markers(chosen).strip("\n")
        return code if _compiles(code) else _trim_to_compilable(code)
    code = _strip_leading_prose(_strip_markers(text)).strip("\n")
    return code if _compiles(code) else _trim_to_compilable(code)


_OUTPUT_PHRASE_RE = re.compile(r"(?:output|result|returns?|evaluates? to|would be|will be)\s*(?:is|:)?\s*:?\s*`{1,3}([^`\n]+?)`{1,3}", re.I)
_OUTPUT_LINE_RE = re.compile(r"^\s*(?:the\s+)?(?:final\s+)?output\s*(?:is|:)\s*(.+?)\s*$", re.I | re.M)
_OUTPUT_COMMENT_RE = re.compile(r"#\s*(?:expected\s+)?(?:output|result|returns?)\s*(?:is|will be|would be|:)?\s*:?\s*(.+?)\s*$", re.I | re.M)
_BACKTICK_RE = re.compile(r"`([^`\n]+)`")


def _clean_literal(s: str) -> str:
    """Trim trailing sentence punctuation / markdown around an extracted literal."""
    s = s.strip().strip("`").strip()
    s = re.sub(r"\s*(\.|,|;)\s*$", "", s) if not re.search(r"[\]\)\}'\"]\s*(\.|,|;)?\s*$", s) else re.sub(r"\s*(\.|,|;)\s*$", "", s)
    return s.strip()


def extract_prediction(response_text: Optional[str]) -> Optional[str]:
    """Extract the predicted output literal from an output-prediction (CRUXEval) response.

    Order: ``assert f(...) == X`` in a fenced block; the last non-code fenced block when it comes
    after any output phrase; ``output is `X``` phrases; ``Output: X`` lines; ``# Output: X`` code
    comments; the last back-ticked span; a bare short answer (the whole response when it is a
    single line without code); ``None`` when nothing applies.
    """
    text = (response_text or "").replace("\r\n", "\n")
    if not text.strip():
        return None
    blocks = [body.strip() for _, body in _FENCE_RE.findall(text)]
    literal_blocks = [b for b in blocks if b and not re.search(r"^\s*(def |class |import |from |print\(|assert |#)", b, re.M)]
    for b in reversed(blocks):
        m = re.findall(r"assert\s+f\(.*?\)\s*==\s*(.+?)\s*$", b, re.M)
        if m:
            return _clean_literal(m[-1])
    phrases = _OUTPUT_PHRASE_RE.findall(text)
    if literal_blocks and (not phrases or text.rfind("```") > text.rfind(phrases[-1])):
        return literal_blocks[-1]
    if phrases:
        return _clean_literal(phrases[-1])
    lines = _OUTPUT_LINE_RE.findall(text)
    if lines:
        return _clean_literal(lines[-1])
    comments = _OUTPUT_COMMENT_RE.findall(text)
    if comments:
        return _clean_literal(comments[-1])
    ticks = _BACKTICK_RE.findall(text)
    if ticks:
        return _clean_literal(ticks[-1])
    if literal_blocks:
        return literal_blocks[-1]
    stripped = text.strip()
    if "\n" not in stripped and len(stripped) <= 200 and not re.match(r"^(def |class |import |from )", stripped):
        return stripped
    return None


def _import_lines(prompt: Optional[str]) -> List[str]:
    """Top-level import statements of a prompt (used to complete extracted solutions)."""
    out: List[str] = []
    for line in (prompt or "").split("\n"):
        if re.match(r"^(import\s+\S|from\s+\S+\s+import\s)", line):
            out.append(line.rstrip())
    return out


def _with_prompt_imports(code: str, prompt: Optional[str]) -> str:
    """Prepend the prompt's import lines that the code does not already contain."""
    missing = [l for l in _import_lines(prompt) if l not in code]
    if not missing:
        return code
    return _hoist_future("\n".join(missing) + "\n" + code)


def failure_kind(result: Dict[str, Any], code: Optional[str] = None) -> str:
    """Coarse cause label for a verification result (see module docstring)."""
    if result.get("passed") is None:
        return "not_runnable"
    if result.get("passed"):
        return "passed"
    err = result.get("error") or ""
    if err.startswith("empty response"):
        return "empty_response"
    if result.get("timed_out") or err.startswith("timeout"):
        return "timeout"
    if code is not None and not code.strip():
        return "extraction_failure"
    if err.startswith(("SyntaxError", "IndentationError", "TabError")) or err.startswith("no prediction"):
        return "extraction_failure"
    if err.startswith(("ModuleNotFoundError", "ImportError")):
        return "missing_lib"
    if err.startswith(("NameError", "AttributeError")) or "no Solution." in err or "not defined" in err:
        return "harness_mismatch"
    if err.startswith(("AssertionError", "FAILED", "wrong answer", "prediction mismatch")) or err == "exit code 1":
        return "assertion"
    return "runtime_error"


def verify_response(task: Dict[str, Any], response_text: Optional[str], *, timeout_s: Optional[float] = None, mem_mb: int = 1024, max_cases: Optional[int] = None) -> Dict[str, Any]:
    """Verify a model response against the reconstructed task (source-specific assembly).

    Returns the ``run_python_tests`` dict plus ``code`` (what was executed / predicted), ``kind``
    (``script`` / ``stdin`` / ``functional`` / ``cruxeval`` / ``ds1000``) and ``failure_kind``.
    Non-runnable tasks return ``passed=None, frac=None, failure_kind='not_runnable'``.
    """
    src = task.get("source")
    if not task.get("runnable"):
        return {"passed": None, "frac": None, "stdout": "", "stderr": "", "error": f"not runnable: {task.get('notes', '')}", "elapsed_s": 0.0,
                "n_cases": 0, "n_passed": 0, "timed_out": False, "returncode": None, "code": None, "kind": None, "failure_kind": "not_runnable"}
    t_s = float(timeout_s) if timeout_s is not None else DEFAULT_TIMEOUTS.get(src, 10)
    text = response_text or ""
    if not text.strip():
        res = _result(False, 0.0, {"stdout": "", "stderr": "", "timed_out": False, "returncode": None, "elapsed_s": 0.0}, error="empty response")
        res.update(code="", kind=None, failure_kind="empty_response")
        return res
    if src == "cruxeval":
        pred = extract_prediction(text)
        if pred is None:
            res = _result(False, 0.0, {"stdout": "", "stderr": "", "timed_out": False, "returncode": None, "elapsed_s": 0.0}, error="no prediction found in response")
            res.update(code=None, kind="cruxeval", failure_kind="extraction_failure")
            return res
        script = _CRUXEVAL_HARNESS.format(pred=pred, inp=task["meta"]["input"])
        res = run_python_tests(task["raw_prompt"], script, timeout_s=t_s, mem_mb=mem_mb)
        res.update(code=pred, kind="cruxeval")
        res["failure_kind"] = failure_kind(res, pred)
        return res
    if src == "lcb":
        tests = task["tests"]
        func = tests.get("func_name")
        code = extract_code(text, "python", entry_point=func or None)
        if tests.get("type") == "functional" and func and not re.search(rf"def\s+{re.escape(func)}\s*\(", code):
            alt = extract_code(text, "python", entry_point="Solution")
            if "class Solution" in alt:
                code = alt
        res = run_python_tests(code, tests, entry_point=func, timeout_s=t_s, mem_mb=mem_mb, max_cases=max_cases if max_cases is not None else DEFAULT_LCB_MAX_CASES)
        res.update(code=code, kind=tests.get("type"))
        res["failure_kind"] = failure_kind(res, code)
        return res
    if src == "ds1000":
        solution = extract_code(text, "python")
        ctx = task["tests"]
        script = ctx.rstrip() + "\n\n" + f"test_execution({solution!r})\n" + (f"test_string({solution!r})\n" if "def test_string" in ctx else "") + "print('OK')\n"
        res = run_python_tests("", script, timeout_s=t_s, mem_mb=mem_mb)
        res.update(code=solution, kind="ds1000")
        res["failure_kind"] = failure_kind(res, solution)
        return res
    entry = task.get("entry_point")
    code = extract_code(text, "python", entry_point=entry)
    if src in ("bcb", "humaneval", "he", "hex"):
        code = _with_prompt_imports(code, task.get("raw_prompt"))
    res = run_python_tests(code, task["tests"], entry_point=entry, timeout_s=t_s, mem_mb=mem_mb)
    res.update(code=code, kind="script")
    res["failure_kind"] = failure_kind(res, code)
    return res


__all__ = ["run_python_tests", "extract_code", "extract_prediction", "verify_response", "failure_kind", "classify_error", "DEFAULT_TIMEOUTS", "LCB_HEADER", "PYTHON"]

