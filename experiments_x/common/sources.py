"""Reconstruct prompts and tests for CodeRouterBench task ids from the local Hugging Face cache.

Everything is read directly from ``~/.cache/huggingface/hub/datasets--*`` (parquet via pandas /
pyarrow, jsonl / json via the standard library); no network access and no ``datasets`` builder.
Nothing is loaded at import time; each source is loaded lazily on first use and memoised.

Public API
----------
* ``parse_task_id(task_id)``  -> dict(dim, dim_no, dimension, source, variant, item, language)
* ``load_task(task_id)``      -> dict(task_id, source, dimension, language, prompt, tests,
                                 entry_point, runnable, notes, ...)  (see ``TASK_KEYS``)
* ``iter_tasks(dimensions=None, sources=None, split=None)`` -> iterator over ``load_task`` dicts
  joined with ``$ACR/coderouterbench/id_tasks.jsonl`` (adds ``split`` and ``source_split``)
* ``lib_available(name)``     -> bool (importable in this venv, via ``importlib.util.find_spec``)
* ``FIELD_MAP``               -> the per-source field mapping below, as data (used in reports)

Task-id grammar (``dim{N}_{source}_{rest}``; the source is the second underscore token)
-------------------------------------------------------------------------------------
============  =====================================  ==========================================
source        rest                                   dataset key
============  =====================================  ==========================================
humaneval     ``HumanEval_51``                       evalplus/humanevalplus ``HumanEval/51``
he            ``completion_HumanEval_0`` (dim4),      same HumanEval item; dim9 = write tests
              ``testgen_HumanEval_150`` (dim9)
mbpp          ``760`` (dim1), ``completion_11``       evalplus/mbppplus ``task_id`` (int)
              (dim4), ``testgen_2`` (dim9)
bcb           ``BigCodeBench/767``, ``v014_…``,       bigcode/bigcodebench v0.1.4 ``task_id``
              ``algo2_…`` (dim2), ``completion_…``    (the ``v014_``/``algo2_``/``completion_``/
              (dim4), ``ds_…`` (dim6)                 ``ds_`` prefixes are cell labels only)
cruxeval      ``7``                                  cruxeval-org/cruxeval ``sample_7``
hex           ``python_Python/2``, ``java_Java/118``  THUDM/humaneval-x ``task_id`` (all 5 langs)
lcb           ``2953``, ``abc366_f`` (dim2);          livecodebench/code_generation_lite
              ``testgen_3353`` (dim9)                 ``question_id`` over test.jsonl..test6.jsonl
ds1000        ``0``                                  xlangai/DS-1000 ``metadata.problem_id``
multiple      ``rs_HumanEval_0_has_close_elements``  nuprl/MultiPL-E humaneval-{rs,ts} ``name``
debugbench    ``3851``; ``sup_1114``                  Rtian/DebugBench eval.json row ``n``;
                                                     ``sup_n`` -> row ``(n-1000)//2``
bugs2fix      ``1044``; ``sup_7093``                  code_x_glue_cc_code_refinement
                                                     medium/test row ``n``; ``sup_n`` -> ``n-7000``
codexglue     ``3471``; ``sup_15001``                 code_x_glue_ct_code_to_text python/test
                                                     row ``n``; ``sup_n`` -> ``n-15000``
canitedit     ``78``                                 nuprl/CanItEdit ``id`` (3 of 20 ids absent)
swebench      ``django__django-16041``               princeton-nlp/SWE-bench_Verified, then
                                                     SWE-bench_Lite (test, dev) ``instance_id``
============  =====================================  ==========================================

Exact fields used per source (also available as ``FIELD_MAP``)
---------------------------------------------------------------
humaneval / he : evalplus/humanevalplus ``prompt`` (signature + docstring; verbatim in ``raw_prompt``),
                 ``entry_point``, ``canonical_solution`` -> ``reference``.  ``tests`` = the ORIGINAL
                 HumanEval ``check(candidate)`` suite taken from THUDM/humaneval-x ``Python/N`` (its
                 trailing ``check(...)`` call removed; the sandbox appends ``check(<entry_point>)``);
                 the stricter EvalPlus ``test`` is kept in ``meta['tests_plus']``.  Reason: on the
                 smoke sample the released CodeRouterBench scores agree with the base tests
                 (0.959 / 0.981 for humaneval / he) but not with EvalPlus (0.919 / 0.931).  dim9
                 testgen: the prompt asks for tests; ``runnable=False``.
mbpp           : ``prompt`` (task description), ``code`` -> ``reference`` and the entry point (first
                 ``def`` in ``code``), ``test`` (EvalPlus harness: top-level loop over ``inputs`` /
                 ``results``, calls the entry point directly), ``test_list`` and ``test_imports``
                 -> ``meta``.  dim1 prompt = description only (the recorded responses use the
                 reference function name in only 10-35 % of cases, so the original prompt evidently
                 did not show the asserts); dim4 completion prompt = description + the reference
                 ``def`` header; dim9 testgen prompt = description + reference code, ``runnable=False``.
bcb            : ``complete_prompt`` (imports + signature + docstring), ``test`` (``unittest``
                 ``TestCases``), ``entry_point`` (``task_func``), ``libs`` (parsed with
                 ``ast.literal_eval``), ``canonical_solution`` -> ``reference``.  ``runnable`` iff
                 every lib is importable in the venv.  Version v0.1.4 for every id (the tests of the
                 benchmark ids are identical in v0.1.2 - v0.1.4; one id differs in v0.1.0/v0.1.1).
cruxeval       : ``code``, ``input``, ``output``.  Prompt = predict ``f(<input>)``; ``tests`` =
                 ``assert f(<input>) == <output>``; ``entry_point='f'``; ``runnable=True``.
hex            : ``prompt`` (declaration + docstring), ``test`` (Python: defines and calls
                 ``check``), ``declaration`` (entry point parsed from it), ``canonical_solution``
                 -> ``reference``.  Only ``python`` is runnable.
lcb            : ``question_content``, ``starter_code``, ``platform``, ``public_test_cases`` (JSON
                 list of {input, output, testtype}), ``private_test_cases`` (base64 -> zlib ->
                 pickle -> JSON list, official LCB decoding; ``tests['private_decoded']`` says
                 whether it worked), ``metadata.func_name`` (leetcode).  ``tests`` is a dict
                 ``{type: 'stdin'|'functional', func_name, public, private, ...}``.  dim9 testgen:
                 same fields, prompt asks for test cases, ``runnable=False``.
ds1000         : ``prompt`` (verbatim), ``reference_code`` -> ``reference``, ``code_context`` ->
                 ``tests`` (defines ``test_execution(solution)`` and sometimes
                 ``test_string(solution)``), ``metadata.library`` / ``problem_id``.  ``runnable``
                 iff every module imported anywhere in ``code_context`` is available.
multiple       : ``prompt``, ``tests``, ``language`` (rs / ts).  ``runnable=False`` (no toolchain).
debugbench     : ``question``, ``examples``, ``constraints``, ``buggy_code``, ``language``, ``slug``,
                 ``solution`` -> ``reference``.  ``runnable=False`` (LeetCode hidden tests).
bugs2fix       : ``buggy`` -> prompt, ``fixed`` -> ``reference``.  ``runnable=False``.
codexglue      : ``code`` -> prompt (summarise), ``docstring`` -> ``reference``.  ``runnable=False``.
canitedit      : ``before``, ``instruction_descriptive`` (fallback ``instruction_lazy``), ``tests``,
                 ``after`` -> ``reference``.  ``runnable=False`` (per the shared spec; the tests are
                 kept in ``tests`` for later use).
swebench       : ``problem_statement`` (+ ``repo``, ``base_commit``), ``patch`` -> ``reference``.
                 ``runnable=False``.

Mapping confidence (recorded in each task's ``notes``): every source except debugbench /
codexglue / bugs2fix ``sup_`` ids resolves by an explicit key.  Index-based mappings were checked
against the recorded model responses in ``response_matrix.json`` by token overlap: bugs2fix row n
matches the global best row in 37/40 sampled ids (sup: 36/40); debugbench row n reproduces the
response's language for 914/929 ids and the LeetCode slug for ~73 % (rows with the same slug but a
different bug are indistinguishable from the fixed code), sup ``(n-1000)//2`` reproduces the slug for
104/111; codexglue row n is the global best for 20/40 sampled ids (summaries vs code overlap is a
weak signal; spot checks such as 3471 -> ``register_json`` and sup_15001 -> ``dailymotion_download``
match exactly).
"""
from __future__ import annotations

import ast
import base64
import functools
import glob
import importlib.util
import json
import os
import pickle
import re
import zlib
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import pandas as pd

# ------------------------------------------------------------------------------------------ paths

HF_HUB = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME_HUB") or os.path.expanduser("~/.cache/huggingface/hub")
ACR = os.environ.get("ACROUTER_DATA", "data/agent-as-a-router/data")
ID_TASKS_PATH = os.path.join(ACR, "coderouterbench", "id_tasks.jsonl")

DIMENSION_NAMES: Dict[str, str] = {
    "dim1": "code_generation",
    "dim2": "algorithm",
    "dim3": "bug_fixing",
    "dim4": "code_completion",
    "dim5": "code_refactoring",
    "dim6": "data_science",
    "dim7": "multi_language",
    "dim8": "code_understanding",
    "dim9": "test_generation",
}

SOURCES: Tuple[str, ...] = (
    "humaneval", "he", "mbpp", "bcb", "cruxeval", "hex", "lcb", "ds1000",
    "multiple", "debugbench", "bugs2fix", "codexglue", "canitedit", "swebench",
)
RUNNABLE_SOURCES: Tuple[str, ...] = ("humaneval", "he", "mbpp", "bcb", "cruxeval", "hex", "lcb", "ds1000")

TASK_KEYS: Tuple[str, ...] = (
    "task_id", "found", "source", "dim", "dimension", "variant", "language", "prompt", "raw_prompt",
    "tests", "entry_point", "reference", "libs", "runnable", "notes", "meta",
)

FIELD_MAP: Dict[str, Dict[str, Any]] = {
    "humaneval": {"repo": "evalplus/humanevalplus (+ THUDM/humaneval-x Python/N for the base tests)", "file": "data/test-*.parquet", "key": "task_id (HumanEval/N)", "prompt": "prompt", "tests": "base HumanEval check(candidate) from humaneval-x Python/N test (EvalPlus test kept in meta.tests_plus)", "entry_point": "entry_point", "reference": "canonical_solution", "runnable": "True"},
    "he": {"repo": "evalplus/humanevalplus (+ THUDM/humaneval-x Python/N for the base tests)", "file": "data/test-*.parquet", "key": "task_id (HumanEval/N) from completion_/testgen_ rest", "prompt": "prompt (+ completion / test-writing instruction)", "tests": "base HumanEval check(candidate) (EvalPlus test in meta.tests_plus)", "entry_point": "entry_point", "reference": "canonical_solution", "runnable": "dim4 completion True; dim9 testgen False"},
    "mbpp": {"repo": "evalplus/mbppplus", "file": "data/test-*.parquet", "key": "task_id (int)", "prompt": "prompt (dim4: + def header of code; dim9: + code)", "tests": "test (top-level inputs/results loop) ; test_list, test_imports in meta", "entry_point": "first def in code", "reference": "code", "runnable": "dim1/dim4 True; dim9 testgen False"},
    "bcb": {"repo": "bigcode/bigcodebench", "file": "data/v0.1.4-*.parquet", "key": "task_id (BigCodeBench/N), prefixes v014_/algo2_/completion_/ds_ dropped", "prompt": "complete_prompt", "tests": "test (unittest TestCases)", "entry_point": "entry_point", "reference": "canonical_solution", "libs": "libs (ast.literal_eval)", "runnable": "all libs importable"},
    "cruxeval": {"repo": "cruxeval-org/cruxeval", "file": "test.jsonl", "key": "id (sample_N)", "prompt": "code + input (predict output)", "tests": "assert f(<input>) == <output>", "entry_point": "f", "reference": "output", "runnable": "True"},
    "hex": {"repo": "THUDM/humaneval-x", "file": "data/{lang}/data/humaneval.jsonl", "key": "task_id (Python/N, CPP/N, Java/N, JavaScript/N, Go/N)", "prompt": "prompt", "tests": "test (python: defines and calls check)", "entry_point": "parsed from declaration", "reference": "canonical_solution", "runnable": "python only"},
    "lcb": {"repo": "livecodebench/code_generation_lite", "file": "test.jsonl .. test6.jsonl", "key": "question_id", "prompt": "question_content + starter_code", "tests": "public_test_cases (JSON) + private_test_cases (base64/zlib/pickle/JSON) as dict{type,func_name,public,private}", "entry_point": "metadata.func_name (leetcode) else None", "reference": None, "runnable": "dim2 True; dim9 testgen False"},
    "ds1000": {"repo": "xlangai/DS-1000", "file": "test.jsonl", "key": "metadata.problem_id", "prompt": "prompt", "tests": "code_context (test_execution / test_string)", "entry_point": None, "reference": "reference_code", "runnable": "all modules imported in code_context importable"},
    "multiple": {"repo": "nuprl/MultiPL-E", "file": "humaneval-{rs,ts}/test-*.parquet", "key": "name", "prompt": "prompt", "tests": "tests", "entry_point": None, "reference": None, "runnable": "False (no rust/ts toolchain)"},
    "debugbench": {"repo": "Rtian/DebugBench", "file": "eval.json", "key": "row n; sup_n -> (n-1000)//2 (medium confidence)", "prompt": "question + examples + constraints + buggy_code", "tests": None, "entry_point": None, "reference": "solution", "runnable": "False"},
    "bugs2fix": {"repo": "code_x_glue_cc_code_refinement", "file": "medium/test-*.parquet", "key": "row n; sup_n -> n-7000", "prompt": "buggy", "tests": None, "entry_point": None, "reference": "fixed", "runnable": "False"},
    "codexglue": {"repo": "code_x_glue_ct_code_to_text", "file": "python/test-*.parquet", "key": "row n; sup_n -> n-15000 (medium confidence)", "prompt": "code (summarise)", "tests": None, "entry_point": None, "reference": "docstring", "runnable": "False"},
    "canitedit": {"repo": "nuprl/CanItEdit", "file": "data/test-*.parquet", "key": "id (17/20 present)", "prompt": "instruction_descriptive + before", "tests": "tests (kept, not run)", "entry_point": None, "reference": "after", "runnable": "False"},
    "swebench": {"repo": "princeton-nlp/SWE-bench_Verified (+ SWE-bench_Lite)", "file": "data/*.parquet", "key": "instance_id", "prompt": "problem_statement (+ repo, base_commit)", "tests": "FAIL_TO_PASS / PASS_TO_PASS in meta", "entry_point": None, "reference": "patch", "runnable": "False"},
}

DS1000_LIBRARY_MODULES: Dict[str, str] = {
    "Pandas": "pandas", "Numpy": "numpy", "Matplotlib": "matplotlib", "Sklearn": "sklearn",
    "Scipy": "scipy", "Pytorch": "torch", "Tensorflow": "tensorflow",
}
_MODULE_ALIASES: Dict[str, str] = {"mpl_toolkits": "mpl_toolkits", "Crypto": "Crypto"}
_STDLIB_LIKE_ALWAYS = {"__future__"}


class SourceUnavailable(RuntimeError):
    """Raised when a dataset snapshot is missing from the local Hugging Face cache."""


# --------------------------------------------------------------------------------------- utilities


def _snapshot_dir(repo: str) -> str:
    """Return the snapshot directory of ``repo`` (``owner/name`` or bare name) in the HF cache."""
    folder = os.path.join(HF_HUB, "datasets--" + repo.replace("/", "--"))
    ref = os.path.join(folder, "refs", "main")
    if os.path.exists(ref):
        with open(ref, "r", encoding="utf-8") as fh:
            revision = fh.read().strip()
        cand = os.path.join(folder, "snapshots", revision)
        if os.path.isdir(cand):
            return cand
    snaps = sorted(glob.glob(os.path.join(folder, "snapshots", "*")), key=os.path.getmtime)
    if not snaps:
        raise SourceUnavailable(f"no snapshot for {repo} under {HF_HUB}")
    return snaps[-1]


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a jsonl file, skipping blank lines."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _first_parquet(snapshot: str, pattern: str) -> str:
    """Return the single parquet file matching ``pattern`` inside ``snapshot``."""
    files = sorted(glob.glob(os.path.join(snapshot, pattern)))
    if not files:
        raise SourceUnavailable(f"no file {pattern} in {snapshot}")
    return files[0]


def _to_list(v: Any) -> List[Any]:
    """Normalise numpy arrays / tuples / None to a plain list."""
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    try:
        return list(v.tolist())
    except AttributeError:
        return list(v)


def _str(v: Any) -> Optional[str]:
    """Return ``v`` as str (``None`` stays ``None``)."""
    return None if v is None else str(v)


@functools.lru_cache(maxsize=None)
def lib_available(name: str) -> bool:
    """True when top-level module ``name`` can be imported in this interpreter (no import is run)."""
    root = name.split(".")[0]
    if root in _STDLIB_LIKE_ALWAYS:
        return True
    try:
        return importlib.util.find_spec(_MODULE_ALIASES.get(root, root)) is not None
    except (ImportError, ValueError, AttributeError):
        return False


_IMPORT_RE = re.compile(r"^\s*(?:from\s+([A-Za-z_][\w\.]*)\s+import|import\s+([A-Za-z_][\w\.]*(?:\s*,\s*[A-Za-z_][\w\.]*)*))", re.M)


def imported_modules(code: str) -> List[str]:
    """Top-level module names imported anywhere in ``code`` (also inside string literals)."""
    mods: List[str] = []
    for m in _IMPORT_RE.finditer(code or ""):
        if m.group(1):
            mods.append(m.group(1).split(".")[0])
        else:
            for part in m.group(2).split(","):
                mods.append(part.strip().split(".")[0])
    seen: List[str] = []
    for x in mods:
        if x and x not in seen:
            seen.append(x)
    return seen


def _entry_from_def(code: str) -> Optional[str]:
    """Name of the first top-level ``def`` in ``code``."""
    m = re.search(r"^def\s+([A-Za-z_]\w*)\s*\(", code or "", re.M)
    return m.group(1) if m else None


# ----------------------------------------------------------------------------------- dataset loaders


@functools.lru_cache(maxsize=None)
def _humanevalplus() -> Dict[str, Dict[str, Any]]:
    """evalplus/humanevalplus rows keyed by ``HumanEval/N``."""
    df = pd.read_parquet(_first_parquet(_snapshot_dir("evalplus/humanevalplus"), "data/*.parquet"))
    return {str(r["task_id"]): {k: r[k] for k in df.columns} for r in df.to_dict("records")}


@functools.lru_cache(maxsize=None)
def _mbppplus() -> Dict[int, Dict[str, Any]]:
    """evalplus/mbppplus rows keyed by integer task id (``test_list``/``test_imports`` as lists)."""
    df = pd.read_parquet(_first_parquet(_snapshot_dir("evalplus/mbppplus"), "data/*.parquet"))
    out: Dict[int, Dict[str, Any]] = {}
    for r in df.to_dict("records"):
        row = {k: r[k] for k in df.columns}
        row["test_list"] = [str(x) for x in _to_list(r.get("test_list"))]
        row["test_imports"] = [str(x) for x in _to_list(r.get("test_imports"))]
        out[int(r["task_id"])] = row
    return out


@functools.lru_cache(maxsize=None)
def _bigcodebench(version: str = "v0.1.4") -> Dict[str, Dict[str, Any]]:
    """bigcode/bigcodebench rows (given release) keyed by ``BigCodeBench/N``; ``libs`` parsed."""
    df = pd.read_parquet(_first_parquet(_snapshot_dir("bigcode/bigcodebench"), f"data/{version}-*.parquet"))
    out: Dict[str, Dict[str, Any]] = {}
    for r in df.to_dict("records"):
        row = {k: r[k] for k in df.columns}
        libs = r.get("libs")
        if isinstance(libs, str):
            try:
                libs = ast.literal_eval(libs)
            except (ValueError, SyntaxError):
                libs = [x.strip(" '\"[]") for x in libs.split(",") if x.strip(" '\"[]")]
        row["libs"] = [str(x) for x in _to_list(libs)]
        out[str(r["task_id"])] = row
    return out


@functools.lru_cache(maxsize=None)
def _cruxeval() -> Dict[str, Dict[str, Any]]:
    """cruxeval-org/cruxeval rows keyed by ``sample_N``."""
    return {r["id"]: r for r in _read_jsonl(os.path.join(_snapshot_dir("cruxeval-org/cruxeval"), "test.jsonl"))}


@functools.lru_cache(maxsize=None)
def _humaneval_x() -> Dict[str, Dict[str, Any]]:
    """THUDM/humaneval-x rows for all five languages keyed by ``task_id`` (e.g. ``Python/2``)."""
    snap = _snapshot_dir("THUDM/humaneval-x")
    out: Dict[str, Dict[str, Any]] = {}
    for path in sorted(glob.glob(os.path.join(snap, "data", "*", "data", "humaneval.jsonl"))):
        lang_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))
        for r in _read_jsonl(path):
            r = dict(r)
            r["_lang_dir"] = lang_dir
            out[r["task_id"]] = r
    return out


@functools.lru_cache(maxsize=None)
def _lcb() -> Dict[str, Dict[str, Any]]:
    """livecodebench/code_generation_lite rows over test.jsonl..test6.jsonl keyed by ``question_id``."""
    snap = _snapshot_dir("livecodebench/code_generation_lite")
    out: Dict[str, Dict[str, Any]] = {}
    for path in sorted(glob.glob(os.path.join(snap, "test*.jsonl"))):
        for r in _read_jsonl(path):
            r = dict(r)
            r["_file"] = os.path.basename(path)
            out.setdefault(r["question_id"], r)
    return out


@functools.lru_cache(maxsize=None)
def _ds1000() -> Dict[int, Dict[str, Any]]:
    """xlangai/DS-1000 rows keyed by ``metadata.problem_id``."""
    return {int(r["metadata"]["problem_id"]): r for r in _read_jsonl(os.path.join(_snapshot_dir("xlangai/DS-1000"), "test.jsonl"))}


@functools.lru_cache(maxsize=None)
def _multiple() -> Dict[Tuple[str, str], Dict[str, Any]]:
    """nuprl/MultiPL-E humaneval-{lang} rows keyed by ``(lang, name)``."""
    snap = _snapshot_dir("nuprl/MultiPL-E")
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for path in sorted(glob.glob(os.path.join(snap, "humaneval-*", "test-*.parquet"))):
        lang = os.path.basename(os.path.dirname(path)).split("-", 1)[1]
        df = pd.read_parquet(path)
        for r in df.to_dict("records"):
            row = {k: r[k] for k in df.columns}
            row["stop_tokens"] = [str(x) for x in _to_list(r.get("stop_tokens"))]
            out[(lang, str(r["name"]))] = row
    return out


@functools.lru_cache(maxsize=None)
def _debugbench() -> List[Dict[str, Any]]:
    """Rtian/DebugBench eval.json as a list (row order is the id space)."""
    with open(os.path.join(_snapshot_dir("Rtian/DebugBench"), "eval.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


@functools.lru_cache(maxsize=None)
def _bugs2fix() -> pd.DataFrame:
    """code_x_glue_cc_code_refinement medium/test split (columns id, buggy, fixed)."""
    return pd.read_parquet(_first_parquet(_snapshot_dir("code_x_glue_cc_code_refinement"), "medium/test-*.parquet"))


@functools.lru_cache(maxsize=None)
def _codexglue() -> pd.DataFrame:
    """code_x_glue_ct_code_to_text python/test split (id, repo, path, func_name, code, docstring, url)."""
    return pd.read_parquet(
        _first_parquet(_snapshot_dir("code_x_glue_ct_code_to_text"), "python/test-*.parquet"),
        columns=["id", "repo", "path", "func_name", "code", "docstring", "url"],
    )


@functools.lru_cache(maxsize=None)
def _canitedit() -> Dict[int, Dict[str, Any]]:
    """nuprl/CanItEdit rows keyed by integer ``id``."""
    df = pd.read_parquet(_first_parquet(_snapshot_dir("nuprl/CanItEdit"), "data/test-*.parquet"))
    return {int(r["id"]): {k: r[k] for k in df.columns} for r in df.to_dict("records")}


@functools.lru_cache(maxsize=None)
def _swebench() -> Dict[str, Dict[str, Any]]:
    """SWE-bench rows keyed by ``instance_id`` (Verified first, then Lite test/dev)."""
    out: Dict[str, Dict[str, Any]] = {}
    for repo in ("princeton-nlp/SWE-bench_Verified", "princeton-nlp/SWE-bench_Lite"):
        try:
            snap = _snapshot_dir(repo)
        except SourceUnavailable:
            continue
        for path in sorted(glob.glob(os.path.join(snap, "data", "*.parquet"))):
            df = pd.read_parquet(path)
            for r in df.to_dict("records"):
                row = {k: r[k] for k in df.columns}
                row["_repo"] = repo
                out.setdefault(str(r["instance_id"]), row)
    return out


@functools.lru_cache(maxsize=None)
def _id_tasks() -> List[Dict[str, Any]]:
    """Rows of ``$ACR/coderouterbench/id_tasks.jsonl`` (task_id, dimension, split, source_split)."""
    if not os.path.exists(ID_TASKS_PATH):
        raise SourceUnavailable(f"missing {ID_TASKS_PATH} (set ACROUTER_DATA)")
    return _read_jsonl(ID_TASKS_PATH)


@functools.lru_cache(maxsize=None)
def _id_task_index() -> Dict[str, Dict[str, Any]]:
    """``task_id -> row`` view of ``_id_tasks``."""
    return {r["task_id"]: r for r in _id_tasks()}


# ------------------------------------------------------------------------------------ id parsing


def parse_task_id(task_id: str) -> Dict[str, Any]:
    """Split a CodeRouterBench id into dim, source, variant, dataset key (``item``) and language."""
    parts = task_id.split("_")
    if len(parts) < 3:
        raise ValueError(f"unrecognised task id {task_id!r}")
    dim, source = parts[0], parts[1]
    rest = task_id[len(dim) + 1 + len(source) + 1:]
    info: Dict[str, Any] = {"task_id": task_id, "dim": dim, "dim_no": int(dim[3:]) if dim[3:].isdigit() else None,
                            "dimension": DIMENSION_NAMES.get(dim), "source": source, "variant": None, "item": rest, "language": "python"}
    if source == "humaneval":
        info["item"] = rest.replace("HumanEval_", "HumanEval/")
    elif source == "he":
        variant, item = rest.split("_", 1)
        info.update(variant=variant, item=item.replace("HumanEval_", "HumanEval/"))
    elif source == "mbpp":
        if "_" in rest:
            variant, item = rest.split("_", 1)
            info.update(variant=variant, item=int(item))
        else:
            info["item"] = int(rest)
    elif source == "bcb":
        k = rest.find("BigCodeBench/")
        info.update(variant=(rest[:k].rstrip("_") or None) if k > 0 else None, item=rest[k:] if k >= 0 else rest)
    elif source == "lcb":
        if rest.startswith("testgen_"):
            info.update(variant="testgen", item=rest[len("testgen_"):])
    elif source == "cruxeval":
        info["item"] = f"sample_{rest}"
    elif source == "hex":
        lang, item = rest.split("_", 1)
        info.update(language=lang, item=item)
    elif source == "multiple":
        lang, name = rest.split("_", 1)
        info.update(language=lang, item=name)
    elif source in ("debugbench", "bugs2fix", "codexglue"):
        if rest.startswith("sup_"):
            info.update(variant="sup", item=int(rest[4:]))
        else:
            info["item"] = int(rest)
        if source == "bugs2fix":
            info["language"] = "java"
    elif source in ("canitedit", "ds1000"):
        info["item"] = int(rest)
    elif source == "swebench":
        info["item"] = rest
    return info


# --------------------------------------------------------------------------------------- builders


def _base(info: Dict[str, Any]) -> Dict[str, Any]:
    """Skeleton task dict with every key of ``TASK_KEYS``."""
    return {
        "task_id": info["task_id"], "found": True, "source": info["source"], "dim": info["dim"],
        "dimension": info["dimension"], "variant": info.get("variant"), "language": info.get("language", "python"),
        "prompt": None, "raw_prompt": None, "tests": None, "entry_point": None, "reference": None,
        "libs": [], "runnable": False, "notes": "", "meta": {},
    }


def _missing(info: Dict[str, Any], why: str) -> Dict[str, Any]:
    """Task dict for an id that could not be resolved in the local cache."""
    t = _base(info)
    t.update(found=False, notes=f"NOT FOUND: {why}")
    return t


def _fence(code: str, lang: str = "python") -> str:
    """Wrap ``code`` in a markdown fence."""
    return f"```{lang}\n{code.rstrip()}\n```"


def _build_humaneval(info: Dict[str, Any]) -> Dict[str, Any]:
    """humaneval (dim1) and he (dim4 completion / dim9 testgen) from evalplus/humanevalplus."""
    row = _humanevalplus().get(info["item"])
    if row is None:
        return _missing(info, f"{info['item']} not in evalplus/humanevalplus")
    t = _base(info)
    raw = str(row["prompt"])
    variant = info.get("variant")
    # Tests: the original HumanEval `check(candidate)` suite (taken from THUDM/humaneval-x Python/N,
    # which carries the base tests) is the default because the released CodeRouterBench scores agree
    # with it (0.96-0.98 on the smoke sample) and not with the stricter EvalPlus suite (0.92-0.93);
    # the EvalPlus suite stays available in meta['tests_plus'].
    plus_test = str(row["test"])
    base_row = None
    try:
        base_row = _humaneval_x().get("Python/" + info["item"].split("/")[-1])
    except SourceUnavailable:
        base_row = None
    base_test = _base_check_only(str(base_row["test"])) if base_row else None
    tests = base_test or plus_test
    tests_source = "HumanEval base tests (humaneval-x Python)" if base_test else "EvalPlus HumanEval+ tests (base tests unavailable)"
    if variant == "testgen":
        t["prompt"] = ("Write a comprehensive set of unit tests (plain Python `assert` statements or a unittest class) "
                       "for the following function. Do not implement the function itself.\n\n" + _fence(raw))
        t["runnable"] = False
        t["notes"] = f"dim9 test generation: the task is to write tests, so no execution-based verification; {tests_source} kept in tests for reference."
    elif variant == "completion":
        t["prompt"] = "Complete the following Python function. Return the complete function.\n\n" + _fence(raw)
        t["runnable"] = True
        t["notes"] = f"dim4 completion variant of the same HumanEval item; verified with the {tests_source}; EvalPlus suite in meta['tests_plus']."
    else:
        t["prompt"] = "Complete the following Python function.\n\n" + _fence(raw)
        t["runnable"] = True
        t["notes"] = f"{tests_source}; sandbox appends check(<entry_point>); EvalPlus suite in meta['tests_plus']."
    t.update(raw_prompt=raw, tests=tests, entry_point=str(row["entry_point"]), reference=str(row["canonical_solution"]),
             meta={"hf_task_id": info["item"], "tests_plus": plus_test, "tests_source": tests_source})
    return t


def _base_check_only(test: str) -> str:
    """Strip the trailing top-level ``check(<name>)`` call of a HumanEval-X test so the sandbox can append ``check(<entry_point>)``."""
    return re.sub(r"^check\([^\n]*\)\s*$", "", test, flags=re.M).rstrip() + "\n"


def _build_mbpp(info: Dict[str, Any]) -> Dict[str, Any]:
    """mbpp (dim1), mbpp completion (dim4) and mbpp testgen (dim9) from evalplus/mbppplus."""
    row = _mbppplus().get(int(info["item"]))
    if row is None:
        return _missing(info, f"task_id {info['item']} not in evalplus/mbppplus")
    t = _base(info)
    desc = str(row["prompt"]).strip()
    code = str(row["code"])
    entry = _entry_from_def(code)
    header_m = re.search(r"^def\s+[A-Za-z_]\w*\s*\(.*?\)\s*:", code, re.M | re.S)
    header = header_m.group(0) if header_m else (f"def {entry}(...):" if entry else "")
    variant = info.get("variant")
    if variant == "completion":
        t["prompt"] = f"Complete the following Python function.\n\n{desc}\n\n" + _fence(header + "\n    ...")
        t["runnable"] = True
        t["notes"] = "dim4 completion: prompt = MBPP description + the reference def header (recorded responses define this name in >95 % of cases)."
    elif variant == "testgen":
        t["prompt"] = (f"Write a comprehensive set of unit tests (plain Python `assert` statements or a unittest class) "
                       f"for the following function.\n\n{desc}\n\n" + _fence(code))
        t["runnable"] = False
        t["notes"] = "dim9 test generation: no execution-based verification; MBPP+ harness kept in tests for reference."
    else:
        t["prompt"] = desc
        t["runnable"] = True
        t["notes"] = ("dim1 prompt = MBPP+ description only (no asserts / function name: recorded responses use the reference name "
                      "in only 10-35 % of cases, consistent with the 10 % released pass rate). Verified with the MBPP+ harness, "
                      "which calls the reference entry point.")
    t.update(raw_prompt=desc, tests=str(row["test"]), entry_point=entry, reference=code,
             meta={"test_list": row["test_list"], "test_imports": row["test_imports"], "source_file": _str(row.get("source_file"))})
    return t


def _build_bcb(info: Dict[str, Any]) -> Dict[str, Any]:
    """bcb (dim1/dim2/dim4/dim6 cells) from bigcode/bigcodebench v0.1.4."""
    row = _bigcodebench().get(info["item"])
    if row is None:
        return _missing(info, f"{info['item']} not in bigcode/bigcodebench")
    t = _base(info)
    raw = str(row["complete_prompt"])
    libs = list(row["libs"])
    missing = [l for l in libs if not lib_available(l)]
    t.update(
        prompt="Complete the following Python function. Return the complete function including the imports.\n\n" + _fence(raw),
        raw_prompt=raw, tests=str(row["test"]), entry_point=str(row["entry_point"]), reference=str(row["canonical_solution"]),
        libs=libs, runnable=not missing,
        notes=("BigCodeBench v0.1.4 complete_prompt/test/entry_point/libs" + (f"; cell label {info['variant']!r}" if info.get("variant") else "")
               + (f"; NOT runnable: missing libs {missing}" if missing else "; all libs importable")),
        meta={"hf_task_id": info["item"], "instruct_prompt": _str(row.get("instruct_prompt")), "code_prompt": _str(row.get("code_prompt")), "missing_libs": missing},
    )
    return t


def _build_cruxeval(info: Dict[str, Any]) -> Dict[str, Any]:
    """cruxeval (dim4) output prediction from cruxeval-org/cruxeval."""
    row = _cruxeval().get(info["item"])
    if row is None:
        return _missing(info, f"{info['item']} not in cruxeval-org/cruxeval")
    t = _base(info)
    code, inp, out = str(row["code"]), str(row["input"]), str(row["output"])
    t.update(
        prompt=("You are given a Python function and an input. Predict the exact return value of the call. "
                "Give the output as a Python literal inside a code block.\n\n" + _fence(code) + f"\n\nInput: `f({inp})`\nOutput:"),
        raw_prompt=code, tests=f"assert f({inp}) == {out}", entry_point="f", reference=out, runnable=True,
        notes="Output-prediction task; the sandbox extracts the predicted literal from the response and checks f(<input>) == <prediction>.",
        meta={"input": inp, "output": out},
    )
    return t


def _build_hex(info: Dict[str, Any]) -> Dict[str, Any]:
    """hex (dim7) from THUDM/humaneval-x; only the Python subset is runnable."""
    row = _humaneval_x().get(info["item"])
    if row is None:
        return _missing(info, f"{info['item']} not in THUDM/humaneval-x")
    t = _base(info)
    lang = info["language"]
    raw = str(row["prompt"])
    fence_lang = {"python": "python", "cpp": "cpp", "java": "java", "js": "javascript", "go": "go"}.get(lang, lang)
    entry = _entry_from_def(str(row.get("declaration") or "")) if lang == "python" else None
    t.update(
        prompt=f"Complete the following {fence_lang} function.\n\n" + _fence(raw, fence_lang),
        raw_prompt=raw, tests=str(row["test"]), entry_point=entry, reference=str(row["canonical_solution"]),
        runnable=(lang == "python"), language=lang,
        notes=("HumanEval-X python test defines and calls check()" if lang == "python" else f"HumanEval-X {lang}: no toolchain in the sandbox, runnable=False"),
        meta={"hf_task_id": info["item"], "declaration": _str(row.get("declaration")), "example_test": _str(row.get("example_test"))},
    )
    return t


@functools.lru_cache(maxsize=4096)
def _decode_lcb_private_cached(s: str) -> Tuple[Optional[Tuple[Dict[str, Any], ...]], bool]:
    """Memoised body of ``_decode_lcb_private`` (the decode dominates ``load_task`` for lcb ids)."""
    try:
        cases = json.loads(pickle.loads(zlib.decompress(base64.b64decode(s.encode("utf-8")))))
        return tuple(dict(c) for c in cases), True
    except Exception:
        try:
            cases = json.loads(s)
            return tuple(dict(c) for c in cases), True
        except Exception:
            return None, False


def _decode_lcb_private(s: Any) -> Tuple[Optional[List[Dict[str, Any]]], bool]:
    """Decode LiveCodeBench private tests (base64 -> zlib -> pickle -> JSON list); ``(cases, ok)``."""
    if not s:
        return None, False
    cases, ok = _decode_lcb_private_cached(str(s))
    return ([dict(c) for c in cases] if cases is not None else None), ok


def _build_lcb(info: Dict[str, Any]) -> Dict[str, Any]:
    """lcb (dim2 code generation, dim9 test generation) from livecodebench/code_generation_lite."""
    row = _lcb().get(info["item"])
    if row is None:
        return _missing(info, f"question_id {info['item']} not in livecodebench/code_generation_lite")
    t = _base(info)
    try:
        public = [dict(c) for c in json.loads(row.get("public_test_cases") or "[]")]
    except Exception:
        public = []
    private, priv_ok = _decode_lcb_private(row.get("private_test_cases"))
    try:
        meta = json.loads(row.get("metadata") or "{}")
    except Exception:
        meta = {}
    func_name = meta.get("func_name")
    starter = str(row.get("starter_code") or "")
    ttype = "functional" if (public and public[0].get("testtype") == "functional") or (row.get("platform") == "leetcode" and func_name) else "stdin"
    question = str(row["question_content"])
    if ttype == "functional":
        prompt = (f"Solve the following problem in Python. Use the starter code below and implement the method `{func_name}`; "
                  f"return the complete `Solution` class.\n\n{question}\n\n" + _fence(starter))
    else:
        prompt = ("Solve the following problem in Python. Read the input from standard input and write the answer to standard output.\n\n" + question)
    tests = {"type": ttype, "func_name": func_name, "public": public, "private": private, "private_decoded": priv_ok,
             "n_public": len(public), "n_private": len(private or [])}
    if info.get("variant") == "testgen":
        t["prompt"] = ("Write test cases for the following programming problem: give several input/expected-output pairs, "
                       "including edge cases, in the problem's I/O format. Do not solve the problem.\n\n" + question
                       + (("\n\n" + _fence(starter)) if starter else ""))
        t["runnable"] = False
        t["notes"] = f"dim9 test generation: no execution-based verification; official {ttype} tests kept for reference (private decoded={priv_ok})."
    else:
        t["prompt"] = prompt
        t["runnable"] = True
        t["notes"] = (f"LCB {row.get('platform')} {ttype} tests: {len(public)} public + {len(private or [])} private "
                      f"(private decoded={priv_ok}); leetcode responses are called via Solution().{func_name} or a bare function." if ttype == "functional"
                      else f"LCB {row.get('platform')} stdin/stdout tests: {len(public)} public + {len(private or [])} private (private decoded={priv_ok}).")
    t.update(raw_prompt=question, tests=tests, entry_point=func_name, reference=None,
             meta={"platform": row.get("platform"), "difficulty": row.get("difficulty"), "contest_date": row.get("contest_date"),
                   "starter_code": starter, "question_title": row.get("question_title"), "file": row.get("_file")})
    return t


def _build_ds1000(info: Dict[str, Any]) -> Dict[str, Any]:
    """ds1000 (dim6) from xlangai/DS-1000; runnable iff every module in code_context is importable."""
    row = _ds1000().get(int(info["item"]))
    if row is None:
        return _missing(info, f"problem_id {info['item']} not in xlangai/DS-1000")
    t = _base(info)
    ctx = str(row["code_context"])
    patches: List[str] = []
    if "delim_whitespace=True" in ctx:  # removed in pandas 3.0; documented as equivalent to sep=r"\s+"
        ctx = ctx.replace("delim_whitespace=True", 'sep=r"\\s+"')
        patches.append("delim_whitespace=True -> sep=r'\\s+' (argument removed in pandas 3.0)")
    mods = imported_modules(ctx)
    lib = row["metadata"].get("library")
    lib_mod = DS1000_LIBRARY_MODULES.get(str(lib))
    if lib_mod and lib_mod not in mods:
        mods.append(lib_mod)
    missing = [m for m in mods if not lib_available(m)]
    t.update(
        prompt=str(row["prompt"]), raw_prompt=str(row["prompt"]), tests=ctx, entry_point=None, reference=str(row["reference_code"]),
        libs=mods, runnable=not missing,
        notes=(f"DS-1000 {lib}: code_context defines test_execution(solution)" + (" and test_string(solution)" if "def test_string" in ctx else "")
               + (f"; NOT runnable: missing modules {missing}" if missing else "; all modules importable")
               + (f"; harness patched: {'; '.join(patches)}" if patches else "")),
        meta=dict(row["metadata"], missing_libs=missing, harness_patches=patches),
    )
    return t


def _build_multiple(info: Dict[str, Any]) -> Dict[str, Any]:
    """multiple (dim7) from nuprl/MultiPL-E (rust / typescript); prompt + tests, runnable=False."""
    row = _multiple().get((info["language"], info["item"]))
    if row is None:
        return _missing(info, f"({info['language']}, {info['item']}) not in nuprl/MultiPL-E")
    t = _base(info)
    lang = {"rs": "rust", "ts": "typescript"}.get(info["language"], info["language"])
    raw = str(row["prompt"])
    t.update(prompt=f"Complete the following {lang} function.\n\n" + _fence(raw, lang), raw_prompt=raw, tests=str(row["tests"]),
             entry_point=None, reference=None, runnable=False, language=info["language"],
             notes=f"MultiPL-E {lang}: tests available but no {lang} toolchain in the sandbox (runnable=False).",
             meta={"stop_tokens": row.get("stop_tokens"), "original": _str(row.get("original"))})
    return t


def _build_debugbench(info: Dict[str, Any]) -> Dict[str, Any]:
    """debugbench (dim3) from Rtian/DebugBench eval.json rows; prompt only."""
    rows = _debugbench()
    n = int(info["item"])
    idx = (n - 1000) // 2 if info.get("variant") == "sup" else n
    if not 0 <= idx < len(rows):
        return _missing(info, f"row {idx} outside eval.json (len {len(rows)})")
    row = rows[idx]
    t = _base(info)
    lang = str(row.get("language") or "")
    fence_lang = {"python3": "python", "cpp": "cpp", "java": "java"}.get(lang, lang)
    examples = row.get("examples") or []
    if not isinstance(examples, list):
        examples = [str(examples)]
    prompt = (f"The following {fence_lang} solution to a LeetCode problem contains one or more bugs. Fix the code and return the corrected solution.\n\n"
              f"Problem: {row.get('question', '')}\n\n" + ("Examples:\n" + "\n\n".join(str(e) for e in examples) + "\n\n" if examples else "")
              + (f"Constraints:\n{row.get('constraints')}\n\n" if row.get("constraints") else "") + "Buggy code:\n" + _fence(str(row.get("buggy_code") or ""), fence_lang))
    t.update(prompt=prompt, raw_prompt=str(row.get("buggy_code") or ""), tests=None, entry_point=None, reference=_str(row.get("solution")),
             runnable=False, language=fence_lang,
             notes=("DebugBench eval.json row " + (f"(n-1000)//2 = {idx} for sup id" if info.get("variant") == "sup" else f"n = {idx}")
                    + "; index mapping has medium confidence (see module docstring); hidden LeetCode tests -> runnable=False."),
             meta={"row": idx, "slug": row.get("slug"), "category": row.get("category"), "subtype": row.get("subtype"), "level": row.get("level"),
                   "bug_explanation": _str(row.get("bug_explanation"))})
    return t


def _build_bugs2fix(info: Dict[str, Any]) -> Dict[str, Any]:
    """bugs2fix (dim5) from code_x_glue_cc_code_refinement medium/test; prompt only."""
    df = _bugs2fix()
    n = int(info["item"])
    idx = n - 7000 if info.get("variant") == "sup" else n
    if not 0 <= idx < len(df):
        return _missing(info, f"row {idx} outside medium/test (len {len(df)})")
    row = df.iloc[idx]
    t = _base(info)
    buggy = str(row["buggy"]).strip()
    t.update(prompt=("The following Java method (identifiers abstracted as TYPE_n / VAR_n / METHOD_n / STRING_n) contains a bug. "
                     "Return the corrected method in the same abstracted form.\n\n" + _fence(buggy, "java")),
             raw_prompt=buggy, tests=None, entry_point=None, reference=str(row["fixed"]).strip(), runnable=False, language="java",
             notes=("Bugs2Fix (CodeXGLUE code refinement, medium) row " + (f"n-7000 = {idx} for sup id" if info.get("variant") == "sup" else f"n = {idx}")
                    + "; verified by token overlap with recorded responses (37/40); no tests -> runnable=False."),
             meta={"row": idx, "id": int(row["id"])})
    return t


def _build_codexglue(info: Dict[str, Any]) -> Dict[str, Any]:
    """codexglue (dim8) from code_x_glue_ct_code_to_text python/test; prompt only."""
    df = _codexglue()
    n = int(info["item"])
    idx = n - 15000 if info.get("variant") == "sup" else n
    if not 0 <= idx < len(df):
        return _missing(info, f"row {idx} outside python/test (len {len(df)})")
    row = df.iloc[idx]
    t = _base(info)
    code = str(row["code"])
    t.update(prompt="Summarise what the following Python function does in one or two sentences.\n\n" + _fence(code),
             raw_prompt=code, tests=None, entry_point=None, reference=str(row["docstring"]), runnable=False,
             notes=("CodeXGLUE code-to-text (python/test) row " + (f"n-15000 = {idx} for sup id" if info.get("variant") == "sup" else f"n = {idx}")
                    + "; medium-confidence index mapping; summarisation task -> runnable=False."),
             meta={"row": idx, "func_name": _str(row.get("func_name")), "repo": _str(row.get("repo")), "path": _str(row.get("path"))})
    return t


def _build_canitedit(info: Dict[str, Any]) -> Dict[str, Any]:
    """canitedit (dim5) from nuprl/CanItEdit; prompt + tests, runnable=False per the shared spec."""
    row = _canitedit().get(int(info["item"]))
    if row is None:
        return _missing(info, f"id {info['item']} not in nuprl/CanItEdit (HF snapshot has 105 ids)")
    t = _base(info)
    instr = str(row.get("instruction_descriptive") or row.get("instruction_lazy") or "")
    before = str(row["before"])
    t.update(prompt=f"Edit the following Python code as instructed and return the complete edited code.\n\nInstruction: {instr}\n\n" + _fence(before),
             raw_prompt=before, tests=_str(row.get("tests")), entry_point=None, reference=_str(row.get("after")), runnable=False,
             notes="CanItEdit before/instruction_descriptive/tests/after; runnable=False per the shared spec (tests kept).",
             meta={"name": _str(row.get("name")), "instruction_lazy": _str(row.get("instruction_lazy")), "taxonomy": _str(row.get("taxonomy"))})
    return t


def _build_swebench(info: Dict[str, Any]) -> Dict[str, Any]:
    """swebench (dim3) from SWE-bench Verified / Lite; problem statement only."""
    row = _swebench().get(info["item"])
    if row is None:
        return _missing(info, f"instance_id {info['item']} not in SWE-bench Verified/Lite")
    t = _base(info)
    ps = str(row["problem_statement"])
    t.update(prompt=f"Repository: {row['repo']} (base commit {row['base_commit']}).\n\nResolve the following issue and provide a patch.\n\n{ps}",
             raw_prompt=ps, tests=None, entry_point=None, reference=_str(row.get("patch")), runnable=False,
             notes=f"SWE-bench {row.get('_repo')} problem_statement; repository-level task -> runnable=False.",
             meta={"repo": row["repo"], "base_commit": row["base_commit"], "FAIL_TO_PASS": _str(row.get("FAIL_TO_PASS")),
                   "PASS_TO_PASS": _str(row.get("PASS_TO_PASS")), "version": _str(row.get("version"))})
    return t


_BUILDERS = {
    "humaneval": _build_humaneval, "he": _build_humaneval, "mbpp": _build_mbpp, "bcb": _build_bcb,
    "cruxeval": _build_cruxeval, "hex": _build_hex, "lcb": _build_lcb, "ds1000": _build_ds1000,
    "multiple": _build_multiple, "debugbench": _build_debugbench, "bugs2fix": _build_bugs2fix,
    "codexglue": _build_codexglue, "canitedit": _build_canitedit, "swebench": _build_swebench,
}


# ---------------------------------------------------------------------------------------- public


@functools.lru_cache(maxsize=1)
def recorded_tasks() -> Dict[str, Dict[str, Any]]:
    path = os.environ.get("ROUTE_ANYTHING_TASKS", os.path.join(os.path.dirname(__file__), "tasks.jsonl"))
    if not os.path.isfile(path):
        return {}
    return {record["task_id"]: record for record in _read_jsonl(path)}


def load_task(task_id: str, strict: bool = False) -> Dict[str, Any]:
    """Reconstruct one CodeRouterBench task.

    Returns a dict with the keys in ``TASK_KEYS``: ``task_id, found, source, dim, dimension, variant,
    language, prompt, raw_prompt, tests, entry_point, reference, libs, runnable, notes, meta``.
    Unknown sources or unresolvable ids return ``found=False`` (``prompt=None``, ``runnable=False``)
    unless ``strict`` is true, in which case ``KeyError`` is raised.  ``dimension`` comes from
    ``id_tasks.jsonl`` when the id is listed there, else from the ``dimN`` prefix.
    """
    recorded = recorded_tasks().get(task_id)
    if recorded is not None:
        if strict and not recorded["found"]:
            raise KeyError(recorded["notes"])
        return dict(recorded)
    info = parse_task_id(task_id)
    builder = _BUILDERS.get(info["source"])
    if builder is None:
        t = _missing(info, f"unknown source {info['source']!r}")
    else:
        try:
            t = builder(info)
        except SourceUnavailable as e:
            t = _missing(info, str(e))
    meta_row = _id_task_index().get(task_id) if os.path.exists(ID_TASKS_PATH) else None
    if meta_row:
        t["dimension"] = meta_row.get("dimension", t["dimension"])
        t["split"] = meta_row.get("split")
        t["source_split"] = meta_row.get("source_split")
    if strict and not t["found"]:
        raise KeyError(t["notes"])
    return t


def _norm_dims(dimensions: Optional[Iterable[str]]) -> Optional[set]:
    """Accept ``dim1`` labels or dimension names; return the set of ``dimN`` labels (or ``None``)."""
    if dimensions is None:
        return None
    out = set()
    name_to_dim = {v: k for k, v in DIMENSION_NAMES.items()}
    for d in dimensions:
        out.add(name_to_dim.get(d, d))
    return out


def iter_tasks(dimensions: Optional[Iterable[str]] = None, sources: Optional[Iterable[str]] = None, split: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    """Yield ``load_task`` dicts for the ids in ``id_tasks.jsonl`` matching the filters.

    ``dimensions`` may hold ``dimN`` labels or names (``code_generation`` ...), ``sources`` short
    source names (``humaneval``, ``bcb`` ...), ``split`` one of ``probing`` / ``id_test``.
    """
    dims = _norm_dims(dimensions)
    srcs = set(sources) if sources is not None else None
    for row in _id_tasks():
        tid = row["task_id"]
        parts = tid.split("_")
        if dims is not None and parts[0] not in dims:
            continue
        if srcs is not None and parts[1] not in srcs:
            continue
        if split is not None and row.get("split") != split:
            continue
        yield load_task(tid)


def list_task_ids(dimensions: Optional[Iterable[str]] = None, sources: Optional[Iterable[str]] = None, split: Optional[str] = None) -> List[str]:
    """Task ids from ``id_tasks.jsonl`` matching the same filters as ``iter_tasks`` (no loading)."""
    dims = _norm_dims(dimensions)
    srcs = set(sources) if sources is not None else None
    out: List[str] = []
    for row in _id_tasks():
        tid = row["task_id"]
        parts = tid.split("_")
        if dims is not None and parts[0] not in dims:
            continue
        if srcs is not None and parts[1] not in srcs:
            continue
        if split is not None and row.get("split") != split:
            continue
        out.append(tid)
    return out


__all__ = [
    "ACR", "HF_HUB", "ID_TASKS_PATH", "DIMENSION_NAMES", "SOURCES", "RUNNABLE_SOURCES", "TASK_KEYS", "FIELD_MAP",
    "SourceUnavailable", "parse_task_id", "load_task", "iter_tasks", "list_task_ids", "lib_available", "imported_modules",
]

