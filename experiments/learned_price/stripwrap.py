"""Wrapper stripping for task B1 (templates learned on the 7,080 probing prompts only; test prompts never used).

Templates.  A line (or sentence) is a template of source s if its normalized form occurs in at least
max(tau * n_s, MIN_COUNT) of the source's n_s probing prompts (document frequency; MIN_COUNT = 3 guards the
16-prompt canitedit source).  Normalization: strip, collapse whitespace, mask inline code spans `...` -> `_`
(so slot-filled wrappers such as "... implement the method `countPairs`; ..." collapse to one template);
code-fence lines (```lang) are kept literally.  Sentences are split inside lines at [.!?;:] + whitespace and
are counted on the lines that are not line templates.
Affixes.  The source's longest common prefix / suffix of the RAW probing prompts, and the longest common
prefix / suffix of the RESIDUAL probing prompts (after template removal), both snapped to whole tokens
(a word is never cut), are removed after the template lines / sentences.
Order of application: template lines -> template sentences -> raw prefix / suffix -> residual prefix / suffix.

Two application modes:
  per_source : a prompt of source s is stripped with s's templates and affixes (the source label of the prompt
               is used by this pre-processing only, never by the downstream states);
  union      : every prompt is stripped with the union of all sources' templates and the longest matching affix
               among all sources' -- one fixed rule for every prompt, so no provenance label is needed when the
               rule is applied (deployable).
"""
import re
from collections import Counter

FENCE_RE = re.compile(r'^\s*```')
INLINE_RE = re.compile(r'`[^`\n]+`')
WS_RE = re.compile(r'\s+')
SENT_RE = re.compile(r'(?<=[.!?;:])\s+')
BLANKS_RE = re.compile(r'\n{3,}')


def norm(s):
    s = s.strip()
    if FENCE_RE.match(s):
        return WS_RE.sub(' ', s)
    return WS_RE.sub(' ', INLINE_RE.sub('`_`', s)).strip()


def sentences(line):
    return [p for p in SENT_RE.split(line) if p.strip()]


def learn_lines_sents(texts, tau, min_count):
    n = max(len(texts), 1)
    thr = max(tau * n, min_count)
    cl = Counter()
    for t in texts:
        cl.update({norm(l) for l in t.split('\n') if l.strip()})
    TL = {k for k, c in cl.items() if c >= thr}
    cs = Counter()
    for t in texts:
        ss = set()
        for l in t.split('\n'):
            if not l.strip() or norm(l) in TL:
                continue
            ss.update(norm(p) for p in sentences(l))
        cs.update(ss)
    TS = {k for k, c in cs.items() if c >= thr}
    return TL, TS, {k: c / n for k, c in cl.items() if k in TL}, {k: c / n for k, c in cs.items() if k in TS}


def remove_lines_sents(text, TL, TS):
    out = []
    for l in text.split('\n'):
        if not l.strip():
            out.append('')
            continue
        if norm(l) in TL:
            continue
        ps = sentences(l)
        keep = [p for p in ps if norm(p) not in TS]
        if len(keep) == len(ps):
            out.append(l)
        elif keep:
            out.append(' '.join(keep))
    return BLANKS_RE.sub('\n\n', '\n'.join(out)).strip()


def _lcp(texts):
    if not texts:
        return ''
    a, b = min(texts), max(texts)
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    return a[:i]


def snap_prefix(p, texts):
    """Cut a common prefix back to a token boundary (never split a word)."""
    if not p:
        return ''
    if p[-1].isspace() or all(len(t) == len(p) or t[len(p)].isspace() for t in texts):
        return p
    k = max(p.rfind(c) for c in ' \n\t')
    return p[:k + 1] if k >= 0 else ''


def snap_suffix(s, texts):
    if not s:
        return ''
    if s[0].isspace() or all(len(t) == len(s) or t[-len(s) - 1].isspace() for t in texts):
        return s
    ks = [i for i in (s.find(' '), s.find('\n'), s.find('\t')) if i >= 0]
    return s[min(ks):] if ks else ''


def common_affixes(texts):
    texts = [t for t in texts]
    p = snap_prefix(_lcp(texts), texts)
    s = snap_suffix(_lcp([t[::-1] for t in texts])[::-1], texts)
    return (p if p.strip() else ''), (s if s.strip() else '')


def strip_affixes(t, prefixes, suffixes):
    for p in prefixes:
        if p and t.startswith(p) and len(t) > len(p):
            t = t[len(p):]
            break
    for s in suffixes:
        if s and t.endswith(s) and len(t) > len(s):
            t = t[:-len(s)]
            break
    return t.strip()


def _longest_first(xs):
    return sorted({x for x in xs if x}, key=len, reverse=True)


def learn(probing_texts_by_source, tau, min_count=3):
    """Per-source rules and the source-agnostic union rule, all learned from probing prompts."""
    per = {}
    for s, texts in probing_texts_by_source.items():
        rp, rs = common_affixes(texts)
        TL, TS, fl, fs = learn_lines_sents(texts, tau, min_count)
        resid = [strip_affixes(remove_lines_sents(t, TL, TS), [rp], [rs]) for t in texts]
        p, q = common_affixes(resid)
        per[s] = dict(TL=TL, TS=TS, line_df=fl, sent_df=fs, raw_prefix=rp, raw_suffix=rs, prefix=p, suffix=q)
    TLu = set().union(*[v['TL'] for v in per.values()])
    TSu = set().union(*[v['TS'] for v in per.values()])
    raw_p = _longest_first(v['raw_prefix'] for v in per.values()); raw_s = _longest_first(v['raw_suffix'] for v in per.values())
    res_p, res_s = {}, {}
    for s, texts in probing_texts_by_source.items():
        resid = [strip_affixes(remove_lines_sents(t, TLu, TSu), raw_p, raw_s) for t in texts]
        res_p[s], res_s[s] = common_affixes(resid)
    union = dict(TL=TLu, TS=TSu, raw_prefixes=raw_p, raw_suffixes=raw_s,
                 prefixes=_longest_first(res_p.values()), suffixes=_longest_first(res_s.values()))
    return per, union


def apply_per_source(text, source, per):
    r = per[source]
    t = strip_affixes(remove_lines_sents(text, r['TL'], r['TS']), [r['raw_prefix']], [r['raw_suffix']])
    return strip_affixes(t, [r['prefix']], [r['suffix']])


def apply_union(text, union):
    t = strip_affixes(remove_lines_sents(text, union['TL'], union['TS']), union['raw_prefixes'], union['raw_suffixes'])
    return strip_affixes(t, union['prefixes'], union['suffixes'])
