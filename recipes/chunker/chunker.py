"""Jev semantic chunker: Jev reads windows of numbered sentences and judges, per gap, whether the
next sentence starts a new topic. Cutting is a pure function of those gap probabilities, so one set
of Jev requests serves every threshold and size bound."""
from __future__ import annotations
import re
from concurrent.futures import ThreadPoolExecutor
from jevqu.jev import MAX_QUESTIONS, noul

BOUNDARY = re.compile(r"(?<=[.!?])\s+|\s*\n\s*")
MAX_SENTENCE_CHARS = 500  # ponytail: long sentences are truncated in the request only, to bound window size

def sentences(text: str) -> list[tuple[int, int]]:
    """(start, end) character spans of the sentences in `text`, without surrounding whitespace."""
    spans, start = [], 0
    for m in BOUNDARY.finditer(text):
        if m.start() > start:
            spans.append((start, m.start()))
        start = m.end()
    if text[start:].strip():
        spans.append((start, len(text.rstrip())))
    return spans

def windows(n: int, window: int) -> list[tuple[int, list[int]]]:
    """Overlapping windows over n sentences, stride window // 2. Each gap g (between sentences g and
    g + 1) goes to the window whose centre is nearest, so it is judged with context on both sides.
    Returns (first sentence of the window, gaps it asks about)."""
    if n < 2:
        return []
    w = min(window, n)
    starts = list(range(0, n - w + 1, max(1, w // 2)))
    if starts[-1] + w < n:
        starts.append(n - w)
    out = {s: [] for s in starts}
    for g in range(n - 1):
        s = min(starts, key=lambda s: abs(s + (w - 1) / 2 - (g + 0.5)))
        assert s <= g and g + 1 < s + w, (g, s, w)
        out[s].append(g)
    return [(s, gaps) for s, gaps in out.items() if gaps]

def gap_question(k: int) -> dict:
    return noul(f"Does sentence {k + 1} start a new topic?",
                f"Sentence {k + 1} moves to a different subject from sentence {k}: a new section, story, question or theme begins",
                f"Sentence {k + 1} continues the subject of sentence {k}")

def gap_probs(texts: list[str], jev, window: int = 40, workers: int = 8) -> list[float]:
    """P(sentence g + 1 starts a new topic) for every gap g = 0 .. len(texts) - 2, one Jev request per window."""
    plan = windows(len(texts), window)

    def ask(item):
        s, gaps = item
        w = min(window, len(texts))
        state = {"sentences": [{"sentence": k + 1, "text": t[:MAX_SENTENCE_CHARS]}
                               for k, t in enumerate(texts[s:s + w])]}
        qs = {f"gap_{g - s + 1}": gap_question(g - s + 1) for g in gaps}  # sentence numbers are 1-based in the window
        assert len(qs) <= MAX_QUESTIONS
        ans = jev.ask(state, qs)
        return {g: ans[f"gap_{g - s + 1}"]["noul"] for g in gaps}

    probs = {}
    with ThreadPoolExecutor(workers) as ex:
        for part in ex.map(ask, plan):
            probs.update(part)
    return [probs[g] for g in range(len(texts) - 1)]

def cut(probs: list[float], sizes: list[int], threshold: float, min_size: int = 0, max_size: int = 10**9) -> list[int]:
    """Indices of the sentences that open a chunk (always 0). Cut at gap g when P >= threshold and the
    current chunk has reached min_size. When the next sentence would push the chunk past max_size, cut
    instead at the most likely gap inside the chunk that leaves at least min_size before it (the
    overflow point if none does), and rescan from that cut."""
    assert len(probs) == max(0, len(sizes) - 1)
    if not sizes:
        return []
    starts, a, cur, g = [0], 0, sizes[0], 0
    while g < len(probs):
        if probs[g] >= threshold and cur >= min_size:
            starts.append(g + 1)
            a, cur = g + 1, sizes[g + 1]
        elif cur + sizes[g + 1] > max_size:
            left, cands = 0, []
            for h in range(a, g + 1):
                left += sizes[h]
                if left >= min_size:
                    cands.append(h)
            h = max(cands, key=lambda h: (probs[h], h)) if cands else g
            starts.append(h + 1)
            a = g = h + 1
            cur = sizes[a]
            continue
        else:
            cur += sizes[g + 1]
        g += 1
    return starts

def chunks(text: str, spans: list[tuple[int, int]], starts: list[int]) -> list[str]:
    """Chunk strings as exact slices of `text`, from the sentence spans and the chunk starts."""
    ends = starts[1:] + [len(spans)]
    return [text[spans[a][0]:spans[b - 1][1]] for a, b in zip(starts, ends)]
