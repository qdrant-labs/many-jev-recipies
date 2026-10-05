"""QASPER (Dasigi et al., NAACL 2021; CC BY 4.0) for the chunker benchmark: NLP papers, questions
written by researchers who read only the title and abstract, and the paragraphs that answer them,
marked by other researchers who read the full paper. Answer paragraphs become character spans in the
paper text, and a chunker's retrieved spans are scored against them."""
from __future__ import annotations
import random, urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "qasper"
URL = "https://huggingface.co/api/datasets/allenai/qasper/parquet/qasper/{split}/0.parquet"

def paper_text(p: dict) -> str:
    """Title, abstract, then each section name and its paragraphs, separated by blank lines."""
    parts = [p["title"], p["abstract"]]
    for name, paras in zip(p["full_text"]["section_name"], p["full_text"]["paragraphs"]):
        parts += [name or "", *paras]
    return "\n\n".join(x.strip() for x in parts if x and x.strip())

def questions(p: dict, text: str) -> list[dict]:
    """Questions whose evidence, pooled over annotators, is text found verbatim in `text`. Questions
    with no text evidence (unanswerable, or answered by a table or figure) or with evidence missing
    from the text are dropped."""
    out = []
    for qid, q, ans in zip(p["qas"]["question_id"], p["qas"]["question"], p["qas"]["answers"]):
        evs = {e.strip() for a in ans["answer"] for e in a["evidence"] if not e.startswith("FLOAT SELECTED")} - {""}
        refs = sorted((i, i + len(e)) for e in evs if (i := text.find(e)) >= 0)
        if evs and len(refs) == len(evs):
            out.append({"id": qid, "paper": p["id"], "question": q, "refs": refs})
    return out

def load(split: str = "test", n_papers: int = 100, seed: int = 0) -> tuple[dict[str, str], list[dict]]:
    """A seeded sample of papers that have at least one usable question: ({paper id: text}, questions)."""
    import pyarrow.parquet as pq
    path = DATA / f"{split}.parquet"
    if not path.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL.format(split=split), path)
    papers, qs = {}, {}
    for p in pq.read_table(path).to_pylist():
        text = paper_text(p)
        if found := questions(p, text):
            papers[p["id"]], qs[p["id"]] = text, found
    keep = sorted(random.Random(seed).sample(sorted(papers), min(n_papers, len(papers))))
    return {k: papers[k] for k in keep}, [q for k in keep for q in qs[k]]

def scores(retrieved: list[tuple[int, int]], refs: list[tuple[int, int]]) -> dict[str, float]:
    """Character overlap of retrieved spans with reference spans: recall, precision and IoU."""
    got = set().union(*(range(a, b) for a, b in retrieved))
    ref = set().union(*(range(a, b) for a, b in refs))
    inter = len(got & ref)
    return {"recall": inter / len(ref), "precision": inter / len(got) if got else 0.0, "iou": inter / len(got | ref)}
