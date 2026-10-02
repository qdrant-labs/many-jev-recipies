from __future__ import annotations
import math, random, re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import numpy as np
from jevqu.jev import MAX_QUESTIONS, choice, noul
from jevqu.schema import ClassDef, Facet

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
STOP = set(ENGLISH_STOP_WORDS)  # "was", "been", "about" make useless class names

def _kmeans(x: np.ndarray, k: int, seed: int, iters: int = 20) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
    cent = x[rng.choice(len(x), size=min(k, len(x)), replace=False)]
    for _ in range(iters):
        lab = np.argmax(x @ cent.T, axis=1)
        for j in range(len(cent)):
            m = x[lab == j]
            if len(m):
                cent[j] = m.mean(axis=0)
    return np.argmax(x @ cent.T, axis=1)

def bge_small():
    """Default name encoder: the small transformer the collection is usually embedded with."""
    from fastembed import TextEmbedding
    model = TextEmbedding("BAAI/bge-small-en-v1.5")
    return lambda texts: np.array(list(model.query_embed(list(texts))))

def _unit(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)

def _phrases(text: str, n: int = 3) -> set[str]:
    """1-3-word phrases between stop words, punctuation and tokens with digits (RAKE-style candidates)."""
    out, run = set(), []
    for t in re.findall(r"[a-z0-9]+|[^\sa-z0-9]", text.lower()) + ["."]:
        if t.isalpha() and len(t) > 2 and t not in STOP:
            run.append(t)
            continue
        out |= {" ".join(run[i:j]) for i in range(len(run)) for j in range(i + 1, min(i + n, len(run)) + 1)}
        run = []
    return out

def _key(phrase: str) -> str:  # crude singular so "dog collars" and "dog collar" are one name
    def one(w):
        if w.endswith("ies"):
            return w[:-3] + "y"
        if w.endswith("sses"):
            return w[:-2]
        return w[:-1] if w.endswith("s") and not w.endswith(("ss", "us")) and len(w) > 3 else w
    return " ".join(one(w) for w in phrase.split())

def name_options(texts: list[str], vectors: np.ndarray, lab: np.ndarray, encoder, n: int = 8,
                 pool: int = 15, central: int = 20) -> list[tuple[list[str], list[int]]]:
    """Per cluster: up to n candidate names from its own text, its most distinctive phrases (tf-idf) and its most
    widespread ones (a broad group's name is often in many members but in other groups too), ranked by encoder
    similarity to a diverse set of its members; plus its items, most central first."""
    x = _unit(vectors)
    docs = [_phrases(t) for t in texts]
    df = Counter(p for d in docs for p in d)
    out = []
    for j in sorted(set(lab)):
        m = np.where(lab == j)[0]
        forms = defaultdict(Counter)
        for ph, c in Counter(ph for i in m for ph in docs[i]).items():
            if c >= 2 or len(m) < 3:
                forms[_key(ph)][ph] += c
        def tfidf(k):
            return sum(forms[k].values()) * math.log(len(docs) / min(len(docs), sum(df[ph] for ph in forms[k])))
        top = set(sorted(forms, key=lambda k: (-tfidf(k), k))[:pool]) | \
            set(sorted(forms, key=lambda k: (-sum(forms[k].values()), k))[:pool])
        opts = sorted(min(forms[k], key=lambda f: (-forms[k][f], f)) for k in top)  # ties: alphabetical, not hash order
        mid = m[np.argsort(-(x[m] @ x[m].mean(axis=0)), kind="stable")]
        if not opts:
            out.append(([], mid.tolist()))
            continue
        ref = _unit(encoder([texts[i][:200] for i in _diverse(x, m, central)])).mean(axis=0)
        score = _unit(encoder(opts)) @ ref
        out.append(([opts[i] for i in sorted(range(len(opts)), key=lambda i: (-score[i], opts[i]))[:n]], mid.tolist()))
    return out

NAME = "Which option best names what most of these items are?"

def _diverse(x: np.ndarray, m: np.ndarray, n: int, diversity: float = 0.5) -> np.ndarray:
    """MMR around the centroid (as Qdrant's MMR query): close to the cluster's centre but unlike each other."""
    rel = x[m] @ _unit(x[m].mean(axis=0))
    picked = [int(np.argmax(rel))]
    sim = x[m] @ x[m][picked[0]]
    while len(picked) < min(n, len(m)):
        score = (1 - diversity) * rel - diversity * sim
        score[picked] = -np.inf
        picked.append(int(np.argmax(score)))
        sim = np.maximum(sim, x[m] @ x[m][picked[-1]])
    return m[picked]

def name_clusters(texts, vectors, states, lab, jev, encoder, n: int = 8, shown: int = 10, fit_n: int = 15,
                  pool_n: int = 3, min_fit: float = 0.4, max_other: float = 0.15, seed: int = 0,
                  workers: int = 8) -> list[dict]:
    """Name each cluster from the encoder's options. Jev picks one after seeing a diverse set of members; the pick
    stands if Jev places >= min_fit of random members under it, else the best-fitting option does, else None.
    Options that more than max_other of other clusters' items fit are never used (too broad to filter on).
    One row per cluster: {name (None = no class), jev_pick, options: {option: (fit, other)}}."""
    x, rng = _unit(vectors), np.random.default_rng(seed)
    ranked = name_options(texts, vectors, lab, encoder, n)
    clusters = sorted(set(lab))
    members = {j: np.where(lab == j)[0] for j in clusters}
    opts = {j: o for j, (o, _) in zip(clusters, ranked)}
    cls = {o: classes_for([o])[0] for os_ in opts.values() for o in os_}
    def ask(i, names):  # one yes/no per name, chunked to the request limit
        qs = [(cls[o].id, _q(cls[o])) for o in names]
        a = {}
        for k in range(0, len(qs), MAX_QUESTIONS):
            a |= jev.ask(states[i], dict(qs[k:k + MAX_QUESTIONS]))
        return {o: a[cls[o].id]["noul"] >= 0.5 for o in names}
    def pick(j):
        if not opts[j]:
            return None
        q = choice(NAME, {o: None for o in opts[j]} | {"none": "No option fits most of the items"})
        a = jev.ask({"items": [texts[i][:150] for i in _diverse(x, members[j], shown)]}, {"kind": q})["kind"]["choice"]
        return None if a == "none" else a
    held = {j: rng.permutation(members[j])[:fit_n] for j in clusters}  # random, so the fit share is unbiased
    jobs = [(j, int(i)) for j in clusters for i in held[j]]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        picks = list(ex.map(pick, clusters))
        fit_ans = list(ex.map(lambda ji: ask(ji[1], opts[ji[0]]), jobs))
    hits = {j: Counter() for j in clusters}
    for (j, _), a in zip(jobs, fit_ans):
        hits[j].update(o for o, yes in a.items() if yes)
    fit = {j: {o: hits[j][o] / max(1, len(held[j])) for o in opts[j]} for j in clusters}  # exact: 6/15 must reach 0.4
    eligible = sorted({o for j in clusters for o, f in fit[j].items() if f >= min_fit})
    pool = [(j, int(i)) for j in clusters for i in held[j][:pool_n]]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        pool_ans = list(ex.map(lambda ji: ask(ji[1], eligible), pool)) if eligible else []
    keys = {j: {_key(o) for o in opts[j]} for j in clusters}
    def other(o):  # share of items from clusters not offering this name that still fit it
        ys = [a[o] for (j, _), a in zip(pool, pool_ans) if _key(o) not in keys[j]]
        return sum(ys) / len(ys) if ys else 0.0
    out = []
    for j, k in zip(clusters, picks):
        oth = {o: other(o) if o in eligible else None for o in opts[j]}
        ok = {o: f for o, f in fit[j].items() if f >= min_fit and oth[o] <= max_other}
        name = k if k in ok else max(ok, key=lambda o: (ok[o], -opts[j].index(o)), default=None)
        out.append({"name": name, "jev_pick": k, "options": {o: (fit[j][o], oth[o]) for o in opts[j]}})
    return out

def classes_for(kinds: list[str | None]) -> list[ClassDef]:
    out, seen = [], set()
    for kind in kinds:
        cid = re.sub(r"[^a-z0-9]+", "-", kind.lower()).strip("-") if kind else ""
        if cid and _key(kind.lower()) not in seen:  # clusters given the same name (any spelling) become one class
            seen.add(_key(kind.lower()))
            out.append(ClassDef(cid, kind.title(), f"The item is a kind of {kind} or is mainly about {kind}",
                                "The item is a different kind of thing, or is about something else"))
    return out

def propose(texts: list[str], vectors: np.ndarray, states: list, jev, encoder, k: int = 40, seed: int = 0,
            workers: int = 8) -> list[ClassDef]:
    vectors = np.asarray(vectors, dtype=float)
    rows = name_clusters(texts, vectors, states, _kmeans(vectors, k, seed), jev, encoder, workers=workers)
    return classes_for([r["name"] for r in rows])

def _q(c: ClassDef) -> dict:
    return noul(f"Does this item belong to the class `{c.name}`?", c.definition, c.exclusions)

def coverage(cands: list[ClassDef], states: list[dict], jev, workers: int = 8) -> dict[str, list[float]]:
    """P(member) of every sampled item for every candidate: one request per item, all candidates packed."""
    qs = {c.id: _q(c) for c in cands}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        answers = list(ex.map(lambda s: jev.ask(s, qs), states))
    return {c.id: [a[c.id]["noul"] for a in answers] for c in cands}

POPULAR = noul("Is this group something people searching this collection would look for by name or purpose?",
               "People searching this collection commonly look for this kind of item",
               "Few searchers would look for this group; it is a niche or an arbitrary mix")
COHERENT = noul("Do the example items form one recognizable kind of item or topic?",
                "The examples are clearly the same kind of item or topic",
                "The examples are a mix of unrelated items")

def judge(cands: list[ClassDef], cov: dict[str, list[float]], texts: list[str], jev, context: int = 30,
          examples: int = 5, seed: int = 0, workers: int = 8) -> dict[str, tuple[float, float]]:
    """Jev's view of each group from the collection alone: (would searchers look for it, is it one kind of item)."""
    rng = random.Random(seed)
    context_items = [texts[i][:100] for i in rng.sample(range(len(texts)), min(context, len(texts)))]
    def one(c: ClassDef):
        members = [texts[i][:100] for i, p in enumerate(cov[c.id]) if p >= 0.5][:examples]
        a = jev.ask({"collection_items": context_items, "group": {"name": c.name, "definition": c.definition, "examples": members}},
                    {"popular": POPULAR, "coherent": COHERENT})
        return c.id, (a["popular"]["noul"], a["coherent"]["noul"])
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return dict(ex.map(one, cands))

def _share(ps: list[float]) -> float:
    return sum(p >= 0.5 for p in ps) / max(1, len(ps))

def _yes(ps: list[float]) -> set[int]:
    return {i for i, p in enumerate(ps) if p >= 0.5}

def _duplicates(cands: list[ClassDef], cov, ids: set[str], threshold: float = 0.5, first: set[str] = frozenset()) -> dict[str, str]:
    """Classes Jev accepts mostly the same items for (Jaccard > threshold), e.g. synonyms, are one class:
    ids in `first` stay, then the one covering more items. Returns {duplicate id: id it duplicates}."""
    dup, kept = {}, []
    for c in sorted((c for c in cands if c.id in ids), key=lambda c: (c.id not in first, -_share(cov[c.id]), c.id)):
        a = _yes(cov[c.id])
        hit = next((kid for kid, b in kept if a | b and len(a & b) / len(a | b) > threshold), None)
        if hit:
            dup[c.id] = hit
        else:
            kept.append((c.id, a))
    return dup

def _rejected_by(c: ClassDef, cov, facets: list[Facet], band, judged, min_judge: float, dups=None) -> str | None:
    facet_values = {v.lower() for f in facets for v in f.values}
    if not band[0] <= _share(cov[c.id]) <= band[1]:
        return "size"
    if c.name.lower() in facet_values or any(t in facet_values for t in c.id.split("-")):
        return "duplicates a facet"
    if dups and c.id in dups:
        return f"duplicates {dups[c.id]}"
    if judged is not None and min(judged[c.id]) < min_judge:
        return "judge"
    return None

def _dups(cands, cov, facets, band) -> dict[str, str]:
    return _duplicates(cands, cov, {c.id for c in cands if _rejected_by(c, cov, facets, band, None, 0.0) is None})

def select(cands, cov, facets: list[Facet], band=(0.005, 0.40), judged=None, min_judge: float = 0.5) -> list[ClassDef]:
    dups = _dups(cands, cov, facets, band)
    return [c for c in cands if _rejected_by(c, cov, facets, band, judged, min_judge, dups) is None]

def _children(parents: list[ClassDef], pcov, texts, vectors, states, jev, encoder, child_k: int, band, judge_groups: bool,
              min_judge: float, workers: int) -> tuple[list[ClassDef], list[dict]]:
    """Second level: name child_k finer groups of the sample; each goes under the parent that accepts at least half its
    items, and Jev scores it only on that parent's items, as labeling will. A child covering > 90% of its parent adds
    nothing. Returns (kept children, orphans: named groups no parent holds, report rows of the children)."""
    vectors = np.asarray(vectors, dtype=float)
    lab = _kmeans(vectors, child_k, 0)
    rows = name_clusters(texts, vectors, states, lab, jev, encoder, workers=workers)
    yes = {p.id: _yes(pcov[p.id]) for p in parents}
    taken = {_key(p.name.lower()) for p in parents}
    groups = {}  # singular name -> (name, sample items of the groups named so)
    for j, r in zip(sorted(set(lab)), rows):
        if r["name"] and _key(r["name"]) not in taken:
            name, items = groups.get(_key(r["name"]), (r["name"], []))
            groups[_key(r["name"])] = (name, items + np.where(lab == j)[0].tolist())
    kids, orphans, report = [], [], []
    for c, (_, items) in zip(classes_for([n for n, _ in groups.values()]), groups.values()):
        best = max(parents, key=lambda p: len(yes[p.id] & set(items)))
        if len(yes[best.id] & set(items)) >= len(items) / 2:
            kids.append(replace(c, parent=best.id))
        else:
            orphans.append(c)
    kept, report = _score_children(kids, pcov, states, texts, jev, band, judge_groups, min_judge, workers)
    return kept, orphans, report

def _score_children(kids: list[ClassDef], pcov, states, texts, jev, band, judge_groups: bool, min_judge: float,
                    workers: int) -> tuple[list[ClassDef], list[dict]]:
    """Jev scores each child only on the items its parent accepts, as labeling will; then the size, duplicate and judge
    gates. A child covering > 90% of its parent adds nothing."""
    yes = {pid: _yes(pcov[pid]) for pid in {k.parent for k in kids}}
    todo = [(i, [k for k in kids if i in yes[k.parent]]) for i in range(len(states))]
    todo = [(i, ks) for i, ks in todo if ks]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        answers = list(ex.map(lambda t: jev.ask(states[t[0]], {k.id: _q(k) for k in t[1]}), todo))
    cov = {k.id: [0.0] * len(states) for k in kids}
    for (i, _), a in zip(todo, answers):
        for kid, x in a.items():
            cov[kid][i] = x["noul"]
    judged = judge(kids, cov, texts, jev, workers=workers) if judge_groups and kids else None
    dups = _dups(kids, cov, [], band)
    kept, report = [], []
    for k in kids:
        same = len(_yes(cov[k.id])) > 0.9 * len(yes[k.parent])
        reason = "same as parent" if same else _rejected_by(k, cov, [], band, judged, min_judge, dups)
        row = {"id": k.id, "name": k.name, "coverage": _share(cov[k.id]), "kept": reason is None, "rejected_by": reason,
               "parent": k.parent}
        if judged is not None:
            row["popular"], row["coherent"] = judged[k.id]
        report.append(row)
        if reason is None:
            kept.append(k)
    return kept, report

def _promote(parents, pcov, orphans, states, texts, jev, facets, band, judge_groups: bool, min_judge: float,
             room: int, workers: int) -> tuple[list[ClassDef], list[dict]]:
    """Named groups no parent holds become level-1 classes, under the same gates; one that duplicates a parent is dropped."""
    cov = pcov | coverage(orphans, states, jev, workers)
    judged = judge(orphans, cov, texts, jev, workers=workers) if judge_groups else None
    ok = {c.id for c in parents} | {c.id for c in orphans if _rejected_by(c, cov, facets, band, None, 0.0) is None}
    dups = _duplicates(parents + orphans, cov, ok, first={c.id for c in parents})
    kept, report = [], []
    for c in orphans:
        reason = _rejected_by(c, cov, facets, band, judged, min_judge, dups) or (None if len(kept) < room else "max_classes")
        row = {"id": c.id, "name": c.name, "coverage": _share(cov[c.id]), "kept": reason is None, "rejected_by": reason,
               "parent": None, "promoted": True}
        if judged is not None:
            row["popular"], row["coherent"] = judged[c.id]
        report.append(row)
        if reason is None:
            kept.append(c)
    return kept, report

NAME_SYS = ("You name groups of items from one search collection. You get example items from one group. Reply with JSON: "
            '{"name": "...", "definition": "...", "exclusions": "...", "coherent": true|false}. '
            "name: 1 to 3 words, the common noun phrase for the kind of item or topic most examples are, as people searching "
            "would type it; broad enough to cover most examples; never an adjective, brand, size or audience on its own. "
            "definition: one sentence on what belongs. exclusions: one sentence on what does not, naming the closest other kinds. "
            "coherent: false if the examples are a mix with no shared kind.")
PARENT_SYS = ("You organise the groups of one search collection into a two-level taxonomy. You get numbered groups with names "
              "and definitions. Reply with JSON: "
              '{"parents": [{"name": "...", "definition": "...", "exclusions": "...", "children": [group numbers]}]}. '
              "Make 10 to 20 parents, each a broad kind people searching would recognise; put every group under exactly one "
              "parent; a parent may hold a single group. Parent names follow the same rules as group names.")

def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")

def _llm_class(o: dict, parent: str | None = None) -> ClassDef | None:
    name = str(o.get("name") or "").strip()
    return ClassDef(_slug(name), name, str(o.get("definition") or f"The item is a kind of {name}"),
                    str(o.get("exclusions") or "The item is a different kind of thing"), parent) if _slug(name) else None

def llm_classes(texts, vectors, lab, llm, shown: int = 10, random_n: int = 5, workers: int = 4) -> list[ClassDef | None]:
    """The LLM writes one class per cluster from 15 of its items: 10 diverse (MMR) and 5 random."""
    x, rng = _unit(vectors), np.random.default_rng(0)
    def examples(m):
        div = list(_diverse(x, m, shown))
        rest = [int(i) for i in rng.permutation(m) if i not in set(div)][:random_n]
        return "\n".join(f"{n + 1}. {texts[i][:150]}" for n, i in enumerate(div + rest))
    ex = [examples(np.where(lab == j)[0]) for j in sorted(set(lab))]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [_llm_class(o) for o in pool.map(lambda e: llm.ask_json(NAME_SYS, "Example items:\n" + e), ex)]

def _fit_other(named: dict, lab, states, jev, fit_n: int, pool_n: int, seed: int, workers: int) -> tuple[dict, dict]:
    """Per cluster: share of fit_n random members Jev places in its class, and share of other clusters' items (pool_n
    from each) it accepts too. Clusters given the same name count as one."""
    rng = np.random.default_rng(seed)
    held = {j: [int(i) for i in rng.permutation(np.where(lab == j)[0])[:fit_n]] for j in named}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fit_ans = list(ex.map(lambda ji: jev.ask(states[ji[1]], {named[ji[0]].id: _q(named[ji[0]])}),
                              [(j, i) for j in named for i in held[j]]))
    hits = Counter()
    for (j, _), a in zip([(j, i) for j in named for i in held[j]], fit_ans):
        hits[j] += a[named[j].id]["noul"] >= 0.5
    fit = {j: hits[j] / max(1, len(held[j])) for j in named}
    classes = list({c.id: c for c in named.values()}.values())
    pool = [(j, i) for j in named for i in held[j][:pool_n]]
    qs = [(c.id, _q(c)) for c in classes]
    def ask(i):
        a = {}
        for k in range(0, len(qs), MAX_QUESTIONS):
            a |= jev.ask(states[i], dict(qs[k:k + MAX_QUESTIONS]))
        return a
    with ThreadPoolExecutor(max_workers=workers) as ex:
        pool_ans = list(ex.map(lambda ji: ask(ji[1]), pool))
    other = {}
    for j, c in named.items():
        ys = [a[c.id]["noul"] >= 0.5 for (jj, _), a in zip(pool, pool_ans) if named[jj].id != c.id]
        other[j] = sum(ys) / len(ys) if ys else 0.0
    return fit, other

def induce_llm(texts, vectors, states, jev, llm, facets, k: int = 80, band=(0.005, 0.40), min_fit: float = 0.4,
               max_other: float = 0.15, fit_n: int = 15, pool_n: int = 3, judge_groups: bool = False,
               min_judge: float = 0.5, max_classes: int = MAX_QUESTIONS - 1, seed: int = 0,
               workers: int = 8) -> tuple[list[ClassDef], list[dict]]:
    """Items-only induction where an LLM proposes and Jev decides. Cluster the sample into k groups; the LLM names each
    from 15 of its items; Jev keeps a name if it places >= min_fit of random members under it and <= max_other of other
    groups' items; the LLM groups the kept names into parents; Jev gates parents by coverage and scores each child only
    inside its parent, as labeling will. Children of a rejected parent, or of none, become top-level classes."""
    vectors = np.asarray(vectors, dtype=float)
    lab = _kmeans(vectors, k, seed)
    clusters = sorted(set(lab))
    named = {j: c for j, c in zip(clusters, llm_classes(texts, vectors, lab, llm)) if c is not None}
    fit, other = _fit_other(named, lab, states, jev, fit_n, pool_n, seed, workers)
    report, kids = [], {}
    for j, c in named.items():
        reason = None if fit[j] >= min_fit and other[j] <= max_other else "fits few members" if fit[j] < min_fit else "too broad"
        if reason:
            report.append({"id": c.id, "name": c.name, "coverage": 0.0, "kept": False, "rejected_by": reason, "parent": None})
        else:
            kids.setdefault(_key(c.name.lower()), c)  # groups the LLM named alike are one class
    kids = list(kids.values())
    tree = llm.ask_json(PARENT_SYS, "Groups:\n" + "\n".join(f"{n}. {c.name}: {c.definition}" for n, c in enumerate(kids)))
    parents, parent_of = {}, {}
    for p in tree.get("parents") or []:
        pc = _llm_class(p)
        if pc is None:
            continue
        pc = parents.setdefault(_key(pc.name.lower()), pc)
        for n in p.get("children") or []:
            if isinstance(n, int) and 0 <= n < len(kids) and n not in parent_of and _key(kids[n].name.lower()) != _key(pc.name.lower()):
                parent_of[n] = pc.id
    parents = list(parents.values())
    pcov = coverage(parents, states, jev, workers) if parents else {}
    pj = judge(parents, pcov, texts, jev, workers=workers) if judge_groups and parents else None
    pdups = _dups(parents, pcov, facets, band)
    kept = []
    for p in parents:
        reason = _rejected_by(p, pcov, facets, band, pj, min_judge, pdups) or (None if len(kept) < max_classes else "max_classes")
        row = {"id": p.id, "name": p.name, "coverage": _share(pcov[p.id]), "kept": reason is None, "rejected_by": reason, "parent": None}
        if pj is not None:
            row["popular"], row["coherent"] = pj[p.id]
        report.append(row)
        if reason is None:
            kept.append(p)
    ids = {p.id for p in kept}
    under = [replace(c, parent=parent_of[n]) for n, c in enumerate(kids) if parent_of.get(n) in ids]
    orphans = [c for n, c in enumerate(kids) if parent_of.get(n) not in ids and c.id not in ids]
    kept_kids, kid_report = _score_children(under, pcov, states, texts, jev, band, judge_groups, min_judge, workers) \
        if under else ([], [])
    promoted, promo_report = _promote(kept, {p.id: pcov[p.id] for p in kept}, orphans, states, texts, jev, facets, band,
                                      judge_groups, min_judge, max_classes - len(kept), workers) if orphans else ([], [])
    return kept + promoted + kept_kids, report + promo_report + kid_report

def induce(texts, vectors, states, jev, facets, max_classes: int = MAX_QUESTIONS - 1, rounds: int = 1, k: int = 40,
           band=(0.005, 0.40), judge_groups: bool = False, min_judge: float = 0.5, encoder=None, child_k: int | None = None,
           workers: int = 8) -> tuple[list[ClassDef], list[dict]]:
    """Items-only induction: cluster, name each cluster (encoder-ranked options, Jev picks, fit-checked), keep the classes Jev
    places 0.5%-40% of items into (and, with judge_groups, that Jev judges searched-for and coherent), one class per set of
    near-identical ones. With child_k, a second level of finer classes under them; a finer class no parent holds becomes
    a level-1 class itself. No queries."""
    encoder = encoder or bge_small()
    max_classes = min(max_classes, MAX_QUESTIONS - max(1, len(facets)))  # level-1 classes + injection/facet question must fit one request
    kept, kept_cov, report, residual = [], {}, [], list(range(len(texts)))
    for _ in range(rounds):
        if len(kept) >= max_classes or not residual:
            break
        cands = [c for c in propose([texts[i] for i in residual], np.asarray(vectors)[residual],
                                    [states[i] for i in residual], jev, encoder, k=k, workers=workers)
                 if c.id not in {x.id for x in kept}]
        if not cands:
            break
        cov = coverage(cands, states, jev, workers)
        judged = judge(cands, cov, texts, jev, workers=workers) if judge_groups else None
        dups = _dups(cands, cov, facets, band)
        chosen = select(cands, cov, facets, band, judged, min_judge)[: max_classes - len(kept)]
        for c in cands:
            reason = _rejected_by(c, cov, facets, band, judged, min_judge, dups) or (None if c in chosen else "max_classes")
            row = {"id": c.id, "name": c.name, "coverage": _share(cov[c.id]), "kept": c in chosen, "rejected_by": reason}
            if judged is not None:
                row["popular"], row["coherent"] = judged[c.id]
            report.append(row)
        kept += chosen
        kept_cov |= {c.id: cov[c.id] for c in chosen}
        explained = {i for c in chosen for i, p in enumerate(cov[c.id]) if p >= 0.5}
        residual = [i for i in residual if i not in explained]
        if not chosen:
            break
    if child_k and kept:
        kids, orphans, kid_report = _children(kept, kept_cov, texts, vectors, states, jev, encoder, child_k, band,
                                              judge_groups, min_judge, workers)
        promoted, promo_report = _promote(kept, kept_cov, orphans, states, texts, jev, facets, band, judge_groups,
                                          min_judge, max_classes - len(kept), workers) if orphans else ([], [])
        kept, report = kept + promoted + kids, report + promo_report + kid_report
    return kept, report
