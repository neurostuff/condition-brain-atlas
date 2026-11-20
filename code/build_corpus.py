"""Step 4: the patient-versus-control author-topic corpus, in the task model's corpus format
(task_networks.corpus.load_corpus).

Documents: step 1's patient-versus-healthy contrasts whose measure falls in one of measures.py's groups (grey and
white matter morphometry, cortical thickness, ALFF, fALFF, ReHo, degree centrality, FA, diffusivity, glucose
metabolism, molecular binding, perfusion, task activation, EEG or MEG source activity). Seed-based connectivity,
component loadings, meta-analytic maps and the rest are left out, with counts.

Authors of a document (STRUCTURE=nested, the default):
- for every condition of its patient groups, one author. If the condition has at least MIN_LEAF contrasts in this
  direction and measure, the author is "c:<dir>:<condition> | <measure>", whose parent is "c:<dir>:<condition>"; so a
  disorder's pattern in one measure shrinks toward what the disorder shows across measures. Otherwise the author
  is "c:<dir>:<condition>" itself. <dir> is "dec" (controls > patients) or "inc" (patients > controls). A group's
  several conditions, a string naming several, and several pooled patient groups are all represented. A document
  with no usable condition gets "unspecified condition".
- "m:measure:<group>", what this measure's maps look like whatever the condition;
- "m:scope:restricted" when the analysis was not whole-brain;
- "background", the pattern every contrast shares.
STRUCTURE=flat: the condition author is always "c:<dir>:<condition>" (no measure-specific authors).
STRUCTURE=measure_first (James 2026-11-13: "I explicitly expect different spatial features depending on
measurement"): every condition author is specific to its measure, "c:<dir>:<measure>:<condition>", and the whole
hierarchy is inside the measure: condition -> broader condition -> class -> "any condition", all "in this measure
and direction". Nothing is pooled across measures, so cortical thickness in one condition reaches ReHo in another
only through the shared networks and the "background" author.
Hierarchy (kind-of; an author's prior is its parents' pooled mixture): condition -> broader condition -> ... ->
"c:<dir>:class <class>" -> "c:<dir>:any condition". The two directions have separate trees and share networks.
Each document records its paper ("group", for paper-grouped folds) and weight 1 / (its paper's documents).

STRUCTURE=multimodal (James 2026-11-13: measures have their own spatial features, yet "the other measures could
hint where a separate measure should be": corticospinal white matter with motor cortex in VBM or ReHo): for the
model in pc_model.py, where each network has its own map in every (measure, direction) block. A condition then has
ONE author, "c:<condition>", across measures and directions (its mixture over networks is what ties the measures
together), in one hierarchy; there is no measure author (a block's own base map plays that part). Each document
records its block in "measure" and "direction".

STRUCTURE=multimodal_dir (2026-11-14, for pc_model.py --model window, where a network is one map and direction only
picks the window): as multimodal, but a condition with at least MIN_LEAF contrasts in a direction is represented there
by "c:all:<condition> | less" or "c:all:<condition> | more" (less or more in patients), whose parent is
"c:all:<condition>". So a condition can load on one network for decreases and another for increases, and a direction
with little data falls back on the condition's pooled mixture.

STRUCTURE=multimodal_fam (2026-11-14: "more in patients" does not mean the same thing in every measure: more tracer binding
or diffusivity is damage, more grey matter is tissue; Alzheimer disease's pooled "more" loadings put a hippocampal pattern
into its grey matter increases, which its 16 such contrasts do not show): as multimodal_dir, with one more level. A
condition with at least MIN_LEAF contrasts in a direction AND measure family is represented there by
"c:all:<condition> | less | <family>" (parent "c:all:<condition> | less", whose parent is "c:all:<condition>"). Families
(FAMILY below): structure, diffusion, resting function, metabolism and perfusion, molecular, task, electrophysiology. A
cell with little data falls back on the direction's pooled mixture, then on the condition's.

Since 2026-11-15 (the label audit; primary.py):
- LABELS=primary (default): a document's condition authors are its primary conditions only (primary.assign). Its other
  listed conditions (comorbidities of some patients, symptoms, history) become "x:all:<condition>" authors, "also present":
  no parents, so they take no part in a condition's own map; checks.py uses them only with --variant ...+also.
  LABELS=all is the earlier corpus (every listed condition an equal author), kept for the comparison.
- META (optional fifth argument, from paper_meta.py): contrasts of papers flagged as meta-analyses are dropped.
- Vocabulary fixes (since 2026-11-19, `vocab_fixes.py`, `vocabulary_fixes.json`): every hand correction of a class, parent or
  name, with its reason, applied before the hierarchy is built. This includes the 2026-11-15 class fix (conditions left in
  psychiatric_other that are not psychiatric). The build then fails if any parent's class differs from its child's, since a
  condition pools into the class at the top of its parent chain. A condition missing from the vocabulary gets the class
  "unclassified". OUT_DIR/vocabulary.json is the corrected copy.

usage: [STRUCTURE=nested|flat|measure_first|multimodal|multimodal_dir|multimodal_fam] [MIN_LEAF=5] [LABELS=primary|all] python build_corpus.py CONTRASTS.jsonl CONDITION_STRINGS.json VOCABULARY.json OUT_DIR [META.json]
writes OUT_DIR/corpus_pc_<structure>.json, OUT_DIR/conditions.json, OUT_DIR/corpus_counts_<structure>.json, OUT_DIR/vocabulary.json
"""
import collections, json, os, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from measures import LABEL, measure_group
from primary import assign, key
import vocab_fixes

DIR = {"decrease": "dec", "increase": "inc"}
FAMILY = {"grey matter": "structure", "cortical thickness": "structure", "white matter": "structure", "fa": "diffusion", "diffusivity": "diffusion",
          "alff": "resting function", "falff": "resting function", "reho": "resting function", "degree centrality": "resting function",
          "glucose metabolism": "metabolism and perfusion", "perfusion": "metabolism and perfusion", "molecular binding": "molecular",
          "task activation": "task", "eeg meg": "electrophysiology"}
STRUCTURE = os.environ.get("STRUCTURE", "nested"); MIN_LEAF = int(os.environ.get("MIN_LEAF", "5")); MM = STRUCTURE.startswith("multimodal"); LABELS = os.environ.get("LABELS", "primary"); assert LABELS in ("primary", "all")


def main():
    cp, sp, vp, out = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4]); out.mkdir(parents=True, exist_ok=True)
    table = json.load(open(sp)); fin = {x["condition"]: x for x in json.load(open(vp))}
    fixes = json.load(open(vocab_fixes.TABLE)); vocab_fixes.apply(fin, table, fixes); vocab_fixes.check(fin, fixes["classes_retired"])
    json.dump(sorted(fin.values(), key=lambda x: x["condition"]), open(out / "vocabulary.json", "w"), ensure_ascii=False, indent=0)
    meta = {p for p, v in json.load(open(sys.argv[5])).items() if v["meta"]} if len(sys.argv) > 5 else set(); dropped = set()
    fin.setdefault("unspecified condition", {"condition": "unspecified condition", "class": "not_a_condition", "parent": None})
    n = collections.Counter(); rows = []; unmapped = collections.Counter()
    for line in open(cp):
        r = json.loads(line); m, why = measure_group(r)
        if m is None:
            n[f"left out: {why}"] += 1; continue
        if r["paper"] in meta:
            n["dropped: contrasts of meta-analysis papers"] += 1; dropped.add(r["paper"]); continue
        for g in r["patient_groups"]:
            for s in g["conditions"]:
                if table.get(key(s)) is None: unmapped[key(s)] += 1
        conds, extra, tier = assign(r, table, fin)
        if LABELS == "all": conds, extra = conds + extra, []
        if not conds:
            conds = ["unspecified condition"]; n["documents with no usable condition"] += 1
        n[f"tier {tier}"] += 1; n["documents with additional conditions"] += bool(extra)
        rows.append((r, m, "all" if MM else DIR[r["direction"]], conds, extra, tier))
    n["dropped: meta-analysis papers"] = len(dropped)
    leaf_n = collections.Counter((d, c, m) for _, m, d, cs, _, _ in rows for c in cs)
    dir_n = collections.Counter((r["direction"], c) for r, _, _, cs, _, _ in rows for c in cs)
    fam_n = collections.Counter((r["direction"], FAMILY[m], c) for r, m, _, cs, _, _ in rows for c in cs)
    docs, use, mu, papers = {}, collections.Counter(), collections.Counter(), collections.defaultdict(set); trees = set(); also = collections.Counter()
    for r, m, d, conds, extra, tier in rows:
        ids = []
        for c in conds:
            nested = STRUCTURE == "nested" and leaf_n[d, c, m] >= MIN_LEAF
            by_dir = STRUCTURE in ("multimodal_dir", "multimodal_fam") and dir_n[r["direction"], c] >= MIN_LEAF
            by_fam = STRUCTURE == "multimodal_fam" and fam_n[r["direction"], FAMILY[m], c] >= MIN_LEAF
            ids.append(f"c:{d}:{m}:{c}" if STRUCTURE == "measure_first" else f"c:{d}:{c} | {m}" if nested else
                       f"c:{d}:{c} | {'less' if r['direction'] == 'decrease' else 'more'} | {FAMILY[m]}" if by_fam else
                       f"c:{d}:{c} | {'less' if r['direction'] == 'decrease' else 'more'}" if by_dir else f"c:{d}:{c}")
            trees.add(f"{d}:{m}" if STRUCTURE == "measure_first" else d)
            use[d, c] += 1; mu[d, c, m] += 1; papers[d, c].add(r["paper"])
        restricted = (r["spatial_scope"] or "") != "whole_brain"
        ids += [f"x:all:{c}" for c in extra] + ([] if MM else [f"m:measure:{m}"]) + (["m:scope:restricted"] if restricted else []) + ["background"]
        for c in extra: also[c] += 1
        docs[r["id"]] = {"ids": ids, "group": r["paper"], "effect": "contrast", "foci": r["foci"], "direction": r["direction"], "measure": m, "scope": r["spatial_scope"], "tier": tier}
        n["documents"] += 1; n[f"documents {r['direction']}"] += 1; n[f"measure {m}"] += 1; n["restricted"] += restricted
        n["documents with several conditions"] += len(conds) > 1
    per = collections.Counter(x["group"] for x in docs.values())
    for x in docs.values(): x["weight"] = 1.0 / per[x["group"]]
    parents = {"background": [], "m:scope:restricted": []}; labels = {"background": "shared pattern", "m:scope:restricted": "region-restricted analysis"}
    for c in also: parents[f"x:all:{c}"] = []; labels[f"x:all:{c}"] = f"{c} (also present)"
    for m in {x["measure"] for x in docs.values()} if not MM else ():
        parents[f"m:measure:{m}"] = []; labels[f"m:measure:{m}"] = LABEL[m]

    def add(d, c, seen=()):
        node = f"c:{d}:{c}"
        if node in parents: return node
        info = fin.get(c) or {"parent": None, "class": "unclassified"}; p = info.get("parent")
        if p and p != c and p not in seen and p in fin:
            parents[node] = [add(d, p, seen + (c,))]
        else:
            cl = f"c:{d}:class {info['class'].replace('_', ' ')}"
            if cl not in parents:
                root = f"c:{d}:any condition"; parents.setdefault(root, []); labels[root] = "any condition" if d == "all" else f"any condition ({'less' if d[:3] == 'dec' else 'more'} in patients)"
                parents[cl] = [root]; labels[cl] = f"{info['class'].replace('_', ' ')} (class)"
            parents[node] = [cl]
        labels[node] = c
        return node
    if STRUCTURE == "measure_first":              # one tree per (direction, measure); a tree key is "dec:reho"
        for (d, c, m) in mu: add(f"{d}:{m}", c)
        for k in list(labels):
            if k.startswith("c:") and k[2:5] in ("dec", "inc") and k.count(":") >= 3:
                d, m = k[2:5], k[6:].split(":", 1)[0]
                labels[k] = f"{labels[k]} [{LABEL[m]}, {'less' if d == 'dec' else 'more'} in patients]" if "(class)" in labels[k] or "any condition" in labels[k] else f"{labels[k]} ({LABEL[m]})"
    else:
        for (d, c) in use: add(d, c)
    for x in docs.values():
        for i in x["ids"]:
            while i not in parents:                    # a measure-, direction- or family-specific condition author, and its chain up
                d, rest = i[2:5], i[6:]; c, m = rest.rsplit(" | ", 1)
                parents[i] = [f"c:{d}:{c}"]; labels[i] = f"{c} ({LABEL.get(m, m + ' in patients' if m in ('less', 'more') else m)})"
                i = parents[i][0]
    cnt = collections.Counter(i for x in docs.values() for i in x["ids"]); pap = collections.defaultdict(set)
    for x in docs.values():
        for i in x["ids"]: pap[i].add(x["group"])
    H = {"docs": docs, "parents": parents, "labels": labels, "n_contrasts": {k: cnt.get(k, 0) for k in parents}, "n_papers": {k: len(pap.get(k, ())) for k in parents},
         "structure": {"kind": STRUCTURE, "min_leaf": MIN_LEAF},
         "stats": {"contrasts": len(docs), "papers": len(per), "peaks": sum(len(x["foci"]) for x in docs.values()), "nodes": len(parents)}}
    json.dump(H, open(out / f"corpus_pc_{STRUCTURE}.json", "w"))
    conds = {c for _, c in use} | set(also)
    vocab = sorted(({"condition": c, "class": (fin.get(c) or {}).get("class"), "parent": (fin.get(c) or {}).get("parent"),
                     "also_present": also.get(c, 0), "decrease": use.get(("dec", c), 0) + use.get(("all", c), 0), "increase": use.get(("inc", c), 0), "papers": len(papers.get(("dec", c), set()) | papers.get(("inc", c), set()) | papers.get(("all", c), set())),
                     "by_measure": {m: [mu.get(("dec", c, m), 0), mu.get(("inc", c, m), 0)] for m in LABEL if mu.get(("dec", c, m), 0) + mu.get(("inc", c, m), 0)}}
                    for c in conds), key=lambda x: -(x["decrease"] + x["increase"]))
    json.dump(vocab, open(out / "conditions.json", "w"), indent=0, ensure_ascii=False)
    leaf = [k for k in parents if " | " in k]
    n.update({"papers": len(per), "peaks": H["stats"]["peaks"], "nodes": len(parents), "conditions": len(vocab), "measure-specific condition authors": len(leaf),
              "condition strings not in the normalization": len(unmapped), "their uses": sum(unmapped.values()),
              "conditions with >= 10 contrasts": sum(v["decrease"] + v["increase"] >= 10 for v in vocab), "conditions with >= 30 contrasts": sum(v["decrease"] + v["increase"] >= 30 for v in vocab),
              "authors per document, mean": round(sum(len(x["ids"]) for x in docs.values()) / len(docs), 2)})
    json.dump(dict(n), open(out / f"corpus_counts_{STRUCTURE}.json", "w"), indent=1); print(json.dumps(dict(n), indent=1))
    print("top conditions (dec, inc):", [(v["condition"], v["decrease"], v["increase"]) for v in vocab[:30]])
    cl = collections.Counter()
    for v in vocab: cl[v["class"]] += v["decrease"] + v["increase"]
    print("uses by class:", cl.most_common())


if __name__ == "__main__":
    main()
