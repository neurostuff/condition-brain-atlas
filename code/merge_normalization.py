"""Step 3b: one vocabulary and one string -> conditions table from the two normalization rounds.

Round 1 (eight readers, then one reconciler): ROUND1 = [{raw, condition, also, ...}], V2 = [{name, final, class,
parent}] mapping every round-1 name to its final. Round 2 (seven readers, mapping onto the round-1 vocabulary):
ROUND2_DIR/out_*.json = [{raw, condition, parent, class, also}].
A round-2 name that is not in the vocabulary is added with the class and parent its reader gave (the parent only if
it is in the vocabulary; else the name hangs under its class). Names differing only in punctuation or spacing are
merged. "not_a_condition" names are dropped from the table.

usage: python merge_normalization.py ROUND1.json V2.json ROUND2_DIR OUT_DIR
writes OUT_DIR/condition_strings.json ({string: [final condition...]}), OUT_DIR/vocabulary.json ([{condition, class, parent}])
"""
import collections, glob, json, re, sys
from pathlib import Path

r1, v2p, r2d, out = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4]); out.mkdir(parents=True, exist_ok=True)
V2 = json.load(open(v2p)); to_final = {x["name"]: x["final"] for x in V2}
vocab = {}
for x in V2:
    vocab.setdefault(x["final"], {"class": x["class"], "parent": x.get("parent")})
squash = lambda s: re.sub(r"[^a-z0-9]+", "", s.lower())
by_squash = {squash(k): k for k in vocab}
n = collections.Counter(); table = {}


def resolve(name, cls=None, parent=None):
    """a reader's condition name -> its final vocabulary name, adding it if new"""
    name = (name or "").strip().lower()
    if not name:
        return None
    if name in to_final:
        name = to_final[name]
    if name in vocab:
        return name
    if squash(name) in by_squash:
        return by_squash[squash(name)]
    p = resolve(parent) if parent and parent.strip().lower() != name else None
    vocab[name] = {"class": cls or (vocab[p]["class"] if p else "psychiatric_other"), "parent": p}; by_squash[squash(name)] = name; n["new names"] += 1
    return name


for x in json.load(open(r1)):
    fs = [resolve(c) for c in [x["condition"]] + list(x.get("also") or [])]
    table[x["raw"]] = [f for f in dict.fromkeys(fs) if f and vocab[f]["class"] != "not_a_condition"]; n["round 1 strings"] += 1
for f in sorted(glob.glob(r2d + "/out_*.json")):
    for x in json.load(open(f)):
        fs = [resolve(x.get("condition"), x.get("class"), x.get("parent"))] + [resolve(c, x.get("class")) for c in x.get("also") or []]
        table[x["raw"]] = [f_ for f_ in dict.fromkeys(fs) if f_ and vocab[f_]["class"] != "not_a_condition"]; n["round 2 strings"] += 1
for k, v in vocab.items():          # a parent must exist and must not make a cycle
    seen, p = {k}, v["parent"]
    while p:
        if p not in vocab or p in seen:
            v["parent"] = None; n["parents dropped (missing or cyclic)"] += 1; break
        seen.add(p); p = vocab[p]["parent"]
json.dump(table, open(out / "condition_strings.json", "w"), ensure_ascii=False)
json.dump([{"condition": k, **v} for k, v in sorted(vocab.items())], open(out / "vocabulary.json", "w"), ensure_ascii=False, indent=0)
n.update({"strings": len(table), "strings with no condition": sum(not v for v in table.values()), "strings with several": sum(len(v) > 1 for v in table.values()),
          "vocabulary": len(vocab), "with a parent": sum(bool(v["parent"]) for v in vocab.values())})
print(json.dumps(dict(n), indent=1)); print("classes:", collections.Counter(v["class"] for v in vocab.values()).most_common())
