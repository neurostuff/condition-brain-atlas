"""Corrections to the condition vocabulary (`vocabulary_fixes.json`), applied by build_corpus.py before it builds the hierarchy.

The vocabulary (condition, class, parent) comes from the language-model normalization (NORMALIZE.md, merge_normalization.py).
The fixes table records every correction made by hand, each with the audit item and reason: a class or parent changed, or one
name merged into another. They run in table order. Within one fix: merges, then classes, then parents. A fix that names a
condition the vocabulary doesn't hold fails.

check() enforces what build_corpus.add() relies on: a condition pools into the class of the top of its parent chain, so a
parent must have its child's class. Otherwise the condition's data feed another class's map while its page shows its own
class. That is how Parkinson disease (parent: parkinsonism, movement_other) fed the movement_other map while its page called it
neurodegenerative, until 2026-11-19.
"""
import json
from pathlib import Path

TABLE = Path(__file__).resolve().parent / "vocabulary_fixes.json"


def apply(fin, table, fixes=None):
    """fin: {condition: {condition, class, parent}}; table: {string: [condition...]}. Both changed in place, and returned."""
    fx = fixes or json.load(open(TABLE))
    for c, row in fx.get("conditions_added", {}).items():
        assert c not in fin, f"added condition already in the vocabulary: {c}"
        fin[c] = {"condition": c, **row}
    for f in fx["fixes"]:
        for c, into in f.get("merge", {}).items():
            assert c in fin and into in fin, (f["id"], c, into)
            del fin[c]
            for row in fin.values():
                if row.get("parent") == c:
                    row["parent"] = into
            for s, cs in table.items():
                if c in cs:
                    table[s] = list(dict.fromkeys(into if x == c else x for x in cs))
        for c, cl in f.get("class", {}).items():
            assert c in fin, (f["id"], c)
            fin[c]["class"] = cl
        for c, p in f.get("parent", {}).items():
            assert c in fin and (p is None or p in fin), (f["id"], c, p)
            fin[c]["parent"] = p
    # as merge_normalization.py does: a name that is not a condition of the patients leaves the table
    dropped = {c for c, r in fin.items() if r["class"] == "not_a_condition"}
    for s, cs in table.items():
        if dropped & set(cs):
            table[s] = [x for x in cs if x not in dropped]
    return fin, table


def check(fin, retired=()):
    """fail on a parent of another class, and on a condition left in a retired class"""
    cross = sorted(f"{c} ({r['class']}) -> {r['parent']} ({fin[r['parent']]['class']})" for c, r in fin.items()
                   if r.get("parent") and r["parent"] != c and r["parent"] in fin and fin[r["parent"]]["class"] != r["class"])
    left = sorted(f"{c} ({r['class']})" for c, r in fin.items() if r["class"] in retired)
    if cross or left:
        raise ValueError("vocabulary: " + "; ".join([f"{len(cross)} parents of another class: " + ", ".join(cross)] * bool(cross) +
                                                    [f"{len(left)} conditions in retired classes: " + ", ".join(left)] * bool(left)))
