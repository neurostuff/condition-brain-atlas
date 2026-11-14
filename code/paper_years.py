"""Publication year of every paper in contrasts.jsonl, from the ns-pond catalog's metadata summaries (read-only).
usage: paper_years.py CONTRASTS.jsonl OUT.json     -> {paper id: year or null}"""
import json, sqlite3, sys
CAT = "/data/alejandro/projects/ns-pond/catalog"
art = {}
for line in open(sys.argv[1]):
    r = json.loads(line); art[r["paper"]] = r["article"]
c = sqlite3.connect(f"file:{CAT}/catalog.sqlite?mode=ro", uri=True); out = {}
for p, a in art.items():
    b = c.execute("select summary from artifacts where article_id=? and stage='metadata' and status='ok' order by updated_at desc limit 1", (a,)).fetchone()
    try: out[p] = int(json.loads(b[0])["publication_year"])
    except Exception: out[p] = None  # noqa: BLE001
json.dump(out, open(sys.argv[2], "w"))
import collections
n = collections.Counter(out.values()); print(len(out), "papers;", n[None], "without a year"); print(sorted((k, v) for k, v in n.items() if k))
