"""Which papers of contrasts.jsonl are meta-analyses (their pooled coordinate tables are not a study's own result).
A paper is flagged when the ns-pond parse says so (parsed_paper.json, is_meta_analysis) or its title names a meta-analysis
or systematic review (catalog metadata). Both sources are read-only.
usage: paper_meta.py CONTRASTS.jsonl OUT.json     -> {paper id: {"meta": bool, "parse": bool or null, "title": bool or null}}"""
import collections, json, re, sqlite3, sys
ROOT = "/data/alejandro/projects/ns-pond"; TITLE = re.compile(r"meta-?\s?analy|systematic review", re.I)
art = {}
for line in open(sys.argv[1]):
    r = json.loads(line); art[r["paper"]] = r["article"]
c = sqlite3.connect(f"file:{ROOT}/catalog/catalog.sqlite?mode=ro", uri=True); out = {}
for p, a in art.items():
    try: parse = bool(json.load(open(f"{ROOT}/data/{p}/parse/parsed_paper.json")).get("is_meta_analysis"))
    except Exception: parse = None  # noqa: BLE001
    b = c.execute("select summary from artifacts where article_id=? and stage='metadata' and status='ok' order by updated_at desc limit 1", (a,)).fetchone()
    try: title = bool(TITLE.search(json.loads(b[0])["title"]))
    except Exception: title = None  # noqa: BLE001
    out[p] = {"meta": bool(parse or title), "parse": parse, "title": title}
json.dump(out, open(sys.argv[2], "w"))
n = collections.Counter((v["parse"], v["title"]) for v in out.values()); print(len(out), "papers;", sum(v["meta"] for v in out.values()), "flagged; (parse, title):", sorted(n.items(), key=str))
