"""Step 1 of the VBM author-topic corpus: every structural-morphometry contrast between patients and healthy
controls in the ns-pond records, with every medical condition of its patient groups.

Source (read-only; nothing is written in the ns-pond tree): the catalog's latest `record` artifact per paper
(/data/alejandro/projects/ns-pond/catalog, sqlite mode=ro) and the paper's parsed coordinates
(/data/alejandro/projects/ns-pond/data/<base study>/parse/coordinate_parse.json).

An analysis is kept when:
- its measure's family is one of FAMILIES (default structural_morphometry; the family and type are kept: grey or
  white matter volume, density, cortical thickness, ALFF, ReHo and so on; which types enter the corpus is decided
  in step 4, not here);
- its effect is a contrast with a positive and a negative cell;
- all its groups are human (the records also hold animal studies);
- its groups include at least one healthy group (is_healthy) and at least one that is not, and the two cells can be
  told apart as patients and controls (a cell is matched to the groups by name; else by control words);
- its coordinate set has at least one point.
Direction: "decrease" = controls > patients (less in patients); "increase" = patients > controls.
Conditions: the `medical_condition` list of EVERY patient group on the patient side, kept per group, so a group with
several conditions, or several patient groups pooled, are all represented. Whole-brain or not is recorded
(`spatial_scope`), not filtered.

usage: python build_contrasts.py OUT_DIR [WORKERS]
writes OUT_DIR/contrasts.jsonl, OUT_DIR/build_counts.json
"""
import collections, gzip, json, os, re, sqlite3, sys
from multiprocessing import Pool
from pathlib import Path

CAT = Path("/data/alejandro/projects/ns-pond/catalog"); DATA = Path("/data/alejandro/projects/ns-pond/data")
#: measure families kept (env FAMILIES, comma-separated; "all" = every family). Default: structural morphometry.
FAMILIES = None if os.environ.get("FAMILIES") == "all" else set((os.environ.get("FAMILIES") or "structural_morphometry").split(","))
CONTROL = re.compile(r"\b(controls?|healthy|hcs?|ncs?|cn|td|tdc|typically|normal|comparison|unaffected|volunteers?|hv)\b", re.I)


def v(x):
    """a record field's value: fields are {"value": ..., "extraction_status": ...} wrappers"""
    if isinstance(x, dict) and ("value" in x or "extraction_status" in x):
        return x.get("value")
    return x


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def side_of(cell, groups):
    """which involved groups a contrast cell names: (set of group ids, how)"""
    keys = [norm(v(cell.get("level"))), norm(v(cell.get("label")))]
    keys = [k for k in keys if k]
    hit = {g["local_id"] for g in groups for k in keys if k and k in (norm(v(g.get("name"))), norm(g.get("local_id")).replace("grp ", ""))}
    if hit:
        return hit, "name"
    hit = {g["local_id"] for g in groups for k in keys if len(k) >= 3 and norm(v(g.get("name"))) and (k in norm(v(g.get("name"))) or norm(v(g.get("name"))) in k)}
    if hit:
        return hit, "partial"
    return set(), "none"


def one(args):
    art, base = args
    c = sqlite3.connect(f"file:{CAT}/catalog.sqlite?mode=ro", uri=True)
    b = c.execute("select blob from artifacts where article_id=? and stage='record' and status='ok' order by updated_at desc limit 1", (art,)).fetchone()
    c.close()
    n = collections.Counter(); out = []
    if not b or not b[0]:
        return out, n
    try:
        r = json.load(gzip.open(CAT / "blobs" / b[0][:2] / f"{b[0]}.json.gz"))
    except Exception:  # noqa: BLE001
        n["record unreadable"] += 1; return out, n
    n["records"] += 1
    meas = {m["local_id"]: m for m in r.get("measures") or [] if isinstance(m, dict)}
    grp = {g["local_id"]: g for g in r.get("groups") or [] if isinstance(g, dict) and g.get("local_id")}
    sets = {cs.get("analysis"): cs.get("local_id") for cs in r.get("coordinate_sets") or [] if isinstance(cs, dict)}
    parse = None
    for a in r.get("analyses") or []:
        if not isinstance(a, dict):
            continue
        m = meas.get(a.get("measure")) or {}
        fam = v(m.get("family"))
        if FAMILIES is not None and fam not in FAMILIES:
            continue
        n["analyses in the measure families"] += 1
        eff = a.get("effect") or {}; cells = [x for x in eff.get("cells") or [] if isinstance(x, dict)]
        pos = [x for x in cells if v(x.get("direction")) == "positive"]; neg = [x for x in cells if v(x.get("direction")) == "negative"]
        if v(eff.get("kind")) != "contrast" or len(pos) != 1 or len(neg) != 1:
            n["not a two-sided contrast"] += 1; continue
        gs = [grp[x["group"]] for x in a.get("groups") or [] if isinstance(x, dict) and x.get("group") in grp]
        gn = {x["group"]: x.get("n") for x in a.get("groups") or [] if isinstance(x, dict) and x.get("group")}
        if any(str(v(g.get("species")) or "human").strip().lower() not in ("human", "humans", "homo sapiens") for g in gs):
            n["not human"] += 1; continue
        healthy = {g["local_id"] for g in gs if v(g.get("is_healthy")) is True}; patient = {g["local_id"] for g in gs if v(g.get("is_healthy")) is False}
        if not healthy or not patient:
            n["no healthy group" if not healthy else "no patient group"] += 1; continue
        sp, hp = side_of(pos[0], gs); sn, hn = side_of(neg[0], gs)
        def kind(s, cell):
            if s and s <= healthy: return "healthy"
            if s and s <= patient: return "patient"
            if not s and CONTROL.search(" ".join(str(v(cell.get(k)) or "") for k in ("level", "label"))): return "healthy"
            return None
        kp, kn = kind(sp, pos[0]), kind(sn, neg[0])
        if kp is None and kn == "healthy": kp, sp = "patient", patient
        if kn is None and kp == "healthy": kn, sn = "patient", patient
        if kp is None and kn == "patient" and len(healthy) >= 1: kp, sp = "healthy", healthy
        if kn is None and kp == "patient" and len(healthy) >= 1: kn, sn = "healthy", healthy
        if {kp, kn} != {"healthy", "patient"}:
            n["sides not told apart" if None in (kp, kn) else "not patients against healthy"] += 1; continue
        direction = "increase" if kp == "patient" else "decrease"
        pside = (sp if kp == "patient" else sn) or patient; hside = (sp if kp == "healthy" else sn) or healthy
        key = sets.get(a.get("local_id")) or v(a.get("source_table_analysis"))
        if parse is None:
            try:
                parse = {x["key"]: x for x in json.load(open(DATA / base / "parse" / "coordinate_parse.json")).get("analyses") or []}
            except Exception:  # noqa: BLE001
                parse = {}
        pa = parse.get(key)
        pts = [p["coordinates"] for p in (pa or {}).get("points") or [] if p.get("coordinates") and len(p["coordinates"]) == 3]
        if not pts:
            n["no coordinates"] += 1; continue
        space = (v(a.get("coordinate_space")) or (pa or {}).get("coordinate_space") or "MNI")
        space = "TAL" if str(space).upper().startswith("TAL") else "MNI"
        pg = []
        for gid in sorted(pside):
            g = grp[gid]
            pg.append({"group": gid, "name": v(g.get("name")), "conditions": [str(x) for x in (v(g.get("medical_condition")) or []) if x],
                       "n": gn.get(gid), "age_mean": v(g.get("age_mean")), "medication_status": v(g.get("medication_status")),
                       "description": (v(g.get("description")) or "")[:300]})
        out.append({"id": f"{base}|{a.get('local_id')}", "paper": base, "article": art, "analysis": a.get("local_id"), "name": v(a.get("name")),
                    "direction": direction, "matched": [hp, hn], "patient_groups": pg,
                    "healthy_groups": [{"group": gid, "name": v(grp[gid].get("name")), "n": gn.get(gid), "age_mean": v(grp[gid].get("age_mean"))} for gid in sorted(hside)],
                    "measure_family": fam, "measure_type": v(m.get("type")), "specific_metric": v(m.get("specific_metric")), "spatial_scope": v(a.get("spatial_scope")),
                    "outcome": v(a.get("outcome")), "space": space, "foci": [[float(x), float(y), float(z), space] for x, y, z in pts]})
        n["kept"] += 1; n[f"kept {direction}"] += 1
    return out, n


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); workers = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    c = sqlite3.connect(f"file:{CAT}/catalog.sqlite?mode=ro", uri=True)
    arts = c.execute("select a.article_id, min(l.value) from artifacts a join aliases l on l.article_id = a.article_id and l.kind = 'neurostore' "
                     "where a.stage = 'record' and a.status = 'ok' group by a.article_id").fetchall()
    c.close()
    tot = collections.Counter(); k = 0
    with Pool(workers) as p, open(out / "contrasts.jsonl", "w") as fo:
        for rows, n in p.imap_unordered(one, arts, chunksize=200):
            tot.update(n)
            for r in rows:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n"); k += 1
    tot["papers with a record"] = len(arts)
    json.dump(dict(tot), open(out / "build_counts.json", "w"), indent=1)
    print(json.dumps(dict(tot), indent=1)); print("contrasts written", k)


if __name__ == "__main__":
    main()
