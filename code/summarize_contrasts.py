"""Counts over step 1's contrasts.jsonl: measure types, spatial scope, peaks, raw condition strings. usage: summarize_contrasts.py DATA_DIR"""
import collections, json, sys
R = [json.loads(l) for l in open(sys.argv[1] + "/contrasts.jsonl")]
C = collections.Counter
print(len(R), "contrasts,", len({r["paper"] for r in R}), "papers,", sum(len(r["foci"]) for r in R), "peaks")
print("direction:", C(r["direction"] for r in R).most_common())
print("measure types:", C(r["measure_type"] for r in R).most_common(20))
print("spatial scope:", C(r["spatial_scope"] for r in R).most_common(10))
print("outcome:", C(r["outcome"] for r in R).most_common(6))
print("how sides were matched:", C(tuple(r["matched"]) for r in R).most_common(6))
ng = C(len(r["patient_groups"]) for r in R); print("patient groups per contrast:", sorted(ng.items()))
nc = C(len({c for g in r["patient_groups"] for c in g["conditions"]}) for r in R); print("distinct conditions per contrast:", sorted(nc.items()))
raw = C(c.strip() for r in R for c in {c for g in r["patient_groups"] for c in g["conditions"]})
print("distinct raw condition strings:", len(raw), "; used once:", sum(v == 1 for v in raw.values()))
print("top 60:", raw.most_common(60))
json.dump(raw.most_common(), open(sys.argv[1] + "/conditions_raw.json", "w"), ensure_ascii=False, indent=0)
low = C(c.strip().lower() for r in R for c in {c for g in r["patient_groups"] for c in g["conditions"]}); print("distinct after lowercasing:", len(low))
