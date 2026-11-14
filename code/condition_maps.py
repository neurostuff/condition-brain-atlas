"""What the model says one condition looks like in each measure, next to what that condition's own contrasts show.
usage: condition_maps.py CORPUS FIT_PREFIX "CONDITION" OUT_PREFIX     (run from the work dir; PYTHONPATH needs task_networks)

Predicted map in block b (measure x direction): sum_k theta[condition, k] beta[k, b, .] (the condition's author alone:
no background, no comorbid authors). Observed: the token density of the condition's single-condition contrasts in
that block. Per measure the direction with more of the condition's contrasts is shown. For each block: contrasts,
correlation over voxels of predicted with observed, and of the block's own map s[b] with observed (what any
condition would get). In-sample: the fit saw these contrasts.
"""
import collections, json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from task_networks.corpus import load_corpus

corpus, fit, cname, out = sys.argv[1:5]
H = json.load(open(corpus)); C = load_corpus(corpus); M = json.load(open(fit + ".json")); nodes, names = M["nodes"], M["blocks"]; B = len(names)
be, s, n, xyz = (np.load(f"{fit}_{x}.npy").astype(float) for x in ("beta", "s", "n", "xyz")); V = len(xyz); K = n.shape[1]; be = be.reshape(K, B, V); s = s.reshape(B, V)
ix = {x: i for i, x in enumerate(nodes)}; own = [i for x, i in ix.items() if x == f"c:all:{cname}" or x.startswith(f"c:all:{cname} | ")]
t = n[own].sum(0); t = t / t.sum()                      # the condition's pooled mixture (its own author and any direction children)
bix = {b: i for i, b in enumerate(names)}
obs = collections.defaultdict(lambda: np.zeros(V)); nd = collections.Counter()
for d, a in enumerate(C["ids"]):
    cs = {nodes[i].split(" | ")[0] for i in C["auth"][d] if nodes[i].startswith("c:all:")}
    if cs != {f"c:all:{cname}"} or len(C["toks"][d]) == 0: continue
    x = H["docs"][a]; b = bix[f"{x['measure']} | {x['direction']}"]; np.add.at(obs[b], np.asarray(C["toks"][d]) % V, 1.0); nd[b] += 1
meas = sorted({b.split(" | ")[0] for b in names}); rows = []
for m in meas:
    b = max((bix[f"{m} | decrease"], bix[f"{m} | increase"]), key=lambda i: nd[i]); pred = t @ be[:, b]
    r = {"block": names[b], "contrasts": nd[b]}
    if nd[b] >= 3:
        r["r_predicted"] = float(np.corrcoef(pred, obs[b])[0, 1]); r["r_base"] = float(np.corrcoef(s[b], obs[b])[0, 1])
    rows.append((nd[b], b, pred, r))
rows.sort(key=lambda z: -z[0])
print(cname, "mixture:", [(int(k), round(float(t[k]), 2)) for k in np.argsort(-t)[:6]])
for _, b, _, r in rows: print(r)
json.dump({"condition": cname, "mixture": t.tolist(), "blocks": [r for *_, r in rows]}, open(out + ".json", "w"), indent=1)
lo = xyz.min(0); ijk = np.rint((xyz - lo) / 4).astype(int); shp = ijk.max(0) + 1; mask = np.zeros(shp, bool); mask[tuple(ijk.T)] = True
def draw(ax, v):
    vol = np.zeros(shp); vol[tuple(ijk.T)] = v / max(v.max(), 1e-30); vol[vol < 0.3] = 0
    sag, axi, cor = vol.max(0).T, vol.max(2).T, vol.max(1).T; ms, ma, mc = mask.max(0).T, mask.max(2).T, mask.max(1).T
    Hh = max(sag.shape[0], axi.shape[0], cor.shape[0]); pad = lambda a: np.pad(a.astype(float), ((0, Hh - a.shape[0]), (0, 2)))
    img = np.c_[pad(sag), pad(cor), pad(axi)]; m = np.c_[pad(ms), pad(mc), pad(ma)]
    ax.imshow(np.where(m > 0, 0.93, 1.0), origin="lower", cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.imshow(np.ma.masked_equal(img, 0), origin="lower", cmap="magma_r", vmin=0.2, vmax=1, interpolation="nearest")
    ax.contour(m, [0.5], colors="0.45", linewidths=0.5); ax.axis("off")
fig, axes = plt.subplots(len(rows), 3, figsize=(15, len(rows) * 1.75))
for j, (k, b, pred, r) in enumerate(rows):
    lab = names[b].replace(" | decrease", ", less in patients").replace(" | increase", ", more in patients")
    draw(axes[j, 0], pred); axes[j, 0].set_title(f"{lab}: model's prediction for {cname}", fontsize=8, loc="left")
    if k >= 3:
        draw(axes[j, 1], obs[b]); axes[j, 1].set_title(f"observed, {k} contrasts (r with prediction {r['r_predicted']:.2f}; with the measure's own map {r['r_base']:.2f})", fontsize=8, loc="left")
    else:
        axes[j, 1].axis("off"); axes[j, 1].text(0.5, 0.5, f"{k} contrasts: nothing to compare with", ha="center", va="center", fontsize=8, color="0.5", transform=axes[j, 1].transAxes)
    draw(axes[j, 2], s[b]); axes[j, 2].set_title("the measure's own map (any condition)", fontsize=8, loc="left")
fig.savefig(out + ".png", dpi=100, bbox_inches="tight"); print("figure", out + ".png")
