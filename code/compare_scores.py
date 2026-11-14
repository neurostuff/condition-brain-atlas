"""Paired comparison of two runs of checks.py additive --save-scores on the same held-out contrasts.
Prints the difference B - A in log-likelihood per held-out peak with a 95% interval from a bootstrap over papers (a paper's
contrasts stay together), overall and for decrease / increase blocks (even block index = decrease).
usage: compare_scores.py A.json B.json [paper cell future]   (reads A_paper.npy etc. next to the JSON files)
"""
import sys, numpy as np

a, b = sys.argv[1:3]; tests = sys.argv[3:] or ["paper", "cell", "future"]; rng = np.random.default_rng(0)
for t in tests:
    try: A = np.load(a.replace(".json", f"_{t}.npy")); B = np.load(b.replace(".json", f"_{t}.npy"))
    except FileNotFoundError: continue
    assert A.shape == B.shape and np.array_equal(A[:, 0], B[:, 0]), "the two runs did not score the same contrasts"
    diff = B[:, 1] - A[:, 1]; n = A[:, 3]; pap = A[:, 5].astype(int); out = []
    for name, sel in [("all", np.ones(len(A), bool)), ("decrease", A[:, 4] % 2 == 0), ("increase", A[:, 4] % 2 == 1)]:
        P = int(pap.max()) + 1; num = np.bincount(pap[sel], diff[sel], P); den = np.bincount(pap[sel], n[sel], P); keep = den > 0; num, den = num[keep], den[keep]
        bs = [num[i].sum() / den[i].sum() for i in (rng.integers(0, len(num), len(num)) for _ in range(2000))]
        out.append(f"{name} {num.sum() / den.sum():+.4f} [{np.quantile(bs, 0.025):+.4f}, {np.quantile(bs, 0.975):+.4f}]")
    print(f"{t}: " + "; ".join(out))
