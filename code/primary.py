"""Which of a patient group's listed conditions is the diagnosis it was recruited for, and which are additional
(2026-11-15; from an audit of 73 grey matter contrasts against the papers' full text, not in this repository).

A group's condition list mixes its diagnosis with comorbidities of some patients, symptoms and history. Rule:
- primary string of a group: the first of its list that does not name a part of the sample (PARTIAL);
- diagnosis of a condition: its highest ancestor that is not a generic grouping (GENERIC);
- tier of a contrast: "strict" every listed condition has the diagnosis of a primary one; "supported" others are listed
  and each group's name or description names its primary one (the string, its initials or an alias); "unsupported"
  otherwise; "none" no group has a primary string.
Audit (73 grey matter contrasts read against the full text): diagnosis right in 36 of 36 strict, 24 of 29 supported, 4 of 8
unsupported.

assign() -> (primary conditions, additional conditions, tier):
- strict, supported: primary = every condition of the groups' primary strings, plus listed conditions with the same
  diagnosis, less any that is an ancestor of another; additional = the other listed conditions that are not ancestors of
  a primary one.
- unsupported, none: which condition is the diagnosis is not known, so every listed condition is primary and none is
  additional (the contrast is shared equally, as the corpus did for every contrast before 2026-11-15).
"""
import re

GENERIC = {"psychotic disorder", "depressive disorder", "mood disorder", "anxiety disorder", "substance use disorder", "behavioral addiction",
           "specific learning disorder", "eating disorder", "personality disorder", "disruptive behavior disorder"}
PEOPLE = r"(patients?|subjects?|participants?|cases|individuals|children|adolescents|adults|men|women|members|youths?|probands?|persons?|people)"
PARTIAL = re.compile(rf"\b(in|among|for|of)\s+(\S+\s+){{0,4}}?{PEOPLE}\b|\d\s*%|\bn\s*=\s*\d|\b\d+\s+{PEOPLE}\b|^(comorbid|co-morbid|past|previous|prior|former|lifetime|history of|some|possible)\b", re.I)
ALIAS = {"schizophrenia": ["sz", "scz", "sch", "schiz", "sczs", "szs", "sp", "fes", "schizophrenic", "schizophrenics"], "major depressive disorder": ["mdd", "depression", "depressed", "depressive", "md", "mdds", "trd", "lld", "rmdd"],
         "bipolar disorder": ["bd", "bp", "bipolar", "bdi", "bdii", "bpd-i", "manic", "mania"], "obsessive-compulsive disorder": ["ocd"], "posttraumatic stress disorder": ["ptsd"],
         "attention deficit hyperactivity disorder": ["adhd"], "autism spectrum disorder": ["asd", "autism", "autistic", "asperger", "as", "hfa"], "dyslexia": ["dyslexic", "dyslexics", "dys", "dd"],
         "alcohol use disorder": ["alcohol", "alcoholic", "alcoholics", "aud", "ad", "alc"], "internet gaming disorder": ["igd", "iga", "gaming"], "cocaine use disorder": ["cocaine", "cud"],
         "tobacco use disorder": ["smokers", "smoker", "smoking", "nicotine"], "anorexia nervosa": ["an", "anorexia"], "panic disorder": ["pd", "panic"], "borderline personality disorder": ["bpd", "borderline"],
         "generalized anxiety disorder": ["gad"], "social anxiety disorder": ["sad", "social phobia"], "opioid use disorder": ["heroin", "opioid", "opiate"], "methamphetamine use disorder": ["methamphetamine", "ma", "meth"]}


def key(s):
    return re.sub(r"\s+", " ", s.replace("’", "'").replace("–", "-").replace("—", "-").strip().lower())


def words(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def chain(c, V):
    ch = [c]
    while V.get(ch[-1], {}).get("parent") and V[ch[-1]]["parent"] not in ch and V[ch[-1]]["parent"] in V: ch.append(V[ch[-1]]["parent"])
    return ch


def diag(c, V):
    ch = chain(c, V); ok = [x for x in ch if x not in GENERIC]; return ok[-1] if ok else ch[0]


def supported(raw, cond, d, g):
    """does the group's name or description name this condition?"""
    text = " ".join(str(g.get(k) or "") for k in ("name", "description")); tw = words(text); tl = " ".join(tw)
    cands = [" ".join(words(raw)), cond, d]; ini = ["".join(w[0] for w in words(x) if w not in ("of", "the", "and")) for x in cands]
    return any(c and c in tl for c in cands) or any(len(i) >= 2 and i in tw for i in ini) or any(a in tw or (" " in a and a in tl) for x in (cond, d) for a in ALIAS.get(x, []))


def assign(r, table, V):
    prim, allc, sup = [], [], True
    for g in r["patient_groups"]:
        strs = [s for s in g["conditions"] if table.get(key(s))]
        for s in strs: allc += [f for f in table[key(s)] if f not in allc]
        first = next((s for s in strs if not PARTIAL.search(s)), None)
        if first is None: continue
        cs = table[key(first)]; prim += [c for c in cs if c not in prim]; sup &= supported(first, cs[0], diag(cs[0], V), g)
    if not prim: return allc, [], "none"
    ds = {diag(c, V) for c in prim}; strict = all(diag(c, V) in ds for c in allc)
    if not strict and not sup: return allc, [], "unsupported"
    prim += [c for c in allc if c not in prim and diag(c, V) in ds]
    anc = {a for c in prim for a in chain(c, V)[1:]}; prim = [c for c in prim if c not in anc]
    return prim, [c for c in allc if c not in prim and c not in anc], "strict" if strict else "supported"
