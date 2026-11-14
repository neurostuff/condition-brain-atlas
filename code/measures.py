"""Measure groups for patient-versus-healthy contrasts: which of step 1's contrasts enter the corpus, and under
which measure author. One place, used by the corpus builder and the counts.

`measure_group(row)` returns (group, tissue) or (None, reason). Left out by design: seed-based and other
connectivity strength, component loadings (network maps), meta-analytic maps, decoding, ratios of two measures,
surface shape measures (gyrification, curvature, sulcal depth), and BOLD-derived metrics that are none of ALFF,
fALFF, ReHo or degree centrality.
"""
import re

META = re.compile(r"meta-anal|\bsdm\b|signed differential|likelihood estimat|\bale\b|spatial convergence|coordinate-based", re.I)
VOX = {"gray_matter_volume": "grey", "grey_matter_volume": "grey", "gray_matter_density": "grey", "gray matter concentration": "grey",
       "white_matter_volume": "white", "white_matter_density": "white", "white matter concentration": "white"}
OTHER_IN = re.compile(r"voxel-based|\bvbm\b|tensor-based|deformation-based|jacobian|volume|density|concentration|matter probability", re.I)
OTHER_OUT = re.compile(r"gyrif|curvature|sulcal|surface|thickness|fractal|lateral|asymmetr|\bratio\b|t1w/t2w|intensity|covarian|network|connectiv", re.I)
LABEL = {"grey matter": "grey matter volume or density", "white matter": "white matter volume or density", "cortical thickness": "cortical thickness",
         "alff": "ALFF", "falff": "fractional ALFF", "reho": "regional homogeneity", "degree centrality": "degree centrality or connectivity density",
         "fa": "fractional anisotropy", "diffusivity": "mean, radial or axial diffusivity", "glucose metabolism": "glucose metabolism (FDG)",
         "molecular binding": "receptor, transporter or other tracer binding", "perfusion": "cerebral blood flow", "task activation": "task activation",
         "eeg meg": "EEG or MEG source activity"}


def measure_group(r):
    fam, mt, sm = r.get("measure_family") or "structural_morphometry", r.get("measure_type") or "", r.get("specific_metric") or ""
    s = sm.lower()
    if META.search(sm) or META.search(mt) or META.search(fam):
        return None, "meta-analytic map"
    if mt in ("connectivity_strength",) or re.search(r"\bseed\b|seed-based|functional connectivity (with|between|of the)", s):
        return None, "connectivity strength"
    if mt == "component_loading":
        return None, "component loading"
    if re.search(r"/|\bratio\b|coupling", mt, re.I):
        return None, "ratio of two measures"
    if fam == "structural_morphometry":
        if mt in VOX:
            return ("grey matter" if VOX[mt] == "grey" else "white matter"), VOX[mt]
        if mt == "cortical_thickness":
            return "cortical thickness", "grey"
        if mt == "structural_morphometry_other" and OTHER_IN.search(sm) and not OTHER_OUT.search(sm):
            white = bool(re.search(r"white", s)) and not re.search(r"gr[ae]y", s)
            return ("white matter" if white else "grey matter"), ("white" if white else "grey" if re.search(r"gr[ae]y", s) else "mixed")
        return None, "other structural measure"
    if fam == "functional_bold":
        if mt == "bold_response":
            return "task activation", None
        if mt == "bold_derived_metric":
            if re.search(r"regional homogeneity|\breho\b|kendall", s) and not re.search(r"alff|amplitude of low", s):
                return "reho", None
            if re.search(r"falff|fractional amplitude", s):
                return "falff", None
            if re.search(r"alff|amplitude of low", s):
                return "alff", None
            if re.search(r"degree centrality|connectivity density|\bfcd\b|global (brain )?connectivity|connectivity strength|\bfcs\b|eigenvector centrality|nodal degree|number of voxel", s):
                return "degree centrality", None
            return None, "other BOLD-derived metric"
        return None, "other BOLD measure"
    if fam == "diffusion" and mt == "diffusion_metric":
        if re.search(r"anisotropy|\bfa\b", s) and not re.search(r"diffusivity|\bmd\b|\brd\b|\bad\b", s):
            return "fa", "white"
        if re.search(r"diffusivity|\bmd\b|\brd\b|\bad\b|apparent diffusion|\badc\b", s):
            return "diffusivity", "white"
        return None, "other diffusion metric"
    if fam == "molecular_imaging":
        if re.search(r"fdg|glucose|fluorodeoxy", s) or mt == "metabolic_rate":
            return "glucose metabolism", None
        if re.search(r"ecd|hmpao|perfusion|blood flow|\bcbf\b|iodoamphetamine|\bimp\b", s):
            return "perfusion", None
        if mt in ("receptor_binding", "tracer_uptake"):
            return "molecular binding", None
        return None, "other molecular measure"
    if fam == "perfusion" and mt in ("cerebral_blood_flow",):
        return "perfusion", None
    if fam == "electrophysiology" and mt in ("electrophysiological_amplitude", "electrophysiological_power"):
        return "eeg meg", None
    return None, "other measure"
