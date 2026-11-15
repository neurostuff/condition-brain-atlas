# Condition Brain Atlas

A model of where published neuroimaging studies report differences between patients and healthy controls, across
medical conditions and imaging measures, and a website for finding hypotheses the literature has not tested.

**Website:** https://neurostuff.github.io/condition-brain-atlas/

Experimental. Every map is a prediction from published coordinates, not a measurement.

## What it is

- **Data:** 16,693 patient-versus-control contrasts from 6,418 papers (85,641 reported peaks), each tagged with its
  medical conditions (1,567 in all, in 23 classes), one of 14 imaging measures (grey and white matter volume, cortical
  thickness, fractional anisotropy, diffusivity, ALFF, fALFF, ReHo, degree centrality, glucose metabolism, perfusion,
  tracer binding, task activation, EEG / MEG sources) and a direction (less or more in patients). Conditions, measures
  and directions were read from the papers by language models and are not individually checked.
  A contrast counts toward the diagnosis its patients were recruited for (`code/primary.py`); the other conditions listed
  for the patients (comorbidities, symptoms, history; 3,479 contrasts have some) are kept as separate "also present"
  labels that do not shape a condition's map. For 780 contrasts the diagnosis could not be told apart from the rest of
  the list, and they count toward every listed condition. Papers that are meta-analyses (194) are left out.
- **Model:** a regularised regression with a point-process likelihood. The probability that a contrast's peak falls at
  voxel `v` is `softmax_v(log window[measure, v] + Phi[v] . sum of label coefficients)`. The window is where that measure
  usually reports differences. `Phi` is 1,075 Gaussian bumps (8 mm) on a 12 mm lattice. The labels of a contrast are its
  condition, the condition's parents and class, and the condition within a family of related measures, each separate for
  "less" and "more"; plus the measure and direction. A ridge penalty shrinks each label's map toward zero.
- **Website:** for 249 conditions with 15 or more contrasts: the published evidence for each measure and direction, the
  predicted map for every cell including the untested ones, what each prediction is built from, and three lists of
  candidates (untested predictions, measures that depart from a condition's pattern, similar conditions from different
  classes).

## How well it predicts

Gain in log-likelihood per held-out peak over the measure's window alone (95% interval from a bootstrap over papers).

| Held out | Gain |
|---|---|
| A fifth of the papers (5 folds) | +0.048 [+0.040, +0.056] |
| One whole measure for a condition | +0.070 (69% of cells won) |
| Papers from 2020 on, fitted to earlier papers | +0.021 |
| ... of those, a measure new for a known condition | +0.007 |

The gains are real and small, and most of the gain comes from a condition's broad class. `MODEL_PLAN.md` records which additions
to the model were tested on held-out data and which were kept (a lighter penalty on family-level labels, direction by
meaning for diffusivity, labels split by direction) or dropped (a learned coupling between measures).

## How uncertain it is

The model is refitted 50 times on papers resampled with replacement. The website shows, for each region of a cell, the
80% interval of its share of predicted peaks over those refits. A single new study varies more, because peaks from one
paper cluster (within-paper correlation about 0.1), and the interval for a study's peak count in a region includes that.
That interval was tested on held-out contrasts, for regions expected to hold at least one peak:

| Held out | 80% interval, peaks treated as independent | 50% interval | 80% interval | 95% interval |
|---|---|---|---|---|
| A fifth of the papers (5 folds) | 63% | 48% | 81% | 96% |
| One whole measure for a condition | 64% | 50% | 81% | 95% |

The intervals of the expected pattern itself cannot be tested this way, and they leave out the bias from the penalty.

## Layout

| Path | What |
|---|---|
| `code/checks.py` | the regression (`additive`): fit, held-out tests by paper, by held-out measure and by year; options for every tested addition |
| `code/compare_scores.py` | paired comparison of two runs with a bootstrap over papers |
| `code/uncertainty.py`, `unc_report.py` | refits on resampled papers; coverage of the intervals on held-out contrasts |
| `code/export_site_hyp.py` | fits the kept model on all contrasts and writes the website's data, with intervals when given the refits |
| `code/build_contrasts.py`, `measures.py`, `merge_normalization.py`, `build_corpus.py` | building the corpus from extracted papers |
| `code/primary.py`, `paper_meta.py` | which listed condition is the diagnosis and which are additional; which papers are meta-analyses |
| `code/pc_model.py` and the report scripts | an earlier author-topic model of the same data, kept for comparison (it predicted about half as well) |
| `code/*.sh` | job wrappers as run on our GPU server; paths are specific to it |
| `docs/` | the website (`index.html`, `data.json`, `maps.txt`), served by GitHub Pages |
| `MODEL_PLAN.md` | the plan for the model, with the test each addition had to pass and its outcome |

## Running it

The corpus is not in this repository: it is built from full-text papers that cannot be redistributed. With a corpus in
the format `build_corpus.py` writes, and the `task_networks` package (corpus loader) on `PYTHONPATH`:

```bash
python code/checks.py additive corpus.json out.json --window-from foci --lam 20 --pen fam=5 --variant full+dirsplit+meaning --tests paper,cell,future --save-scores
```

```bash
python code/export_site_hyp.py corpus.json vocabulary.json docs JHU-ICBM-labels-2mm.nii.gz JHU-labels.xml out.json out_cell.npy
```

For the intervals, `code/uncertainty.py` (`fit` and `score`; see its docstring) and `code/site_job.sh` for the order of the steps.

Needs Python 3.12, PyTorch with a GPU (about 3 GB of memory), NumPy, nibabel and nilearn's Harvard-Oxford atlas files.

## What the website publishes

Condition names (normalised), counts of contrasts and papers, and model coefficients. No text from papers, no
coordinates of individual studies.

## Limits

The intervals cover resampling of papers and clustering within a paper, not errors in reading the papers or the choice
of model. Peaks say where a difference was reported, not how large it was. The model learns from
what was published, including where researchers chose to look.
