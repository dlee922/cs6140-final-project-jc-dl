# Metastatic Site Classification of Lung Adenocarcinoma  
**Comparing Clinical, Genomic, and Combined Feature Sets Across Machine Learning Methods**

---

## Overview
This project predicts:
- **Ever-metastatic status** (binary classification)
- **Seven metastatic sites** (multilabel classification)

### Key Findings
- Clinical features consistently outperform genomic features across tasks  
- Combining feature sets improves performance for binary classification, but not multilabel classification  
- Linear models outperform nonlinear models, which tend to overfit due to limited data  
- No substantial sex disparity observed; insufficient representation of non-White patients limits racial fairness evaluation  
[Read the full paper](paper/luad_project.pdf)
---

## Training

```bash
python -m scripts.train -f <FEATURE_SET> -m <MODEL> -t <TASK>
```

### Arguments

| Flag | Description | Options |
|------|------------|--------|
| `-f`, `--feature_set` | Feature set used for training | `clinical`, `genomic`, `combined`, `combined_interact` |
| `-m`, `--model` | Model type | `logistic`, `logistic_ridge`, `logistic_lasso`, `random_forest`, `svm` |
| `-t`, `--task` | Prediction task | `binary`, `multilabel` |

> If `-m` is omitted, all models (including MLP) will be trained.

---

## Evaluation

```bash
python -m scripts.evaluate -f <FEATURE_SET> -m <MODEL>
```

---

## Visualization

```bash
python -m utils.visualize -f <FEATURE_SET>
```

---

## Configuration

Hyperparameters are defined in:

```bash
configs.yaml
```

Modify this file to adjust model settings and training behavior.

---

## Example Workflow

```bash
python -m scripts.train -f clinical -t binary
python -m scripts.evaluate -f clinical
python -m utils.visualize -f clinical
```

---

## Notes
- Run build scripts to get the necessary data
- For full CLI options, run:

```bash
python -m scripts.train --help
```

LUAD Met Organotropism 2023 dataset:  
- https://datahub.assets.cbioportal.org/luad_mskcc_2023_met_organotropism.tar.gz
