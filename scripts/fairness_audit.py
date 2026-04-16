import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import ClassifierChain

# Load data
X_full = pd.read_csv('data/processed/X_combined.csv')
y = pd.read_csv('data/processed/y.csv')
y = y.drop(columns=['sampleId', 'patientId']).astype(int)

# keep demographic columns before dropping from X
demo_cols = ['RACE_Black', 'RACE_White', 'RACE_Other', 'SEX_Male']
demographics = X_full[demo_cols].copy()

X = X_full.drop(columns=['patientId', 'sampleId', 'GENE_PANEL'])
X = X.values
y = y.values

LABEL_NAMES = ['Adrenal', 'Bone', 'CNS', 'Liver', 'LN', 'Lung', 'Pleura']
CHAIN_ORDER = [2, 1, 3, 0, 5, 6, 4]
MIN_SAMPLES = 10

# Train/test split
X_train, X_test, y_train, y_test, demo_train, demo_test = train_test_split(
    X, y, demographics.values, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit best model — Ridge with GridSearchCV
lr_param_grid = {'estimator__C': np.logspace(-2, 2, 5)}
model = ClassifierChain(
    LogisticRegression(
        penalty='l2',
        class_weight='balanced',
        max_iter=2000,
        random_state=42
    ),
    order=CHAIN_ORDER,
    random_state=42
)
clf = GridSearchCV(estimator=model, param_grid=lr_param_grid, cv=5, scoring='f1_macro')
clf.fit(X_train_scaled, y_train)
best_model = clf.best_estimator_
y_pred = best_model.predict(X_test_scaled)

# Overall performance
overall_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
print(f'Overall Macro F1: {overall_macro:.3f}')

# Subgroup analysis
subgroups = {
    'Race: White': 0,
    'Race: Black': 1,
    'Race: Other': 2,
    'Sex: Male': 3,
}

print('\n' + '='*60)
print('FAIRNESS AUDIT — Subgroup Performance')
print('='*60)
print(f'Minimum subgroup size threshold: {MIN_SAMPLES} samples\n')

subgroup_results = {}

for group_name, col_idx in subgroups.items():
    mask = demo_test[:, col_idx] == 1
    n = mask.sum()

    if n < MIN_SAMPLES:
        print(f'{group_name}: n={n} — too few samples, skipping')
        continue

    macro_f1 = f1_score(y_test[mask], y_pred[mask],
                        average='macro', zero_division=0)
    per_label = f1_score(y_test[mask], y_pred[mask],
                         average=None, zero_division=0)

    subgroup_results[group_name] = {'n': n, 'macro_f1': macro_f1, 'per_label': per_label}

    print(f'{group_name} (n={n})')
    print(f'  Macro F1: {macro_f1:.3f} | Gap from overall: {macro_f1 - overall_macro:+.3f}')
    print(f'  Per-label F1:')
    for label, score in zip(LABEL_NAMES, per_label):
        print(f'    {label:<10} {score:.3f}')
    print()

# female as complement of male
male_mask = demo_test[:, 3] == 1
female_mask = ~male_mask
n_female = female_mask.sum()

if n_female >= MIN_SAMPLES:
    female_macro = f1_score(y_test[female_mask], y_pred[female_mask],
                            average='macro', zero_division=0)
    female_per_label = f1_score(y_test[female_mask], y_pred[female_mask],
                                average=None, zero_division=0)
    subgroup_results['Sex: Female'] = {
        'n': n_female, 'macro_f1': female_macro, 'per_label': female_per_label
    }
    print(f'Sex: Female (n={n_female})')
    print(f'  Macro F1: {female_macro:.3f} | Gap from overall: {female_macro - overall_macro:+.3f}')
    print(f'  Per-label F1:')
    for label, score in zip(LABEL_NAMES, female_per_label):
        print(f'    {label:<10} {score:.3f}')
    print()

# Summary 
print('='*60)
print('SUMMARY')
print('='*60)
print(f'  {"Subgroup":<20} {"n":>6} {"Macro F1":>10} {"Gap":>8}')
print(f'  {"Overall":<20} {len(y_test):>6} {overall_macro:>10.3f} {"—":>8}')
for name, res in subgroup_results.items():
    gap = res['macro_f1'] - overall_macro
    print(f'  {name:<20} {res["n"]:>6} {res["macro_f1"]:>10.3f} {gap:>+8.3f}')