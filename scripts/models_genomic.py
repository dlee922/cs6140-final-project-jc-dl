import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

X = pd.read_csv('data/processed/X_combined.csv')
y = pd.read_csv('data/processed/y.csv')
y = y.drop(columns=['sampleId', 'patientId']).astype(int)

# dropping non-feature columns
X = X.drop(columns=['patientId', 'sampleId', 'GENE_PANEL'])

X = X.values
y = y.values  # shape: (455, 7)

LABEL_NAMES = [
    'Adrenal', 'Bone', 'CNS',
    'Liver', 'LN', 'Lung', 'Pleura'
]

N_SPLITS = 5
mskf = MultilabelStratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# storing results from the folds here
fold_macro_f1s = []
fold_per_label_f1s = []

for fold, (train_idx, test_idx) in enumerate(mskf.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # scale the features on the training set only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # baseline softmax logistic regression
    base_clf = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    model = MultiOutputClassifier(base_clf)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # evaluate
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    per_label_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)

    fold_macro_f1s.append(macro_f1)
    fold_per_label_f1s.append(per_label_f1)

    print(f'Fold {fold+1} | Macro F1: {macro_f1:.3f}')
    print(f'Per-label F1: { {k: round(v, 3) for k, v in zip(LABEL_NAMES, per_label_f1)} }')

# results
print('\n' + '='*60)
print(f'Mean Macro F1: {np.mean(fold_macro_f1s):.3f} +- {np.std(fold_macro_f1s):.3f}')
print('\nMean Per-label F1:')
mean_per_label = np.mean(fold_per_label_f1s, axis=0)
for label, score in zip(LABEL_NAMES, mean_per_label):
    print(f'  {label:<10} {score:.3f}')