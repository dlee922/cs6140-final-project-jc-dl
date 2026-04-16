import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import ClassifierChain
from sklearn.pipeline import Pipeline

# ── 1. Load data ──────────────────────────────────────────────────────────────
X = pd.read_csv('data/processed/X_combined.csv')
y = pd.read_csv('data/processed/y.csv')
y = y.drop(columns=['sampleId', 'patientId']).astype(int)

X = X.drop(columns=['patientId', 'sampleId', 'GENE_PANEL'])
X = X.values
y = y.values

LABEL_NAMES = ['Adrenal', 'Bone', 'CNS', 'Liver', 'LN', 'Lung', 'Pleura']
CHAIN_ORDER = [2, 1, 3, 0, 5, 6, 4]

# ── 2. Train/test split — consistent with Jason's pipeline ───────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f'Train size: {len(X_train)} | Test size: {len(X_test)}')
print(f'Positives per label (train): {y_train.sum(axis=0)}')
print(f'Positives per label (test):  {y_test.sum(axis=0)}')

# ── 3. Scale features ─────────────────────────────────────────────────────────
# fit on train only — consistent with leakage prevention
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── 4. Helper to print results ────────────────────────────────────────────────
def print_results(name, y_test, y_pred, y_train, y_train_pred):
    test_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    train_f1 = f1_score(y_train, y_train_pred, average='macro', zero_division=0)
    per_label = f1_score(y_test, y_pred, average=None, zero_division=0)
    acc = accuracy_score(y_test, y_pred)

    print(f'\n{name}')
    print(f'  Train F1: {train_f1:.3f} | Test F1: {test_f1:.3f} | Gap: {train_f1 - test_f1:+.3f}')
    print(f'  Test Accuracy: {acc:.3f}')
    print(f'  Per-label F1:')
    for label, score in zip(LABEL_NAMES, per_label):
        print(f'    {label:<10} {score:.3f}')
    return test_f1, train_f1, per_label

results = {}

# ── 5. Baseline LR — no penalty, no grid search ───────────────────────────────
baseline = ClassifierChain(
    LogisticRegression(
        penalty=None,
        class_weight='balanced',
        max_iter=2000,
        random_state=42
    ),
    order=CHAIN_ORDER,
    random_state=42
)
baseline.fit(X_train_scaled, y_train)
y_pred = baseline.predict(X_test_scaled)
y_train_pred = baseline.predict(X_train_scaled)
results['Baseline LR'] = print_results(
    'Logistic Regression (Baseline)',
    y_test, y_pred, y_train, y_train_pred
)

# ── 6. Ridge and Lasso — GridSearchCV over C ─────────────────────────────────
lr_param_grid = {'estimator__C': np.logspace(-3, 3, 7)}

for name, penalty, solver, max_iter in [
    ('Ridge (L2)', 'l2', 'lbfgs', 2000),
    ('Lasso (L1)', 'l1', 'saga', 5000)
]:
    model = ClassifierChain(
        LogisticRegression(
            penalty=penalty,
            solver=solver,
            class_weight='balanced',
            max_iter=max_iter,
            random_state=42
        ),
        order=CHAIN_ORDER,
        random_state=42
    )
    clf = GridSearchCV(
        estimator=model,
        param_grid=lr_param_grid,
        cv=5,
        scoring='f1_macro'
    )
    clf.fit(X_train_scaled, y_train)
    best = clf.best_estimator_
    y_pred = best.predict(X_test_scaled)
    y_train_pred = best.predict(X_train_scaled)
    print(f'  Best C: {clf.best_params_}')
    results[name] = print_results(
        name, y_test, y_pred, y_train, y_train_pred
    )

# ── 7. LDA with shrinkage ─────────────────────────────────────────────────────
# using lsqr solver consistent with Jason's pipeline
# priors=[0.5, 0.5] uses uniform priors to handle class imbalance
lda = ClassifierChain(
    LinearDiscriminantAnalysis(
        solver='lsqr',
        shrinkage='auto',
        priors=None  # let LDA estimate from data, consistent with our pipeline
    ),
    order=CHAIN_ORDER,
    random_state=42
)
lda.fit(X_train_scaled, y_train)
y_pred = lda.predict(X_test_scaled)
y_train_pred = lda.predict(X_train_scaled)
results['LDA'] = print_results(
    'LDA (Shrinkage)', y_test, y_pred, y_train, y_train_pred
)

# ── 8. Random Forest — GridSearchCV ──────────────────────────────────────────
rf_param_grid = {
    'estimator__n_estimators': [100, 200, 300],
    'estimator__max_depth': [3, 5, 10],
    'estimator__criterion': ['gini', 'log_loss'],
    'estimator__min_samples_leaf': [1, 5, 10, 20],
    'estimator__min_samples_split': [2, 10, 20, 30]
}
rf_model = ClassifierChain(
    RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    order=CHAIN_ORDER,
    random_state=42
)
rf_clf = GridSearchCV(
    estimator=rf_model,
    param_grid=rf_param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)
rf_clf.fit(X_train_scaled, y_train)
best_rf = rf_clf.best_estimator_
y_pred = best_rf.predict(X_test_scaled)
y_train_pred = best_rf.predict(X_train_scaled)
print(f'  Best RF params: {rf_clf.best_params_}')
results['Random Forest'] = print_results(
    'Random Forest (tuned)', y_test, y_pred, y_train, y_train_pred
)

# ── 9. SVM ────────────────────────────────────────────────────────────────────
svm = ClassifierChain(
    SVC(
        kernel='linear',
        class_weight='balanced',
        probability=True,
        random_state=42
    ),
    order=CHAIN_ORDER,
    random_state=42
)
svm.fit(X_train_scaled, y_train)
y_pred = svm.predict(X_test_scaled)
y_train_pred = svm.predict(X_train_scaled)
results['SVM'] = print_results(
    'SVM (Linear)', y_test, y_pred, y_train, y_train_pred
)

# ── 10. Summary ───────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('SUMMARY — Test Macro F1')
print('='*60)
for name, (test_f1, train_f1, _) in results.items():
    print(f'  {name:<30} Test: {test_f1:.3f} | Train: {train_f1:.3f} | Gap: {train_f1-test_f1:+.3f}')