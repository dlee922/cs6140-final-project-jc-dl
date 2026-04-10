import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split, KFold, GridSearchCV
import numpy as np
from sklearn.metrics import f1_score
import pandas as pd

def plot_missing_overall(df, threshold=0.05):
    miss = df.isnull().mean().sort_values(ascending=False)
    miss = miss[miss > 0]  # only variables with any missingness
    print(len(miss))

    fig, ax = plt.subplots(figsize=(10, len(miss) * 0.35 + 1))
    colors = ['tomato' if v > threshold else 'steelblue' for v in miss]
    ax.barh(miss.index, miss.values, color=colors)
    ax.axvline(threshold, color='black', linestyle='--', linewidth=1, label=f'{int(threshold*100)}% threshold')
    ax.set_xlabel('Proportion Missing')
    ax.set_title('Overall Missingness per Variable')
    ax.legend()
    plt.tight_layout()
    plt.savefig('missing_overall.png', dpi=150)
    plt.show()


def nested_cv(model, p_grid, X, y, 
              num_trials=10, 
              n_innersplit=5, 
              n_outersplit=5, 
              shuffle=True,
              scoring='f1_macro'):
    """
    Performs nested cross-validation with multiple random trials to obtain
    an unbiased estimate of model generalization performance.
    
    Nested CV separates hyperparameter tuning (inner loop) from performance
    evaluation (outer loop), preventing optimistic bias that arises when the
    same data is used for both tuning and evaluation.
    
    Parameters
    ----------
    model : sklearn estimator
        The base model to evaluate (e.g. LogisticRegression(), LDA()).
    p_grid : dict
        Hyperparameter grid passed to GridSearchCV.
        e.g. {'C': [0.01, 0.1, 1, 10]}
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,) or (n_samples, n_labels)
        Target variable. Supports multilabel format.
    num_trials : int, default=10
        Number of random trials with different seeds to stabilize score
        estimates, particularly important for small datasets.
    n_innersplit : int, default=5
        Number of folds in the inner CV loop for hyperparameter tuning.
    n_outersplit : int, default=5
        Number of folds in the outer CV loop for performance evaluation.
    shuffle : bool, default=True
        Whether to shuffle data before splitting into folds.
    
    Returns
    -------
    nested_scores : np.ndarray of shape (num_trials,)
        Mean outer CV score for each trial. Average these for final
        performance estimate.
    
    Notes
    -----
    Total model fits = num_trials x n_outersplit x n_innersplit x len(param_grid)
    For large param grids, consider using RandomizedSearchCV instead of
    GridSearchCV to reduce computational cost.
    
    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> model = LogisticRegression()
    >>> p_grid = {'C': [0.01, 0.1, 1, 10]}
    >>> scores = nested_cv(model, p_grid, X_train, y_train)
    >>> print(f"Mean score: {scores.mean():.3f} +/- {scores.std():.3f}")
    """
    nested_scores = np.zeros(num_trials)

    for i in range(num_trials):
        inner_cv = KFold(n_splits=n_innersplit, shuffle=shuffle, random_state=i)
        outer_cv = KFold(n_splits=n_outersplit, shuffle=shuffle, random_state=i)
        
        trial_scores = []
        
        for train_idx, test_idx in outer_cv.split(X, y): # split data into train and test to prevent optimistic bias on param search
            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]
            
            clf = GridSearchCV(estimator=model, param_grid=p_grid, 
                             cv=inner_cv, scoring=scoring) #searches for best params
            clf.fit(X_train_fold, y_train_fold) # fit data on training fold
            
            y_pred = clf.predict(X_test_fold) # test 
            score = f1_score(y_test_fold, y_pred, average='macro', zero_division=0)
            trial_scores.append(score)
        

        nested_scores[i] = np.mean(trial_scores)
    
    return nested_scores


def load_train_test(): 
    X = pd.read_csv("../../data/processed/X_clinical.csv", sep=',')
    y = pd.read_csv("../../data/processed/y.csv", sep=',')

    # drop sampleId, patientId, and GENE_PANEL
    X = X.drop(columns=['sampleId', 'patientId', 'GENE_PANEL'])
    y = y.drop(columns=['sampleId', 'patientId'])

    assert(len(X) == len(y))

    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)
    
    return (X_train, X_test, y_train, y_test)



