from sklearn.model_selection import cross_val_score, train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import joblib
import os

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

    """
    nested_scores = np.zeros(num_trials)

    for i in range(num_trials):
        inner_cv = KFold(n_splits=n_innersplit, shuffle=shuffle, random_state=i)
        outer_cv = KFold(n_splits=n_outersplit, shuffle=shuffle, random_state=i)
        
        trial_scores = []
        clf = GridSearchCV(estimator=model, param_grid=p_grid, cv=inner_cv, scoring=scoring) #searches for best params
        score = cross_val_score(clf, X, y, cv=outer_cv, scoring=scoring)
        nested_scores[i] = score.mean()
    
    return nested_scores

def load_train_test(X_path: str, y_path: str, scale: bool = False) -> tuple:
    '''Loads train and test data'''
    X = pd.read_csv(X_path, sep=',')
    y = pd.read_csv(y_path, sep=',')

    # drop identifiers — only drop columns that actually exist in this dataset
    id_cols = ['sampleId', 'patientId', 'GENE_PANEL']
    X = X.drop(columns=[c for c in id_cols if c in X.columns])
    y = y.drop(columns=[c for c in id_cols if c in y.columns])

    assert len(X) == len(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.2, random_state=42
    )
    if scale: # applies StandardScaler fit on training data only, fitting on full = leakage
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    return (X_train, X_test, y_train, y_test)


def train_model(model, X_train, y_train, scoring_method = 'f1_macro', param_grid=None, tune_params=True):
    '''Trains a given model from sklearn given X and y training data and labels. 
    Also performs grid search CV to find the best hyperparameters for the given model by default'''
    if tune_params: # wrap model in GridSearchCV with cv=5
        model = GridSearchCV(estimator=model, param_grid=param_grid, 
                                cv=5, scoring=scoring_method) #searches for best params
        model.fit(X_train, y_train) # fit data on training data
        return model
    else:
        model.fit(X_train, y_train)
        return model
