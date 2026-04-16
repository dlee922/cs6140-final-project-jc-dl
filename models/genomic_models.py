'''
defines all sklearn model constructors for the genomic feature pipeline
all models use ClassifierChain with the same ordering to model label dependencies
'''
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multioutput import ClassifierChain

ORDER = [2, 1, 3, 0, 5, 6, 4]
# CNS -> Bone -> Liver -> Adrenal -> Lung -> Pleura -> LN

def logistic_regression_no_penalty():
    '''unregularized baseline, no penalty'''
    return ClassifierChain(
        LogisticRegression(
            penalty=None,
            solver='lbfgs',
            class_weight='balanced',
            max_iter=2000
        ),
        order=ORDER
    )

def logistic_regression_ridge():
    return ClassifierChain(
        LogisticRegression(
            penalty='l2',
            solver='lbfgs',
            class_weight='balanced',
            max_iter=2000,
            random_state=42
        ),
        order=ORDER,
        random_state=42
    )

def logistic_regression_lasso():
    return ClassifierChain(
        LogisticRegression(
            penalty='l1',
            solver='saga',
            class_weight='balanced',
            max_iter=5000,
            random_state=42
        ),
        order=ORDER,
        random_state=42
    )

def LDA():
    return ClassifierChain(
        LinearDiscriminantAnalysis(
            shrinkage='auto',
            solver='lsqr'
        ),
        order=ORDER,
        random_state=42
    )

def random_forest(random_state=42):
    return ClassifierChain(
        RandomForestClassifier(
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        ),
        order=ORDER,
        random_state=random_state
    )

def SVM():
    return ClassifierChain(
        SVC(
            kernel='linear',
            class_weight='balanced',
            probability=True,
            random_state=42
        ),
        order=ORDER,
        random_state=42
    )