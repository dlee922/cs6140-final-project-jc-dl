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
    return LogisticRegression(
            C = np.inf,
            solver='lbfgs',
            class_weight='balanced',
            max_iter=2000)

def logistic_regression_ridge():
    return LogisticRegression(
            l1_ratio = 0,
            solver='lbfgs',
            class_weight='balanced',
            max_iter=2000,
            random_state=42
        )

def logistic_regression_lasso():
    return LogisticRegression(
            l1_ratio= 1,
            solver='liblinear',
            class_weight='balanced',
            max_iter=5000,
            random_state=42
        )

def LDA():
    return LinearDiscriminantAnalysis(
            shrinkage='auto',
            solver='lsqr'
        )

def random_forest(random_state=42):
    return RandomForestClassifier(
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )

def SVM():
    return SVC(
            kernel='linear',
            class_weight='balanced',
            probability=True,
            random_state=42
        )
