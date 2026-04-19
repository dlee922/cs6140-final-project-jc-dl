import sys
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
import numpy as np

sys.path.append('../') 

ORDER = [2,1,3,0,5,6,4]

def dummy(strategy):
    return DummyClassifier(strategy='most_frequent')

def logistic_regression(l1_ratio, solver):
    return LogisticRegression(l1_ratio=l1_ratio, solver= solver, class_weight='balanced', max_iter=6000)
def logistic_regression_no_penalty():
    return LogisticRegression(C=np.inf, class_weight='balanced', max_iter=5000)

def LDA():
    return LinearDiscriminantAnalysis(shrinkage='auto', solver='lsqr', priors=[0.5,0.5])

def random_forest(random_state=42):
    return RandomForestClassifier(class_weight='balanced', random_state=random_state, n_jobs=-1)

def SVM():
    return SVC( 
            kernel='linear',
            class_weight='balanced',
            probability=True,
            random_state=42
        )
