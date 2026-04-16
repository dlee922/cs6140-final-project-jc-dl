import sys
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import ClassifierChain
import numpy as np

sys.path.append('../') 

ORDER = [2,1,3,0,5,6,4]

def dummy(strategy):
    return ClassifierChain(DummyClassifier(strategy='most_frequent'), order=ORDER)

def logistic_regression(l1_ratio, solver):
    return ClassifierChain(LogisticRegression(l1_ratio=l1_ratio, solver= solver, class_weight='balanced', max_iter=5000), order=ORDER)

def logistic_regression_no_penalty():
    return ClassifierChain(LogisticRegression(C=np.inf, class_weight='balanced', max_iter=5000), order=ORDER)

def LDA():
    return ClassifierChain(LinearDiscriminantAnalysis(shrinkage='auto', solver='lsqr', priors=[0.5,0.5]), order=ORDER)

def random_forest(random_state=42):
    return ClassifierChain(RandomForestClassifier(class_weight='balanced', random_state=random_state, n_jobs=-1), order=ORDER)

def SVM():
    pass

def MLP():
    pass

