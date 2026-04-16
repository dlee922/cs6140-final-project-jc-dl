import sys
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multioutput import ClassifierChain
import pandas as pd
import numpy as np
from utils.model_utils import nested_cv, load_train_test


sys.path.append('../') 

ORDER = [2,1,3,0,5,6,4]

def dummy(strategy):
    return ClassifierChain(DummyClassifier(strategy='most_frequent'), order=ORDER)

def logistic_regression(l1_ratio, solver):
    return ClassifierChain(LogisticRegression(l1_ratio=l1_ratio, solver= solver, class_weight='balanced', max_iter=2000), order=ORDER)

def logistic_regression_no_penalty():
    return ClassifierChain(LogisticRegression(C=np.inf, class_weight='balanced', max_iter=2000), order=ORDER)

def LDA():
    return ClassifierChain(LinearDiscriminantAnalysis(shrinkage='auto', solver='lsqr', priors=[0.5,0.5]), order=ORDER)

def random_forest(random_state=42):
    return ClassifierChain(RandomForestClassifier(class_weight='balanced', random_state=random_state, n_jobs=-1), order=ORDER)

def SVM():
    pass

def MLP():
    pass

