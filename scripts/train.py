'''
main training script
run via CLI: python -m scripts.train -f [clinical/genomic/combined]
to train single model: python -m scripts.train -f genomic -m logistic_ridge
'''
from sklearn.metrics import accuracy_score, f1_score
from utils.model_utils import nested_cv, load_train_test, train_model
from utils.display import print_step, print_success, print_info
from models.clinical_models import dummy, logistic_regression, logistic_regression_no_penalty, LDA, SVM, random_forest
from models.genomic_models import (
    logistic_regression_no_penalty as genomic_logistic,
    logistic_regression_ridge as genomic_ridge,
    logistic_regression_lasso as genomic_lasso,
    LDA as genomic_LDA,
    random_forest as genomic_rf,
    SVM as genomic_SVM
)

from sklearn.multioutput import ClassifierChain

import yaml
import argparse
import warnings
import joblib
import os
import subprocess
import sys
import numpy as np

warnings.filterwarnings('ignore', message='Setting penalty=None')
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

ORDER = [2,1,3,0,5,6,4]

name_to_model = {
            'logistic': logistic_regression_no_penalty(),
            'logistic_ridge': logistic_regression(l1_ratio=0, solver='lbfgs'),
            'logistic_lasso': logistic_regression(l1_ratio=1, solver='liblinear'),
            'LDA': LDA(),
            'random_forest': random_forest(),
            'SVM': SVM()
            }
genomic_name_to_model = {
    'logistic': genomic_logistic(),
    'logistic_ridge': genomic_ridge(),
    'logistic_lasso': genomic_lasso(),
    'LDA': genomic_LDA(),
    'random_forest': genomic_rf(),
    'SVM': genomic_SVM(),
}

feature_set_paths = {
    'clinical': 'data/processed/X_clinical.csv',
    'genomic': 'data/processed/X_genomic.csv',
    'combined': 'data/processed/X_combined.csv'
}

def main():
    
    args = get_cli_args()
    y_filepath = f'data/processed/y_{args.task}.csv'
    X_filepath = feature_set_paths[args.feature_set]

    # genomic & combined feature sets scaled for continuous summary features
    scale = args.feature_set in ['clinical','genomic', 'combined']
    X_train, X_test, y_train, y_test = load_train_test(X_filepath, y_filepath, scale=scale) # applied on training data only
    print(y_train.shape)
    if args.task == 'binary':
        y_train = y_train.ravel()
        y_test = y_test.ravel()


    # select model registry based on feature set
    if args.feature_set == 'clinical' or args.feature_set == 'combined':
        model_registry = name_to_model
    else:
        model_registry = genomic_name_to_model

    # train models other than mlp

    models = train_models(X_train, y_train, args.feature_set, args.model, model_registry, task= args.task) 

    # train mlp
    run_mlp(args.feature_set, args.task)
    
    # extract best model if grid search else just return the single model
    best_models = {model_name: model.best_estimator_ if hasattr(model, 'best_estimator_') else model for model_name, model in models.items()}
    
    # test_logistic_model = best_models['logistic_lasso']
    # print(f"train pred distribution: {np.unique(test_logistic_model.predict(X_train), return_counts=True)}")
    # print(f"test pred distribution: {np.unique(test_logistic_model.predict(X_test), return_counts=True)}")
    # save grid 
    os.makedirs(f'models/fitted_models/{args.feature_set}/{args.task}', exist_ok=True) # create subfolder if it doesn't exist
    for model_name, model in models.items():
        if hasattr(model, 'cv_results_'):
            joblib.dump(model, f'models/fitted_models/{args.feature_set}/{args.task}/gridsearch_{model_name}.pkl')  # save full clf

    # save all fitted models
    for model_name, model in best_models.items():
        joblib.dump(model, f'models/fitted_models/{args.feature_set}/{args.task}/{model_name}_{args.feature_set}.pkl')


def train_models(X_train, y_train, feature_set, model_name, model_registry, task:str) -> dict:
    tuned_models = {}
    if model_name == 'All':
        for i, (model_, params) in enumerate(config[feature_set]['hyperparameters'].items(), 1):
            base_model = model_registry[model_] # the model

            model = __get_model(base_model, task, ORDER)

            if task == 'multilabel':
                param_grid = {f'estimator__{k}': v for k, v in params.items()}
            else:
                param_grid = params  
         
            print_step(i, len(model_registry), f'Training {model_} model')
            print_info(f"Using parameters: {param_grid}", indent=True)
            if param_grid:
                best_model = train_model(model,  X_train, y_train, param_grid=param_grid)
            else:
                best_model = train_model(model, X_train, y_train, tune_params=False)
            tuned_models[model_] = best_model
    else:
        print_step(1, 1, f'Running {model_name} model')
        base_model = model_registry[model_name]
        model = __get_model(base_model, task, ORDER)

        params = config[feature_set]['hyperparameters'][model_name]
        if task == 'multilabel':
            param_grid = {f'estimator__{k}': v for k, v in params.items()}
        else:
            param_grid = params        
        best_model = train_model(model=model, X_train=X_train, y_train=y_train, param_grid=param_grid)
        tuned_models[model_name] = best_model
    print_success("Done!")
    return tuned_models

def run_mlp(feature_set, task):
    result = subprocess.run(
        [sys.executable, 'models/mlp.py', 
         '--feature_set', feature_set,
         '--task', task],
        text=True
    )
    if result.returncode != 0:
        print(f"MLP failed:\n{result.stderr}")
    return result.returncode == 0

def get_cli_args():
    parser = argparse.ArgumentParser(description='Description')
    parser.add_argument('--feature_set',
                        '-f',
                        type=str,
                        choices = ['clinical', 'genomic', 'combined'],
                        help='select clinical, genomic, or combined',
                        required=True)
    parser.add_argument('--model',
                        '-m',
                        type=str,
                        choices= ['logistic', 'logistic_ridge', 'logistic_lasso', 'random_forest', 'LDA', 'SVM', 'All'],
                        help='select which model to run',
                        required=False,
                        default='All')
    parser.add_argument('--task',
                        '-t',
                        type=str,
                        help='select binary or multilabel prediction',
                        choices =['binary', 'multilabel'],
                        required=True)
    return parser.parse_args()

# helper to check whether to use classifier chain
def __get_model(base_model, task: str, order):
    if task == 'multilabel':
        model = ClassifierChain(base_model, order=order, random_state=42)
    else:
        model = base_model
    return model

if __name__ == "__main__":
    main()
