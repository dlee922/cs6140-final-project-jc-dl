'''
main training script
run via CLI: python -m scripts.train -f [clinical/genomic/combined]
to train single model: python -m scripts.train -f genomic -m logistic_ridge
'''
from sklearn.metrics import accuracy_score, f1_score
from utils.model_utils import nested_cv, load_train_test, train_model, scale_features
from utils.display import print_step, print_success, print_info
from models.clinical_models import dummy, logistic_regression, logistic_regression_no_penalty, LDA, SVM, random_forest, MLP
from models.genomic_models import (
    logistic_regression_no_penalty as genomic_logistic,
    logistic_regression_ridge as genomic_ridge,
    logistic_regression_lasso as genomic_lasso,
    LDA as genomic_LDA,
    random_forest as genomic_rf,
    SVM as genomic_SVM
)
import yaml
import argparse
import warnings
import joblib
import os
    
warnings.filterwarnings('ignore', message='Setting penalty=None')
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

name_to_model = {
                 'logistic': logistic_regression_no_penalty(),
                 'logistic_ridge': logistic_regression(l1_ratio=0, solver='lbfgs'),
                 'logistic_lasso': logistic_regression(l1_ratio=1, solver='liblinear'),
                 'LDA': LDA(),
                 'random_forest': random_forest()
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
    y_filepath = "data/processed/y.csv"
    X_filepath = feature_set_paths[args.feature_set]

    # genomic & combined feature sets scaled for continuous summary features
    scale = args.feature_set in ['genomic', 'combined']
    X_train, X_test, y_train, y_test = load_train_test(X_filepath, y_filepath, scale=scale) # applied on training data only

    # select model registry based on feature set
    if args.feature_set == 'clinical' or args.feature_set == 'combined':
        model_registry = name_to_model
    else:
        model_registry = genomic_name_to_model

    models = train_models(X_train, y_train, args.feature_set, args.model, model_registry)
    
    # create subfolder if it doesn't exist
    os.makedirs(f'models/fitted_models/{args.feature_set}', exist_ok=True)

    # save all fitted models
    for model_name, model in models.items():
        joblib.dump(model, f'models/fitted_models/{args.feature_set}/{model_name}_{args.feature_set}.pkl')


def train_models(X_train, y_train, feature_set, model_name, model_registry) -> dict:
    tuned_models = {}
    if model_name == 'All':
        for i, (model_, params) in enumerate(config[feature_set]['hyperparameters'].items(), 1):
            model = model_registry[model_]
            param_grid = {f'estimator__{k}': v for k, v in params.items()}
            print_step(i, len(model_registry), f'Training {model_} model')
            print_info(f"Using parameters: {param_grid}", indent=True)
            if param_grid:
                best_model = train_model(model, X_train, y_train, param_grid=param_grid)
            else:
                best_model = train_model(model, X_train, y_train, tune_params=False)
            tuned_models[model_] = best_model
    else:
        print_step(1, 1, f'Running {model_name} model')
        model = model_registry[model_name]
        params = config[feature_set]['hyperparameters'][model_name]
        param_grid = {f'estimator__{k}': v for k, v in params.items()}
        best_model = train_model(model=model, X_train=X_train, y_train=y_train, param_grid=param_grid)
        tuned_models[model_name] = best_model
    print_success("Done!")
    return tuned_models

def get_cli_args():
    parser = argparse.ArgumentParser(description='Description')
    parser.add_argument('--feature_set',
                        '-f',
                        type=str,
                        help='select clinical, genomic, or combined',
                        required=True)
    parser.add_argument('--model',
                        '-m',
                        type=str,
                        help='select which model to run',
                        required=False,
                        default='All')
    return parser.parse_args()

if __name__ == "__main__":
    main()
