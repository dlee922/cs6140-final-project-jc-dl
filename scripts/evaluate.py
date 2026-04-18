'''
loads fitted models from disk and evalutes them on test set
run via CLI: python -m scripts.evaluate -f [clinical/genomic/combined]
'''
from sklearn.metrics import accuracy_score, f1_score
from utils.model_utils import load_train_test
from pathlib import Path
import argparse
import joblib
import pandas as pd
import os

feature_set_paths = {
    'clinical': 'data/processed/X_clinical.csv',
    'genomic': 'data/processed/X_genomic.csv',
    'combined': 'data/processed/X_combined.csv'
}

def main():
    args = get_cli_args()
    y_filepath = "data/processed/y.csv"
    X_filepath = feature_set_paths[args.feature_set]
    scale = args.feature_set in ['genomic', 'combined']
    X_train, X_test, y_train, y_test = load_train_test(X_filepath, y_filepath, scale=scale)

    # load from feature-set specific subfolder
    path = Path(f'models/fitted_models/{args.feature_set}')
    models = {}
    for item in path.iterdir():
        if item.is_file() and item.suffix == '.pkl':
            model = joblib.load(str(item))
            model_name = item.stem.replace(f'_{args.feature_set}', '')
            models[model_name] = model

    # evaluate
    LABEL_NAMES = ['Adrenal', 'Bone', 'CNS', 'Liver', 'LN', 'Lung', 'Pleura']

    eval_results = {}
    for model_name, model in models.items():
        eval_results[model_name] = evaluate_model(model, (X_train, X_test, y_train, y_test), label_names=LABEL_NAMES)

    # save results
    os.makedirs(f'results/evaluation/', exist_ok=True) # create subfolder if it doesn't exist

    df_eval = pd.DataFrame(eval_results)
    df_eval.to_csv(f'results/evaluation/evaluation_{args.feature_set}.csv')
    print(f'✓ Saved: results/evaluation/evaluation_{args.feature_set}.csv')

def evaluate_model(model, data, label_names):
    X_train, X_test, y_train, y_test = data
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    
    per_label_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
    
    results = {
        'test_f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'test_accuracy': accuracy_score(y_test, y_pred),
        'train_f1': f1_score(y_train, y_train_pred, average='macro', zero_division=0),
        'train_accuracy': accuracy_score(y_train, y_train_pred),
    }
    
    # add per label f1
    for label, score in zip(label_names, per_label_f1):
        results[f'test_f1_{label.lower()}'] = score
    
    return results

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
