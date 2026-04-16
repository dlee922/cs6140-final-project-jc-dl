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
    eval_results = {}
    for model_name, model in models.items():
        eval_results[model_name] = evaluate_model(model, (X_train, X_test, y_train, y_test))

    # save results
    df_eval = pd.DataFrame(eval_results)
    df_eval.to_csv(f"results/evaluation_{args.feature_set}.csv")
    print(f'✓ Saved: results/evaluation_{args.feature_set}.csv')
    print(df_eval)

def evaluate_model(model, data: tuple) -> dict:
    X_train, X_test, y_train, y_test = data
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    return {
        'test_f1': f1_score(y_test, y_pred_test, average='macro', zero_division=0),
        'test_accuracy': accuracy_score(y_test, y_pred_test),
        'train_f1': f1_score(y_train, y_pred_train, average='macro', zero_division=0),
        'train_accuracy': accuracy_score(y_train, y_pred_train)
    }


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
