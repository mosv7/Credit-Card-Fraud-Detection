import argparse
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

from credit_fraud_utils_data import load_data, prepare_and_split_data
from credit_fraud_utils_eval import find_optimal_threshold, evaluate_model, print_evaluation_report

def argument_parser():
    parser = argparse.ArgumentParser(description='Credit Card Fraud Detection Trainig Pipline')
    parser.add_argument(
    '--data_path',
    type=str, 
    default=r'..\Data\creditcard.csv', 
    help='Path to the dataset CSV file'
    )

    parser.add_argument(
    '--model_type',
    type = str, 
    default='xgb', 
    choices=['lr', 'rf', 'voting', 'xgb'],
    help='Model to train: lr (Logistic Regression), rf (Random Forest), voting (Voting Classifier), xgb (XGBoost)'
    )

    parser.add_argument(
    '--output_path',
    type= str,
    default = r'..\Model\model.pkl'
    )

    return parser.parse_args()

def main():
    args = argument_parser()

    print(f'Loading data from: {args.data_path}')
    df = load_data(args.data_path)

    print('preparing and splitting data (Stratified Train/Val/Test)')
    X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler = prepare_and_split_data(df)

    print(f'Training selected Model: [{args.model_type.upper()}]...')

    if args.model_type == 'lr':
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif args.model_type == 'rf':
        model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=1)
    elif args.model_type == 'voting':
        clf_lr = LogisticRegression(max_iter=1000, random_state=42)
        clf_rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        model = VotingClassifier(estimators=[('lr', clf_lr), ('rf', clf_rf)], voting='soft')
    elif args.model_type == 'xgb':
        model = XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0,
            reg_alpha=0.0,
            reg_lambda=1.0,
            scale_pos_weight=1,
            objective='binary:logistic',
            eval_metric='aucpr',
            random_state=42,
            n_jobs=-1
        )

    model.fit(X_train_scaled, y_train)
    print('Optimizing threshold on Validation Set...')
    val_probs = model.predict_proba(X_val_scaled)[:, 1]
    best_thresh, best_val_f1 = find_optimal_threshold(y_val, val_probs)

    val_results = evaluate_model(y_val, val_probs, threshold=best_thresh)
    print_evaluation_report(f'Validation Set ({args.model_type.upper()})', val_results)

    test_probs = model.predict_proba(X_test_scaled)[:, 1]
    test_results = evaluate_model(y_test, test_probs, threshold=best_thresh)
    print_evaluation_report(f'Holdout Test Set ({args.model_type.upper()})', test_results)

    model_payload = {
        'model': model, 
        'threshold': best_thresh,
        'scaler': scaler
    }

    joblib.dump(model_payload, args.output_path)
    print(f"\nModel bundle successfuly saved to '{args.output_path}'!")
    print(f'Contains: Trained {args.model_type.upper()} Model, Optimal Threshold ({best_thresh:.4f}), and Fitted Scaler.')

if __name__ == '__main__':
    main()