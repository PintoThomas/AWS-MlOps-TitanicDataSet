import os
import argparse
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Arguments transmis par SageMaker
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--test-dir", type=str, default=os.environ.get("SM_CHANNEL_TEST", "/opt/ml/input/data/test"))
    
    # Hyperparamètres XGBoost
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    
    args = parser.parse_args()

    # Chargement des CSV (la cible est la première colonne)
    train_df = pd.read_csv(os.path.join(args.train_dir, "train.csv"), header=None)
    test_df = pd.read_csv(os.path.join(args.test_dir, "test.csv"), header=None)

    y_train, X_train = train_df.iloc[:, 0], train_df.iloc[:, 1:]
    y_test, X_test = test_df.iloc[:, 0], test_df.iloc[:, 1:]

    # Entraînement du modèle XGBoost Classifier
    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        eval_metric="logloss",
        random_state=41
    )
    model.fit(X_train, y_train)

    # Évaluation du modèle
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    print(f"--- MÉTROLOGIE MODÈLE TITANIC ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"AUC      : {auc:.4f}")

    # Sauvegarde des poids du modèle dans le dossier officiel SageMaker
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print("Modèle XGBoost sauvegardé sous model.joblib.")