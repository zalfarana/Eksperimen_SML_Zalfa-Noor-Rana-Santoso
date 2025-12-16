# modelling_tuning.py
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import joblib
import os

dagshub.init(
    repo_owner="zalfarana",
    repo_name="Eksperimen_SML_Zalfa-Noor-Rana-Santoso",
    mlflow=True
)

mlflow.set_tracking_uri("https://dagshub.com/zalfarana/Eksperimen_SML_Zalfa-Noor-Rana-Santoso.mlflow")
mlflow.set_experiment("titanic_experiment_zalfa_advanced")

def load_data(path):
    return pd.read_csv(path)

def plot_confusion(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4,4))
    ax.matshow(cm, cmap='Blues')
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha='center', va='center')
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def main():
    # load
    X_train = pd.read_csv("titanic_preprocessing/X_train_processed.csv")
    X_test  = pd.read_csv("titanic_preprocessing/X_test_processed.csv")
    y_train = pd.read_csv("titanic_preprocessing/y_train.csv").squeeze()
    y_test  = pd.read_csv("titanic_preprocessing/y_test.csv").squeeze()

    # model + tuning
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [5, 10, None]
    }
    grid = GridSearchCV(rf, param_grid, cv=3, scoring="accuracy", n_jobs=-1)

    mlflow.set_experiment("titanic_rf_tuning")

    with mlflow.start_run():
        grid.fit(X_train, y_train)
        best = grid.best_estimator_
        preds = best.predict(X_test)

        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, zero_division=0)
        recall = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        # Manual logging
        mlflow.log_param("best_params", grid.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        # Save model locally and log as artifact
        os.makedirs("models", exist_ok=True)
        model_path = "models/rf_best.joblib"
        joblib.dump(best, model_path)
        mlflow.log_artifact(model_path, artifact_path="model")

        # Confusion matrix plot
        plot_path = "models/confusion.png"
        plot_confusion(y_test, preds, plot_path)
        mlflow.log_artifact(plot_path, artifact_path="plots")

        print("Done. Metrics:", acc, precision, recall, f1)

if __name__ == "__main__":
    main()
