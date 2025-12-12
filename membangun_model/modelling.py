import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("Loading data...")

    X_train = pd.read_csv("titanic_preprocessing/X_train_processed.csv")
    X_test  = pd.read_csv("titanic_preprocessing/X_test_processed.csv")
    y_train = pd.read_csv("titanic_preprocessing/y_train.csv").squeeze()
    y_test  = pd.read_csv("titanic_preprocessing/y_test.csv").squeeze()

    # *** WAJIB - SET TRACKING URI ***
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # *** Buat experiment khusus agar tidak bentrok ***
    mlflow.set_experiment("titanic_experiment_zalfa")

    # Aktifkan autolog setelah tracking URI diset
    mlflow.sklearn.autolog()

    with mlflow.start_run():
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        print("Accuracy:", acc)
        print(classification_report(y_test, preds))


if __name__ == "__main__":
    main()
