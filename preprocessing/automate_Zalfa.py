"""
automate_Zalfa.py
Automatisasi preprocessing dataset Titanic
Sesuai dengan pipeline preprocessing pada notebook eksperimen.
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def load_data(path):
    print(f"Loading dataset dari: {path}")
    df = pd.read_csv(path)
    print(f"Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


def preprocess_data(df):
    print("\n=== MULAI PREPROCESSING ===")

    # --- DROP KOLOM TIDAK DIGUNAKAN ---
    drop_cols = ["Name", "Ticket", "Cabin"]
    print(f"Menghapus kolom: {drop_cols}")
    df = df.drop(columns=drop_cols)

    # --- SEPARATE TARGET ---
    y = df["Survived"]
    X = df.drop(columns=["Survived"])

    # --- IDENTIFIKASI KOLOM ---
    numeric_cols = ["Age", "SibSp", "Parch", "Fare"]
    categorical_cols = ["Sex", "Embarked", "Pclass"]

    print(f"Kolom numerik: {numeric_cols}")
    print(f"Kolom kategorikal: {categorical_cols}")

    # --- PIPELINE NUMERIK ---
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # --- PIPELINE KATEGORIKAL ---
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # --- COMBINE ---
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    # --- FIT & TRANSFORM ---
    print("Melakukan transformasi fitur...")
    X_processed = preprocessor.fit_transform(X)

    print("Transformasi selesai!")
    print(f"Shape setelah preprocessing: {X_processed.shape}")

    return X_processed, y, preprocessor


def split_and_save(X, y, output_folder="preprocessing/dataset_preprocessing"):
    print("\n=== SPLIT TRAIN & TEST ===")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train size: {X_train.shape}")
    print(f"Test size:  {X_test.shape}")

    # Pastikan folder output ada
    os.makedirs(output_folder, exist_ok=True)

    # Convert to DataFrame
    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)

    # Save
    X_train_df.to_csv(f"{output_folder}/X_train_processed.csv", index=False)
    X_test_df.to_csv(f"{output_folder}/X_test_processed.csv", index=False)
    y_train.to_csv(f"{output_folder}/y_train.csv", index=False)
    y_test.to_csv(f"{output_folder}/y_test.csv", index=False)

    print("\n=== DATA BERHASIL DISIMPAN ===")
    print(f"Simpan di folder: {output_folder}")


def main():
    RAW_DATA_PATH = "dataset_raw/train.csv"

    df = load_data(RAW_DATA_PATH)

    X_processed, y, preprocessor = preprocess_data(df)

    split_and_save(X_processed, y)

    print("\n=== PREPROCESSING SELESAI TANPA ERROR ===")


if __name__ == "__main__":
    main()
