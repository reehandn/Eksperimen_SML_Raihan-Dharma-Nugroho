import pandas as pd
import numpy as np
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =============================
# Load Data
# =============================
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


# =============================
# Basic Cleaning
# =============================
def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace('?', 'Unknown')
    return df


# =============================
# Feature Engineering
# =============================
def feature_engineering(df: pd.DataFrame):
    df["income"] = df["income"].map({">50K": 1, "<=50K": 0})

    y = df["income"].copy()
    X = df.drop("income", axis=1).copy()

    X.drop(columns=["fnlwgt", "education"], inplace=True)

    X["capital_gain_binary"] = (X["capital-gain"] > 0).astype(int)
    X["capital_loss_binary"] = (X["capital-loss"] > 0).astype(int)

    X.drop(columns=["capital-gain", "capital-loss"], inplace=True)

    return X, y


# =============================
# Build Preprocessor Pipeline
# =============================
def build_preprocessor():
    numeric_features = [
        "age",
        "education-num",
        "hours-per-week",
        "capital_gain_binary",
        "capital_loss_binary"
    ]

    categorical_features = [
        "workclass",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )

    return preprocessor, numeric_features, categorical_features


# =============================
# Full Preprocessing
# =============================
def preprocess_data(input_path: str, output_path: str):
    df = load_data(input_path)
    df = basic_cleaning(df)

    X_raw, y = feature_engineering(df)

    preprocessor, num_feats, cat_feats = build_preprocessor()
    X_processed = preprocessor.fit_transform(X_raw)

    cat_feature_names = (
        preprocessor
        .named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(cat_feats)
    )

    all_feature_names = num_feats + list(cat_feature_names)

    X_processed_df = pd.DataFrame(
        X_processed,
        columns=all_feature_names,
        index=X_raw.index
    )

    final_df = pd.concat([X_processed_df, y], axis=1)
    final_df.to_csv(output_path, index=False)

    return final_df, preprocessor


# =============================
# Run Script
# =============================
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(BASE_DIR)

    INPUT_PATH = os.path.join(ROOT_DIR, "adult_raw.csv")
    OUTPUT_PATH = os.path.join(BASE_DIR, "adult_preprocessed.csv")

    preprocess_data(INPUT_PATH, OUTPUT_PATH)
    print("Preprocessing selesai")
