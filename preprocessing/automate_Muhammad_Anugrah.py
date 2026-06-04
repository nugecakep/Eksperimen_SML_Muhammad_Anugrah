"""Automated preprocessing for SMSML final project.

Script ini merupakan konversi dari tahapan preprocessing manual pada notebook
Eksperimen_Muhammad_Anugrah.ipynb. Output berupa dataset siap latih.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "target"
RAW_DATA_PATH = Path(__file__).resolve().parents[1] / "breast_cancer_raw" / "breast_cancer_raw.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "breast_cancer_preprocessing"

def load_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)

def preprocess_data(df: pd.DataFrame):
    df_clean = df.drop_duplicates().reset_index(drop=True)
    X = df_clean.drop(columns=[TARGET_COLUMN])
    y = df_clean[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

    train_df = X_train_scaled.copy()
    train_df[TARGET_COLUMN] = y_train.values
    test_df = X_test_scaled.copy()
    test_df[TARGET_COLUMN] = y_test.values
    preprocessed_df = pd.concat([train_df, test_df], ignore_index=True)
    return train_df, test_df, preprocessed_df

def save_outputs(train_df: pd.DataFrame, test_df: pd.DataFrame, preprocessed_df: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    preprocessed_df.to_csv(output_dir / "breast_cancer_preprocessed.csv", index=False)

def main() -> None:
    df = load_dataset()
    train_df, test_df, preprocessed_df = preprocess_data(df)
    save_outputs(train_df, test_df, preprocessed_df)
    print("Preprocessing completed.")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Output folder: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
