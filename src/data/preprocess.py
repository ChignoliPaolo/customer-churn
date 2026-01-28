import os
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_project_paths() -> Tuple[str, str]:
    """Restituisce i path per data/raw e data/processed."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_dir = os.path.join(base_dir, "data", "raw")
    processed_dir = os.path.join(base_dir, "data", "processed")
    return raw_dir, processed_dir


def load_raw_data(filename: str = "customer_churn_synthetic.csv") -> pd.DataFrame:
    raw_dir, _ = get_project_paths()
    path = os.path.join(raw_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File {path} non trovato. Esegui prima `python -m src.data.make_dataset`."
        )
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Applica un semplice preprocessing:
    - separazione target
    - standardizzazione numeriche
    - one-hot encoding delle categoriche
    """
    target_col = "churn"
    y = df[target_col]
    X = df.drop(columns=[target_col])

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    X_num = scaler.fit_transform(X[numeric_cols])
    X_cat = encoder.fit_transform(X[categorical_cols]) if categorical_cols else None

    if X_cat is not None:
        X_processed = pd.concat(
            [
                pd.DataFrame(X_num, columns=numeric_cols),
                pd.DataFrame(
                    X_cat, columns=encoder.get_feature_names_out(categorical_cols)
                ),
            ],
            axis=1,
        )
    else:
        X_processed = pd.DataFrame(X_num, columns=numeric_cols)

    return X_processed, y


def save_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    _, processed_dir = get_project_paths()
    os.makedirs(processed_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    train = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
    test = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)

    train.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(processed_dir, "test.csv"), index=False)

    print(f"Dati di train salvati in: {os.path.join(processed_dir, 'train.csv')}")
    print(f"Dati di test salvati in: {os.path.join(processed_dir, 'test.csv')}")


def main() -> None:
    df = load_raw_data()
    X, y = preprocess(df)
    save_train_test(X, y)


if __name__ == "__main__":
    main()

