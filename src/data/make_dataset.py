import os
from typing import Tuple

import numpy as np
import pandas as pd


RANDOM_STATE = 42
N_SAMPLES = 5000


def generate_synthetic_churn_data(
    n_samples: int = N_SAMPLES, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """
    Genera un dataset sintetico per il customer churn.

    Le feature simulate includono:
    - tenure (mesi di permanenza)
    - monthly_charges (spesa mensile)
    - total_charges
    - num_products (numero di prodotti attivi)
    - has_internet (bool)
    - has_phone (bool)
    - contract_type (month-to-month, one-year, two-year)
    - is_senior (bool)
    - has_paperless_billing (bool)

    Il target è:
    - churn (0 = rimane, 1 = abbandona)
    """
    rng = np.random.default_rng(random_state)

    tenure = rng.integers(0, 72, n_samples)  # mesi
    monthly_charges = rng.normal(60, 20, n_samples).clip(10, 200)
    num_products = rng.integers(1, 5, n_samples)
    has_internet = rng.integers(0, 2, n_samples)
    has_phone = rng.integers(0, 2, n_samples)
    is_senior = rng.integers(0, 2, n_samples)
    has_paperless_billing = rng.integers(0, 2, n_samples)
    contract_type = rng.choice(
        ["month-to-month", "one-year", "two-year"], size=n_samples, p=[0.6, 0.25, 0.15]
    )

    total_charges = tenure * monthly_charges * (0.9 + 0.2 * rng.random(n_samples))

    # Funzione logit per la probabilità di churn
    logit = (
        0.8 * (tenure < 12)
        + 0.6 * (contract_type == "month-to-month")
        + 0.4 * (monthly_charges > 90)
        + 0.3 * (has_paperless_billing == 1)
        + 0.4 * (is_senior == 1)
        - 0.5 * (num_products >= 3)
    )
    prob_churn = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, prob_churn)

    df = pd.DataFrame(
        {
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "num_products": num_products,
            "has_internet": has_internet,
            "has_phone": has_phone,
            "is_senior": is_senior,
            "has_paperless_billing": has_paperless_billing,
            "contract_type": contract_type,
            "churn": churn,
        }
    )
    return df


def get_project_paths() -> Tuple[str, str]:
    """Restituisce i path per data/raw e data/processed."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    processed_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    return raw_dir, processed_dir


def main() -> None:
    raw_dir, _ = get_project_paths()

    df = generate_synthetic_churn_data()
    raw_path = os.path.join(raw_dir, "customer_churn_synthetic.csv")
    df.to_csv(raw_path, index=False)

    print(f"Synthetic churn dataset salvato in: {raw_path}")
    print(df.head())


if __name__ == "__main__":
    main()

