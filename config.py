from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class DataConfig:
    path: str = "data/Asphalt-Dataset-ToClass.xlsx"
    file_type: str = "excel"  # "excel" or "csv"
    sheet_name: Optional[str] = None

    # Input feature columns
    input_cols: Tuple[str, ...] = (
        "Viscosity (η) Pa.s",
        "% A.C. by wt. (Pb)",
        "% Eff. AC (Pbe)",
        "Max. Theo. (Gmm)",
        "% Air Voids",
        "Unit Wt. (kg/m³)",
        "p200 (Percent Passing)",
        "p4 (Cumulative Percent Retained)",
        "p38 (Cumulative Percent Retained)",
        "p34 (Cumulative Percent Retained)",
    )

    # Output target columns
    output_cols: Tuple[str, ...] = (
        "Adj. Stability kN",
        "Flow (mm)",
        "ITSM (Mpa) - 20 deg",
        "ITSM (Mpa) - 30 deg",
    )


@dataclass
class ModelConfig:
    n_rules: int = 4
    m: float = 2.2
    max_iter: int = 100
    tol: float = 1e-5
    seed: int = 42
    ridge_lambda: float = 1e-2


@dataclass
class ExperimentConfig:
    train_ratio: float = 0.8
    normalize_x: bool = True
    shuffle_before_split: bool = True
    split_seed: int = 42
