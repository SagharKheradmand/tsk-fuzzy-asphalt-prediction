import pandas as pd
from typing import Optional


def load_dataset(
    path: str, file_type: str = "excel", sheet_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Load dataset from Excel or CSV file.
    """
    if file_type.lower() == "excel":
        df = pd.read_excel(path, sheet_name=sheet_name)

        # Some sheets include section labels ("INPUT"/"OUTPUT") in the first row
        # and the real headers in the second row.
        raw_cols = [str(c).strip() for c in df.columns]
        unnamed_count = sum(c.lower().startswith("unnamed:") for c in raw_cols)
        first_col = raw_cols[0].strip().lower() if raw_cols else ""
        if first_col == "input" and unnamed_count >= max(1, len(raw_cols) // 2):
            df = pd.read_excel(path, sheet_name=sheet_name, header=1)
    elif file_type.lower() == "csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("file_type must be 'excel' or 'csv'")

    df.columns = [str(c).strip() for c in df.columns]
    return df
