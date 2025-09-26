import pandas as pd

class DataValidator:
    """Basic sanity checks for input DataFrames."""

    @staticmethod
    def check_dataframe(df: pd.DataFrame) -> bool:
        if df is None or df.empty:
            print("❌ DataFrame is empty.")
            return False
        if "Close" not in df.columns:
            print("❌ 'Close' column missing.")
            return False
        return True

    @staticmethod
    def check_for_nans(df: pd.DataFrame) -> bool:
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            print(f"❌ Found {nan_count} missing values.")
            return False
        return True

    @staticmethod
    def validate(df: pd.DataFrame) -> bool:
        return DataValidator.check_dataframe(df) and DataValidator.check_for_nans(df)
