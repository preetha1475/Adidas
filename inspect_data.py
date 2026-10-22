
from pathlib import Path
import pandas as pd

# Project data folder
data_folder = Path("data")

# Find CSV and Excel files
files = list(data_folder.glob("*.csv"))
files += list(data_folder.glob("*.xlsx"))
files += list(data_folder.glob("*.xls"))

if not files:
    print("No dataset found!")
    print("Please place your Adidas CSV or Excel file inside the data folder.")
else:
    file_path = files[0]
    print(f"\nDataset found: {file_path.name}")

    # Read the dataset
    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    print("\n" + "=" * 50)
    print("ADIDAS SALES DATASET OVERVIEW")
    print("=" * 50)

    print("\nFirst 5 rows:")
    print(df.head().to_string())

    print("\nDataset shape:")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nBasic statistics:")
    print(df.describe(include="all").to_string())

    print("\nInspection completed!")
