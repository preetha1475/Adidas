
from pathlib import Path
import pandas as pd

# Project folders
data_folder = Path("data")
output_folder = data_folder / "processed"
output_folder.mkdir(parents=True, exist_ok=True)

# Find the Excel file
files = list(data_folder.glob("*.xlsx"))

if not files:
    print("ERROR: Place the Excel file inside the data folder.")
else:
    file_path = files[0]
    print("Reading:", file_path.name)

    # The actual column headers are on Excel row 4.
    # header=3 means skip the first 3 rows.
    df = pd.read_excel(file_path, header=4)
 
    # Remove completely empty rows and columns
    df = df.dropna(how="all")
    df = df.dropna(axis=1, how="all")

    # Remove accidental spaces in column names
    df.columns = df.columns.astype(str).str.strip()

    # Remove rows that do not contain a retailer
    df = df.dropna(subset=["Retailer"])

    # Convert date column
    df["Invoice Date"] = pd.to_datetime(
        df["Invoice Date"], errors="coerce"
    )

    # Convert numeric columns
    numeric_columns = [
        "Retailer ID",
        "Price per Unit",
        "Units Sold",
        "Total Sales",
        "Operating Profit",
        "Operating Margin",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Report duplicates before removing them
    duplicate_count = df.duplicated().sum()
    print("Duplicate rows before cleaning:", duplicate_count)

    # Remove exact duplicate records
    df = df.drop_duplicates().copy()

    # Check important missing values
    print("\nMissing values after cleaning:")
    print(df.isnull().sum().to_string())

    # Save cleaned dataset
    output_file = output_folder / "adidas_sales_cleaned.csv"
    df.to_csv(output_file, index=False)

    print("\nCLEANING COMPLETED!")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Column names:", df.columns.tolist())
    print("Saved to:", output_file)
