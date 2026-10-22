from pathlib import Path
import warnings
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# ADIDAS SALES ANALYSIS
# Keeps the original CSV outputs and adds advanced analytics.
# Run from the C:\adidas project folder:
#     python sales_analysis.py
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "data" / "processed" / "adidas_sales_cleaned.csv"
output_folder = BASE_DIR / "analysis"

if not file_path.exists():
    print(f"Cleaned dataset not found: {file_path}")
    print("Run clean_data.py first, then run sales_analysis.py again.")
    raise SystemExit(1)

df = pd.read_csv(file_path, parse_dates=["Invoice Date"])

required_columns = [
    "Invoice Date", "Product", "Retailer", "Region", "Sales Method",
    "Price per Unit", "Units Sold", "Total Sales", "Operating Profit",
    "Operating Margin"
]
missing = [column for column in required_columns if column not in df.columns]
if missing:
    print("The cleaned dataset is missing these required columns:")
    for column in missing:
        print(f" - {column}")
    print("Check clean_data.py and the cleaned CSV column names.")
    raise SystemExit(1)

# Make numeric columns safe for calculations.
numeric_columns = [
    "Price per Unit", "Units Sold", "Total Sales",
    "Operating Profit", "Operating Margin"
]
for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df["Invoice Date"] = pd.to_datetime(df["Invoice Date"], errors="coerce")
df = df.dropna(subset=["Invoice Date", "Total Sales", "Operating Profit"])
if df.empty:
    print("No valid rows remain after checking dates and sales values.")
    raise SystemExit(1)

output_folder.mkdir(parents=True, exist_ok=True)

# Row-level source for shared website filters, drill-down, scatter plots, alerts, and export.
# Keep only columns used by the dashboard; values still come directly from the cleaned dataset.
record_columns = [
    "Invoice Date", "Product", "Retailer", "Region", "Sales Method",
    "Price per Unit", "Units Sold", "Total Sales", "Operating Profit", "Operating Margin"
]
records_for_dashboard = df[record_columns].copy()
records_for_dashboard["Invoice Date"] = pd.to_datetime(records_for_dashboard["Invoice Date"], errors="coerce").dt.strftime("%Y-%m-%d")
records_for_dashboard.to_csv(output_folder / "dashboard_records.csv", index=False)

print("\n" + "=" * 60)
print("          ADIDAS SALES BUSINESS ANALYSIS")
print("=" * 60)

# 1. Overall KPIs (same output names and metrics as the original script)
total_sales = df["Total Sales"].sum()
total_profit = df["Operating Profit"].sum()
total_units = df["Units Sold"].sum()
average_margin = df["Operating Margin"].mean() * 100

print("\n1. BUSINESS KPIs")
print("-" * 35)
print(f"Total Sales: ${total_sales:,.2f}")
print(f"Total Operating Profit: ${total_profit:,.2f}")
print(f"Total Units Sold: {total_units:,.0f}")
print(f"Average Operating Margin: {average_margin:.2f}%")
print(f"Number of Records: {len(df):,}")

# 2. Product performance
product_sales = (
    df.groupby("Product", as_index=False)
    .agg(
        Total_Sales=("Total Sales", "sum"),
        Total_Profit=("Operating Profit", "sum"),
        Units_Sold=("Units Sold", "sum")
    )
    .sort_values("Total_Sales", ascending=False)
)
print("\n2. PRODUCT PERFORMANCE")
print(product_sales.to_string(index=False))

# 3. Retailer performance
retailer_sales = (
    df.groupby("Retailer", as_index=False)
    .agg(
        Total_Sales=("Total Sales", "sum"),
        Total_Profit=("Operating Profit", "sum"),
        Units_Sold=("Units Sold", "sum")
    )
    .sort_values("Total_Sales", ascending=False)
)
print("\n3. RETAILER PERFORMANCE")
print(retailer_sales.to_string(index=False))

# 4. Regional performance
region_sales = (
    df.groupby("Region", as_index=False)
    .agg(
        Total_Sales=("Total Sales", "sum"),
        Total_Profit=("Operating Profit", "sum"),
        Units_Sold=("Units Sold", "sum")
    )
    .sort_values("Total_Sales", ascending=False)
)
print("\n4. REGIONAL PERFORMANCE")
print(region_sales.to_string(index=False))

# 5. Sales method performance
method_sales = (
    df.groupby("Sales Method", as_index=False)
    .agg(
        Total_Sales=("Total Sales", "sum"),
        Total_Profit=("Operating Profit", "sum"),
        Units_Sold=("Units Sold", "sum")
    )
    .sort_values("Total_Sales", ascending=False)
)
print("\n5. SALES METHOD PERFORMANCE")
print(method_sales.to_string(index=False))

# 6. Monthly sales and profit trend
# Month strings use YYYY-MM so chronological sorting works correctly.
df["Month"] = df["Invoice Date"].dt.to_period("M").astype(str)
monthly_sales = (
    df.groupby("Month", as_index=False)
    .agg(
        Total_Sales=("Total Sales", "sum"),
        Total_Profit=("Operating Profit", "sum")
    )
    .sort_values("Month")
    .reset_index(drop=True)
)
print("\n6. FIRST 12 MONTHLY SALES RESULTS")
print(monthly_sales.head(12).to_string(index=False))

# 7. Pricing and product economics.
# The dataset has Price per Unit, but does not contain a discount field.
pricing_analysis = (
    df.groupby("Product", as_index=False)
    .agg(
        Avg_Price_Per_Unit=("Price per Unit", "mean"),
        Total_Units_Sold=("Units Sold", "sum"),
        Total_Sales=("Total Sales", "sum"),
        Total_Profit=("Operating Profit", "sum"),
        Average_Operating_Margin_pct=("Operating Margin", "mean")
    )
    .sort_values("Total_Sales", ascending=False)
)
pricing_analysis["Average_Operating_Margin_pct"] *= 100

# 8. Month-over-month growth and IQR-based sales anomaly flags.
monthly_growth = monthly_sales.copy()
monthly_growth["Sales_Growth_pct"] = monthly_growth["Total_Sales"].pct_change() * 100
monthly_growth["Profit_Growth_pct"] = monthly_growth["Total_Profit"].pct_change() * 100
q1 = monthly_growth["Total_Sales"].quantile(0.25)
q3 = monthly_growth["Total_Sales"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
monthly_growth["Is_Sales_Anomaly"] = (
    (monthly_growth["Total_Sales"] < lower_bound) |
    (monthly_growth["Total_Sales"] > upper_bound)
)

# 9. Six-month linear trend forecast.
# This is a simple baseline, not a seasonal/causal model; use it as a
# directional estimate and validate it before business use.
def make_linear_forecast(monthly, value_column, output_column, periods=6):
    values = pd.to_numeric(monthly[value_column], errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return pd.DataFrame(columns=["Month", output_column])
    x = np.arange(len(values), dtype=float)
    if len(values) >= 2:
        slope, intercept = np.polyfit(x, values, 1)
    else:
        slope, intercept = 0.0, float(values[0])
    future_x = np.arange(len(values), len(values) + periods, dtype=float)
    predicted = slope * future_x + intercept
    # Sales/profit cannot meaningfully be negative in this baseline forecast.
    predicted = np.maximum(predicted, 0)
    last_period = pd.Period(monthly["Month"].iloc[-1], freq="M")
    future_months = [(last_period + i).strftime("%Y-%m") for i in range(1, periods + 1)]
    return pd.DataFrame({"Month": future_months, output_column: predicted})

sales_forecast = make_linear_forecast(monthly_sales, "Total_Sales", "Forecast_Sales")
profit_forecast = make_linear_forecast(monthly_sales, "Total_Profit", "Forecast_Profit")
sales_forecast = sales_forecast.merge(profit_forecast, on="Month", how="left")

# 10. Transaction-level correlations; correlation does not prove causation.
correlation_pairs = [
    ("Price per Unit", "Units Sold"),
    ("Price per Unit", "Total Sales"),
    ("Units Sold", "Total Sales"),
    ("Total Sales", "Operating Profit"),
    ("Operating Margin", "Operating Profit")
]
correlation_rows = []
for left, right in correlation_pairs:
    pair = df[[left, right]].dropna()
    corr = pair[left].corr(pair[right]) if len(pair) > 1 else np.nan
    correlation_rows.append({"Variable_1": left, "Variable_2": right,
                             "Pearson_Correlation": corr,
                             "Records_Used": len(pair)})
correlation_summary = pd.DataFrame(correlation_rows)

# Save original output files (names and core columns preserved).
product_sales.to_csv(output_folder / "product_sales.csv", index=False)
retailer_sales.to_csv(output_folder / "retailer_sales.csv", index=False)
region_sales.to_csv(output_folder / "region_sales.csv", index=False)
method_sales.to_csv(output_folder / "method_sales.csv", index=False)
monthly_sales.to_csv(output_folder / "monthly_sales.csv", index=False)

kpis = pd.DataFrame({
    "Metric": [
        "Total Sales", "Total Operating Profit", "Total Units Sold",
        "Average Operating Margin (%)", "Number of Records"
    ],
    "Value": [total_sales, total_profit, total_units, average_margin, len(df)]
})
kpis.to_csv(output_folder / "kpis.csv", index=False)

# New outputs consumed by the advanced website charts.
pricing_analysis.to_csv(output_folder / "pricing_analysis.csv", index=False)
monthly_growth.to_csv(output_folder / "monthly_growth.csv", index=False)
sales_forecast.to_csv(output_folder / "sales_forecast.csv", index=False)
correlation_summary.to_csv(output_folder / "correlation_summary.csv", index=False)

# Anomaly-only output makes flagged months easy to inspect separately.
anomaly_months = monthly_growth.loc[monthly_growth["Is_Sales_Anomaly"]].copy()
anomaly_months.to_csv(output_folder / "anomaly_months.csv", index=False)

print("\n7. PRICING ANALYSIS BY PRODUCT")
print(pricing_analysis.to_string(index=False))
print("\n8. MONTH-OVER-MONTH GROWTH")
print(monthly_growth.tail(12).to_string(index=False))
print("\n9. SIX-MONTH BASELINE FORECAST")
print(sales_forecast.to_string(index=False))
print("\n10. CORRELATION SUMMARY (correlation is not causation)")
print(correlation_summary.to_string(index=False))

print("\n" + "=" * 60)
print("ANALYSIS COMPLETED SUCCESSFULLY!")
print(f"Results saved in: {output_folder}")
print("Original dashboard CSVs retained; advanced CSVs and dashboard_records.csv added.")
print("dashboard_records.csv powers shared filters, chart drill-down, scatter plots, insights, and filtered exports.")
print("Discount impact is not calculated because the dataset has no discount column.")
print("Forecasts are simple linear-trend estimates, not guaranteed outcomes.")
print("=" * 60)
