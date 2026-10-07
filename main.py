
import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# 1. LOAD DATA
# ==========================================

# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Input and output files
INPUT_FILE = BASE_DIR / "data.csv"
OUTPUT_FILE = BASE_DIR / "cleaned_data.csv"
QUALITY_REPORT = BASE_DIR / "data_quality_report.csv"
METRICS_REPORT = BASE_DIR / "analytical_metrics.csv"

# Load raw dataset
df = pd.read_csv(INPUT_FILE)

print("=" * 50)
print("REAL ESTATE DATA ANALYSIS")
print("=" * 50)

print("\nOriginal dataset shape:", df.shape)
print("\nOriginal columns:")
print(df.columns.tolist())


# ==========================================
# 2. DATA CLEANING
# ==========================================

print("\nStarting data cleaning...")

# Step 1: Standardize column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(r"\s+", "_", regex=True)
)

# Step 2: Remove exact duplicate records
initial_rows = len(df)
df = df.drop_duplicates()
duplicates_removed = initial_rows - len(df)

# Step 3: Clean text columns
text_columns = df.select_dtypes(include=["object", "string"]).columns

for col in text_columns:
    df[col] = df[col].astype("string").str.strip()

# Replace empty strings with missing values
df = df.replace(r"^\s*$", pd.NA, regex=True)

# Step 4: Clean numerical columns
numeric_columns = ["price", "area", "rate_per_sqft", "bhk_count"]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col]
            .astype("string")
            .str.replace(",", "", regex=False)
            .str.replace(r"[₹]", "", regex=True)
            .str.strip(),
            errors="coerce"
        )

# Step 5: Clean categorical columns
categorical_columns = ["status", "flat_type", "locality", "company_name"]

for col in categorical_columns:
    if col in df.columns:
        df[col] = df[col].astype("string").str.strip().str.lower()

# Standardize RERA approval
if "rera_approval" in df.columns:
    df["rera_approval"] = (
        df["rera_approval"]
        .astype("string")
        .str.strip()
        .str.lower()
        .replace({
            "approved by rera": "approved",
            "not approved by rera": "not approved",
            "yes": "approved",
            "no": "not approved"
        })
    )

# Correct known locality spelling inconsistency
if "locality" in df.columns:
    df["locality"] = df["locality"].replace({
        "camelliaass": "camel iaas".replace(" ", "")
    })

# Step 6: Handle missing values
# Do not blindly replace missing prices or areas with averages.
# These records can distort property analysis.
critical_columns = ["price", "area", "locality"]

existing_critical = [
    col for col in critical_columns if col in df.columns
]

before_missing_filter = len(df)

if existing_critical:
    df = df.dropna(subset=existing_critical)

missing_critical_removed = before_missing_filter - len(df)

# Step 7: Remove invalid numerical records
if "price" in df.columns:
    df = df[df["price"] > 0]

if "area" in df.columns:
    df = df[df["area"] > 0]

if "rate_per_sqft" in df.columns:
    df.loc[df["rate_per_sqft"] <= 0, "rate_per_sqft"] = np.nan

if "bhk_count" in df.columns:
    df.loc[df["bhk_count"] <= 0, "bhk_count"] = np.nan

# Step 8: Recalculate rate per square foot
# Use price / area, assuming price and area are in
# compatible units and rate_per_sqft means price per area unit.
if "price" in df.columns and "area" in df.columns:
    df["calculated_rate_per_sqft"] = (
        df["price"] / df["area"]
    )

# Keep the original rate column for comparison.
# Fill missing rates with the calculated rate.
if "rate_per_sqft" in df.columns:
    df["rate_per_sqft"] = df["rate_per_sqft"].fillna(
        df["calculated_rate_per_sqft"]
    )
elif "calculated_rate_per_sqft" in df.columns:
    df["rate_per_sqft"] = df["calculated_rate_per_sqft"]

# Remove temporary calculation column
df = df.drop(columns=["calculated_rate_per_sqft"], errors="ignore")

# Step 9: Remove duplicates again after standardization
df = df.drop_duplicates()

print("\nData cleaning completed!")
print("Duplicates removed:", duplicates_removed)
print("Rows removed for missing critical values:", missing_critical_removed)
print("Final dataset shape:", df.shape)


# ==========================================
# 3. DATA QUALITY CHECKS
# ==========================================

print("\n" + "=" * 50)
print("DATA QUALITY REPORT")
print("=" * 50)

quality_report = pd.DataFrame({
    "column": df.columns,
    "data_type": [str(df[col].dtype) for col in df.columns],
    "missing_values": [df[col].isna().sum() for col in df.columns],
    "missing_percentage": [
        round(df[col].isna().mean() * 100, 2)
        for col in df.columns
    ],
    "unique_values": [df[col].nunique(dropna=True) for col in df.columns]
})

print("\n", quality_report.to_string(index=False))

print("\nDuplicate rows remaining:", df.duplicated().sum())

# Check invalid values
invalid_checks = {}

if "price" in df.columns:
    invalid_checks["non_positive_price"] = (df["price"] <= 0).sum()

if "area" in df.columns:
    invalid_checks["non_positive_area"] = (df["area"] <= 0).sum()

if "rate_per_sqft" in df.columns:
    invalid_checks["non_positive_rate"] = (
        df["rate_per_sqft"] <= 0
    ).sum()

if "bhk_count" in df.columns:
    invalid_checks["non_positive_bhk"] = (
        df["bhk_count"] <= 0
    ).sum()

print("\nInvalid value checks:")
for check, count in invalid_checks.items():
    print(f"{check}: {count}")

# Export quality report
quality_report.to_csv(QUALITY_REPORT, index=False)


# ==========================================
# 4. ANALYTICAL METRICS
# ==========================================

print("\n" + "=" * 50)
print("REAL ESTATE ANALYTICAL METRICS")
print("=" * 50)

metrics = {}

# Metric 1: Total properties
metrics["total_properties"] = len(df)

# Metric 2: Average property price
if "price" in df.columns:
    metrics["average_property_price"] = round(df["price"].mean(), 2)

# Metric 3: Median property price
if "price" in df.columns:
    metrics["median_property_price"] = round(df["price"].median(), 2)

# Metric 4: Maximum property price
if "price" in df.columns:
    metrics["maximum_property_price"] = round(df["price"].max(), 2)

# Metric 5: Average area
if "area" in df.columns:
    metrics["average_area"] = round(df["area"].mean(), 2)

# Metric 6: Average rate per square foot
if "rate_per_sqft" in df.columns:
    metrics["average_rate_per_sqft"] = round(
        df["rate_per_sqft"].mean(), 2
    )

# Metric 7: Locality with highest average price
if {"locality", "price"}.issubset(df.columns):
    locality_prices = df.groupby("locality")["price"].mean()
    if not locality_prices.empty:
        metrics["highest_avg_price_locality"] = locality_prices.idxmax()
        metrics["highest_locality_avg_price"] = round(
            locality_prices.max(), 2
        )

# Metric 8: Locality with highest average rate per sqft
if {"locality", "rate_per_sqft"}.issubset(df.columns):
    locality_rates = df.groupby("locality")["rate_per_sqft"].mean()
    if not locality_rates.empty:
        metrics["highest_rate_locality"] = locality_rates.idxmax()
        metrics["highest_locality_avg_rate"] = round(
            locality_rates.max(), 2
        )

# Metric 9: Ready-to-move vs under-construction pricing
if {"status", "price"}.issubset(df.columns):
    status_prices = df.groupby("status")["price"].mean()

    metrics["ready_to_move_avg_price"] = round(
        status_prices.get("ready to move", np.nan), 2
    )

    metrics["under_construction_avg_price"] = round(
        status_prices.get("under construction", np.nan), 2
    )

# Metric 10: RERA approval and average pricing
if {"rera_approval", "price"}.issubset(df.columns):
    rera_prices = df.groupby("rera_approval")["price"].mean()

    metrics["rera_approved_avg_price"] = round(
        rera_prices.get("approved", np.nan), 2
    )

    metrics["not_approved_avg_price"] = round(
        rera_prices.get("not approved", np.nan), 2
    )

# Additional metric: Correlation between area and price
if {"area", "price"}.issubset(df.columns):
    metrics["area_price_correlation"] = round(
        df["area"].corr(df["price"]), 3
    )

# Additional metric: Most common BHK configuration
if "bhk_count" in df.columns:
    bhk_counts = df["bhk_count"].mode()
    if not bhk_counts.empty:
        metrics["most_common_bhk"] = bhk_counts.iloc[0]

# Display metrics
for name, value in metrics.items():
    print(f"{name}: {value}")

# Export metrics as a two-column report
metrics_df = pd.DataFrame(
    list(metrics.items()),
    columns=["metric", "value"]
)

metrics_df.to_csv(METRICS_REPORT, index=False)


# ==========================================
# 5. ADDITIONAL GROUPED ANALYSIS
# ==========================================

print("\n" + "=" * 50)
print("GROUPED ANALYSIS")
print("=" * 50)

# Locality-level analysis
if {"locality", "price", "area", "rate_per_sqft"}.issubset(df.columns):
    locality_analysis = (
        df.groupby("locality")
        .agg(
            property_count=("price", "count"),
            average_price=("price", "mean"),
            median_price=("price", "median"),
            average_area=("area", "mean"),
            average_rate_per_sqft=("rate_per_sqft", "mean")
        )
        .round(2)
        .sort_values("average_price", ascending=False)
    )

    print("\nTop 10 localities by average price:")
    print(locality_analysis.head(10).to_string())

    locality_analysis.to_csv(
        BASE_DIR / "locality_analysis.csv"
    )

# Builder-level analysis
if {"company_name", "price", "rate_per_sqft"}.issubset(df.columns):
    builder_analysis = (
        df.groupby("company_name")
        .agg(
            property_count=("price", "count"),
            average_price=("price", "mean"),
            average_rate_per_sqft=("rate_per_sqft", "mean")
        )
        .round(2)
        .sort_values("average_rate_per_sqft", ascending=False)
    )

    print("\nTop 10 builders by average rate per sqft:")
    print(builder_analysis.head(10).to_string())

    builder_analysis.to_csv(
        BASE_DIR / "builder_analysis.csv"
    )


# ==========================================
# 6. EXPORT FINAL CLEANED DATASET
# ==========================================

df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 50)
print("ANALYSIS SUCCESSFUL")
print("=" * 50)

print(f"Cleaned dataset: {OUTPUT_FILE}")
print(f"Quality report: {QUALITY_REPORT}")
print(f"Metrics report: {METRICS_REPORT}")
print("Final rows:", len(df))
print("Final columns:", len(df.columns))