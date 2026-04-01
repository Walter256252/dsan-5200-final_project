import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.ticker as mtick



# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = BASE_DIR / "images"

PRICES_FILE = DATA_DIR / "zhvi.csv"
RENTS_FILE = DATA_DIR / "zori.csv"
RATES_FILE = DATA_DIR / "mortgage_rates.csv"
OUTPUT_FILE = IMAGES_DIR / "dc_own_vs_rent.png"


# -----------------------------
# Helpers
# -----------------------------
def find_city_column(df: pd.DataFrame) -> str:
    """Return the most likely city/name column."""
    candidates = ["City", "RegionName", "region_name", "city", "Metro", "Name"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not find a city column. Available columns are: {list(df.columns)}"
    )


def find_state_column(df: pd.DataFrame) -> str | None:
    """Return the most likely state column if present."""
    candidates = ["State", "StateName", "state", "state_name", "StateCode"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def find_rate_column(df: pd.DataFrame) -> str:
    """Return the most likely mortgage rate column."""
    candidates = [
        "Mortgage_Rate",
        "mortgage_rate",
        "Rate",
        "rate",
        "30YFRM",
        "30yr_fixed_rate",
        "30_year_fixed_rate",
    ]
    for col in candidates:
        if col in df.columns:
            return col

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 1:
        return numeric_cols[0]

    raise ValueError(
        f"Could not identify mortgage rate column. Available columns are: {list(df.columns)}"
    )


def detect_date_columns(df: pd.DataFrame, id_vars: list[str]) -> list[str]:
    """
    Detect wide-format date columns like 2016-01, 2016-02-01, etc.
    Everything not in id_vars that parses as a date is treated as a date column.
    """
    date_cols = []
    for col in df.columns:
        if col in id_vars:
            continue
        try:
            parsed = pd.to_datetime(col, errors="raise")
            if pd.notna(parsed):
                date_cols.append(col)
        except Exception:
            continue
    return date_cols


def filter_dc_row(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to isolate Washington, DC from Zillow-style data.
    First look for exact city matches, then fall back to contains.
    """
    city_col = find_city_column(df)
    state_col = find_state_column(df)

    city_values = df[city_col].astype(str).str.strip()

    # First pass: exact likely values
    exact_matches = ["Washington, DC", "Washington", "District of Columbia"]
    mask = city_values.isin(exact_matches)

    # If there is a state column, prefer DC rows
    if state_col is not None:
        state_values = df[state_col].astype(str).str.strip()
        dc_state_mask = state_values.isin(["DC", "District of Columbia"])
        mask = mask & dc_state_mask if mask.any() else dc_state_mask & city_values.str.contains(
            "Washington", case=False, na=False
        )

    filtered = df.loc[mask].copy()

    # Fallback: contains "Washington"
    if filtered.empty:
        contains_mask = city_values.str.contains("Washington", case=False, na=False)
        if state_col is not None:
            state_values = df[state_col].astype(str).str.strip()
            dc_state_mask = state_values.isin(["DC", "District of Columbia"])
            contains_mask = contains_mask & dc_state_mask
        filtered = df.loc[contains_mask].copy()

    if filtered.empty:
        raise ValueError(
            f"Could not find Washington, DC row. City column appears to be '{city_col}'. "
            f"Sample values: {df[city_col].dropna().astype(str).head(10).tolist()}"
        )

    # If multiple rows remain, keep the first one
    return filtered.iloc[[0]].copy()


def reshape_zillow_row(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Convert one wide Zillow row to long format with Date and value_name."""
    id_vars = list(df.columns)
    date_cols = detect_date_columns(df, id_vars=[])

    if not date_cols:
        raise ValueError(
            f"No date-like columns found. Available columns are: {list(df.columns)}"
        )

    # Keep just one row, then melt only date columns
    city_row = df.iloc[[0]].copy()
    long_df = city_row[date_cols].melt(var_name="Date", value_name=value_name)
    long_df["Date"] = pd.to_datetime(long_df["Date"], errors="coerce")
    long_df = long_df.dropna(subset=["Date", value_name]).sort_values("Date")
    return long_df


def prepare_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize mortgage rates to columns Date and Mortgage_Rate."""

    # Rename columns directly (based on your actual file)
    df = df.rename(columns={
        "observation_date": "Date",
        "MORTGAGE30US": "Mortgage_Rate"
    })

    # Convert types
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Mortgage_Rate"] = pd.to_numeric(df["Mortgage_Rate"], errors="coerce")

    # Clean
    df = df.dropna(subset=["Date", "Mortgage_Rate"])
    df = df.sort_values("Date")

    return df


def monthly_payment(principal: float, annual_rate: float, n_months: int = 360) -> float:
    """30-year fixed-rate mortgage payment formula."""
    if pd.isna(principal) or pd.isna(annual_rate):
        return np.nan

    monthly_rate = annual_rate / 100 / 12

    # Handle zero-rate edge case
    if monthly_rate == 0:
        return principal / n_months

    return principal * (
        monthly_rate * (1 + monthly_rate) ** n_months
    ) / ((1 + monthly_rate) ** n_months - 1)


# -----------------------------
# Load data
# -----------------------------
prices = pd.read_csv(PRICES_FILE)
rents = pd.read_csv(RENTS_FILE)
rates = pd.read_csv(RATES_FILE)

print("Prices columns:", list(prices.columns))
print("Rents columns:", list(rents.columns))
print("Rates columns:", list(rates.columns))


# -----------------------------
# Filter Washington, DC
# -----------------------------
dc_prices_row = filter_dc_row(prices)
dc_rents_row = filter_dc_row(rents)

print("\nSelected price row:")
print(dc_prices_row.head(1).T.head(15))

print("\nSelected rent row:")
print(dc_rents_row.head(1).T.head(15))


# -----------------------------
# Reshape wide -> long
# -----------------------------
dc_prices = reshape_zillow_row(dc_prices_row, "Price")
dc_rents = reshape_zillow_row(dc_rents_row, "Rent")
rates_clean = prepare_rates(rates)


# -----------------------------
# Merge data
# -----------------------------
df = pd.merge(dc_prices, dc_rents, on="Date", how="inner")
df = pd.merge(df, rates_clean, on="Date", how="inner")
df = df.sort_values("Date").copy()

# Ensure numeric
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["Rent"] = pd.to_numeric(df["Rent"], errors="coerce")
df["Mortgage_Rate"] = pd.to_numeric(df["Mortgage_Rate"], errors="coerce")
df = df.dropna(subset=["Price", "Rent", "Mortgage_Rate"])

if df.empty:
    raise ValueError("Merged dataframe is empty. Check date alignment across files.")


# -----------------------------
# Estimate monthly owning cost
# -----------------------------
DOWN_PAYMENT_SHARE = 0.20
LOAN_SHARE = 1 - DOWN_PAYMENT_SHARE

df["Loan_Amount"] = df["Price"] * LOAN_SHARE
df["Monthly_Own_Cost"] = df.apply(
    lambda row: monthly_payment(row["Loan_Amount"], row["Mortgage_Rate"]),
    axis=1
)
df["Monthly_Rent"] = df["Rent"]


# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(11, 6))
plt.plot(df["Date"], df["Monthly_Own_Cost"], linewidth=2, label="Estimated Monthly Cost of Owning")
plt.plot(df["Date"], df["Monthly_Rent"], linewidth=2.5, label="Monthly Rent")

plt.title("Owning vs. Renting in Washington, DC\nMortgage rates drove a widening affordability gap after 2022")
plt.xlabel("Year")
plt.ylabel("Monthly Cost ($)")
plt.legend()
plt.tight_layout()
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
plt.axvline(pd.to_datetime("2022-01-01"), linestyle="--", alpha=0.5)
plt.text(pd.to_datetime("2022-01-01"), 2800, "Rate spike", rotation=90)

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
plt.close()

print(f"\nSaved chart to: {OUTPUT_FILE}")
print("\nPreview of merged data:")
print(df[["Date", "Price", "Rent", "Mortgage_Rate", "Monthly_Own_Cost", "Monthly_Rent"]].head())