import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = BASE_DIR / "images"

PRICES_FILE = DATA_DIR / "zhvi.csv"
RENTS_FILE = DATA_DIR / "zori.csv"
OUTPUT_FILE = IMAGES_DIR / "price_vs_yield_interactive.html"


def detect_date_columns(df: pd.DataFrame) -> list[str]:
    date_cols = []
    for col in df.columns:
        try:
            parsed = pd.to_datetime(col, errors="raise")
            if pd.notna(parsed):
                date_cols.append(col)
        except Exception:
            continue
    return date_cols


def reshape_zillow(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    id_cols = [
        col for col in [
            "RegionID", "SizeRank", "RegionName", "RegionType",
            "StateName", "State", "Metro", "CountyName"
        ] if col in df.columns
    ]
    date_cols = detect_date_columns(df)

    long_df = df[id_cols + date_cols].melt(
        id_vars=id_cols,
        value_vars=date_cols,
        var_name="Date",
        value_name=value_name
    )
    long_df["Date"] = pd.to_datetime(long_df["Date"], errors="coerce")
    long_df[value_name] = pd.to_numeric(long_df[value_name], errors="coerce")
    long_df = long_df.dropna(subset=["Date", value_name])

    return long_df


def classify_market(price: float, yield_pct: float) -> str:
    if pd.isna(price) or pd.isna(yield_pct):
        return "Unknown"
    if price >= 500000 and yield_pct < 6:
        return "High-cost / low-yield"
    elif price < 300000 and yield_pct >= 8:
        return "Lower-cost / high-yield"
    else:
        return "Middle market"


# Load data
prices = pd.read_csv(PRICES_FILE)
rents = pd.read_csv(RENTS_FILE)

# Reshape
prices_long = reshape_zillow(prices, "HomeValue")
rents_long = reshape_zillow(rents, "MonthlyRent")

# Merge
df = pd.merge(
    prices_long,
    rents_long,
    on=["RegionName", "StateName", "State", "Metro", "CountyName", "Date"],
    how="inner"
)

# Keep city rows if RegionType exists
if "RegionType" in df.columns:
    df = df[df["RegionType_x"].fillna(df.get("RegionType_y", "")) == "city"] if "RegionType_x" in df.columns else df[df["RegionType"] == "city"]

# Use a simpler RegionType fix
if "RegionType_x" in df.columns:
    df["RegionType"] = df["RegionType_x"]
elif "RegionType_y" in df.columns:
    df["RegionType"] = df["RegionType_y"]

# Compute annual rent yield
df["AnnualRent"] = df["MonthlyRent"] * 12
df["RentYield"] = (df["AnnualRent"] / df["HomeValue"]) * 100

# Clean
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["HomeValue", "MonthlyRent", "RentYield", "Date"])

# Focus on recent years so the plot is readable
df = df[df["Date"] >= "2016-01-01"].copy()

# Add year column
df["Year"] = df["Date"].dt.year.astype(str)

# Optional: keep only year-end observations to reduce overplotting
df["Month"] = df["Date"].dt.month
df = df[df["Month"] == 12].copy()

# Market type labels
df["MarketType"] = df.apply(
    lambda row: classify_market(row["HomeValue"], row["RentYield"]),
    axis=1
)

# Nice city label
df["CityLabel"] = df["RegionName"] + ", " + df["State"]

# Build chart
fig = px.scatter(
    df,
    x="HomeValue",
    y="RentYield",
    color="MarketType",
    animation_frame="Year",
    hover_name="CityLabel",
    hover_data={
        "HomeValue": ":,.0f",
        "MonthlyRent": ":,.0f",
        "RentYield": ":.2f",
        "MarketType": True,
        "Year": True
    },
    title="Home Prices and Rent Yield Across U.S. Cities",
    labels={
        "HomeValue": "Typical Home Value ($)",
        "RentYield": "Rent Yield (%)"
    }
)

fig.update_layout(
    template="plotly_white",
    xaxis_tickformat="$,.0f",
    yaxis_ticksuffix="%",
    legend_title_text="Market Type"
)

fig.write_html(OUTPUT_FILE, include_plotlyjs="cdn")
print(f"Saved interactive chart to: {OUTPUT_FILE}")