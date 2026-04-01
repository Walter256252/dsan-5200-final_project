import pandas as pd
from pathlib import Path
import plotly.graph_objects as go

# =========================================================
# Paths
# =========================================================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = BASE_DIR / "images"

PRICES_FILE = DATA_DIR / "zhvi.csv"
RENTS_FILE = DATA_DIR / "zori.csv"
OUTPUT_FILE = IMAGES_DIR / "housing_map.html"

START_DATE = pd.Timestamp("2016-01-01")


# =========================================================
# Helpers
# =========================================================
def detect_date_columns(df):
    date_cols = []
    skip_cols = {
        "RegionID", "SizeRank", "RegionName", "RegionType", "StateName",
        "State", "Metro", "CountyName", "City", "Month"
    }

    for col in df.columns:
        col_str = str(col).strip()
        if col_str in skip_cols:
            continue
        try:
            parsed = pd.to_datetime(col_str, errors="raise")
            if pd.Timestamp("1990-01-01") <= parsed <= pd.Timestamp.today() + pd.DateOffset(years=1):
                date_cols.append(col)
        except Exception:
            continue

    return date_cols


def reshape(df, value_name):
    id_cols = [c for c in ["RegionName", "State"] if c in df.columns]
    if "RegionName" not in id_cols or "State" not in id_cols:
        raise ValueError(f"{value_name} file must contain 'RegionName' and 'State' columns.")

    date_cols = detect_date_columns(df)
    if not date_cols:
        raise ValueError(f"No date columns found in {value_name} dataset.")

    df_long = df[id_cols + date_cols].melt(
        id_vars=id_cols,
        var_name="Date",
        value_name=value_name
    )

    df_long["Date"] = pd.to_datetime(df_long["Date"], errors="coerce")
    df_long[value_name] = pd.to_numeric(df_long[value_name], errors="coerce")
    df_long = df_long.dropna(subset=["Date", value_name]).copy()

    # Normalize to month start
    df_long["Date"] = df_long["Date"].dt.to_period("M").dt.to_timestamp()

    return df_long


# =========================================================
# Load and clean data
# =========================================================
prices = pd.read_csv(PRICES_FILE)
rents = pd.read_csv(RENTS_FILE)

prices_long = reshape(prices, "Price")
rents_long = reshape(rents, "Rent")

df = pd.merge(
    prices_long,
    rents_long,
    on=["RegionName", "State", "Date"],
    how="outer"
)

df = df.sort_values(["RegionName", "State", "Date"]).copy()

# Forward fill within each city/state
df["Price"] = df.groupby(["RegionName", "State"])["Price"].transform(lambda x: x.ffill())
df["Rent"] = df.groupby(["RegionName", "State"])["Rent"].transform(lambda x: x.ffill())

# Keep desired time window
df = df[df["Date"] >= START_DATE].copy()

# Remove unusable rows
df = df.dropna(subset=["Price", "Rent"]).copy()

# Create city label
df["City"] = df["RegionName"].astype(str).str.strip() + ", " + df["State"].astype(str).str.strip()

# Keep only the cities you want in the project
project_cities = [
    "Washington, DC",
    "Chicago, IL",
    "Tacoma, WA",
    "Columbus, GA",
    "Augusta, GA",
    "Grovetown, GA",
    "Austin, TX",
    "Phoenix, AZ",
]

df = df[df["City"].isin(project_cities)].copy()

# Rent yield
df["Yield"] = (df["Rent"] * 12 / df["Price"]) * 100
df = df.replace([float("inf"), -float("inf")], pd.NA)
df = df.dropna(subset=["Price", "Rent", "Yield"]).copy()

if df.empty:
    raise ValueError("No valid rows remain after filtering. Check your city names and data.")

# =========================================================
# Latest snapshot
# =========================================================
latest_date = df["Date"].max()
latest = df[df["Date"] == latest_date].copy()

# If some cities are missing at the latest date, keep their most recent available row
latest_fallback = (
    df.sort_values("Date")
      .groupby("City", as_index=False)
      .tail(1)
      .copy()
)

latest = latest_fallback

# =========================================================
# Coordinates for project cities
# =========================================================
city_coords = {
    "Washington, DC": {"lat": 38.9072, "lon": -77.0369},
    "Chicago, IL": {"lat": 41.8781, "lon": -87.6298},
    "Tacoma, WA": {"lat": 47.2529, "lon": -122.4443},
    "Columbus, GA": {"lat": 32.460976, "lon": -84.987709},
    "Augusta, GA": {"lat": 33.4735, "lon": -82.0105},
    "Grovetown, GA": {"lat": 33.4504, "lon": -82.1988},
    "Austin, TX": {"lat": 30.2672, "lon": -97.7431},
    "Phoenix, AZ": {"lat": 33.4484, "lon": -112.0740},
}

latest["Latitude"] = latest["City"].map(lambda x: city_coords.get(x, {}).get("lat"))
latest["Longitude"] = latest["City"].map(lambda x: city_coords.get(x, {}).get("lon"))

map_df = latest.dropna(subset=["Latitude", "Longitude"]).copy()

if map_df.empty:
    raise ValueError("No mapped cities found. Check that city names match the coordinate dictionary.")

# =========================================================
# Bubble size scaling
# =========================================================
# Make marker sizes readable while still reflecting home price
min_size = 18
max_size = 45

price_min = map_df["Price"].min()
price_max = map_df["Price"].max()

if price_max == price_min:
    map_df["MarkerSize"] = 30
else:
    map_df["MarkerSize"] = min_size + (
        (map_df["Price"] - price_min) / (price_max - price_min)
    ) * (max_size - min_size)

# =========================================================
# Build map with explicit hover data
# =========================================================
fig = go.Figure()

fig.add_trace(
    go.Scattermapbox(
        lat=map_df["Latitude"],
        lon=map_df["Longitude"],
        mode="markers+text",
        text=map_df["City"],
        textposition="top right",
        marker=dict(
            size=map_df["MarkerSize"],
            color=map_df["Yield"],
            colorscale="Plasma",
            showscale=True,
            colorbar=dict(title="Yield"),
            opacity=0.80
        ),
        customdata=map_df[["City", "Price", "Rent", "Yield", "Date"]].to_numpy(),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Date: %{customdata[4]|%Y-%m-%d}<br>"
            "Home Price: $%{customdata[1]:,.0f}<br>"
            "Monthly Rent: $%{customdata[2]:,.0f}<br>"
            "Rent Yield: %{customdata[3]:.2f}%<extra></extra>"
        ),
        name="Cities"
    )
)

fig.update_layout(
    title="Housing Prices and Rent Yield Across Cities",
    mapbox=dict(
        style="carto-positron",
        zoom=3,
        center=dict(lat=39.5, lon=-98.35)
    ),
    margin=dict(l=20, r=20, t=60, b=20),
    height=700,
    template="plotly_white"
)

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
fig.write_html(OUTPUT_FILE, include_plotlyjs="cdn", full_html=True)

print("Mapped cities:")
print(sorted(map_df["City"].unique()))
print(f"Saved map to: {OUTPUT_FILE}")