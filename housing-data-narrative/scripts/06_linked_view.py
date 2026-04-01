import json
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================================================
# Paths
# =========================================================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = BASE_DIR / "images"

PRICES_FILE = DATA_DIR / "zhvi.csv"
RENTS_FILE = DATA_DIR / "zori.csv"
OUTPUT_FILE = IMAGES_DIR / "linked_view.html"

START_DATE = pd.Timestamp("2016-01-01")
END_DATE = pd.Timestamp.today().normalize()


# =========================================================
# Helpers
# =========================================================
def detect_date_columns(df):
    """
    Return only columns that truly look like date columns.
    This avoids accidentally pulling in non-date columns.
    """
    date_cols = []
    for col in df.columns:
        col_str = str(col).strip()

        # Skip known id/meta columns
        if col_str in {
            "RegionID", "SizeRank", "RegionName", "RegionType", "StateName",
            "State", "Metro", "CountyName", "City", "Month"
        }:
            continue

        try:
            parsed = pd.to_datetime(col_str, errors="raise")
            # Keep only plausible modern housing time series dates
            if pd.Timestamp("1990-01-01") <= parsed <= END_DATE + pd.DateOffset(years=1):
                date_cols.append(col)
        except Exception:
            continue

    return date_cols


def reshape(df, value_name):
    """
    Convert wide Zillow-style data into long format.
    """
    possible_id_cols = ["RegionName", "State"]
    id_cols = [c for c in possible_id_cols if c in df.columns]

    if "RegionName" not in id_cols or "State" not in id_cols:
        raise ValueError(f"{value_name} file must contain 'RegionName' and 'State' columns.")

    date_cols = detect_date_columns(df)

    if not date_cols:
        raise ValueError(f"No date columns detected in {value_name} dataset.")

    df_long = df[id_cols + date_cols].melt(
        id_vars=id_cols,
        var_name="Date",
        value_name=value_name
    )

    df_long["Date"] = pd.to_datetime(df_long["Date"], errors="coerce")
    df_long[value_name] = pd.to_numeric(df_long[value_name], errors="coerce")

    df_long = df_long.dropna(subset=["Date", value_name]).copy()
    return df_long


def prepare_city_timeseries(sub):
    """
    For one city:
    - sort dates
    - set monthly frequency from 2016 to present
    - forward fill missing values
    - return clean monthly series
    """
    sub = sub.sort_values("Date").copy()
    sub = sub[(sub["Date"] >= START_DATE) & (sub["Date"] <= END_DATE)].copy()

    if sub.empty:
        return sub

    # Use month start for a clean monthly sequence
    sub["Date"] = sub["Date"].dt.to_period("M").dt.to_timestamp(how="start")
    sub = sub.drop_duplicates(subset=["Date"], keep="last")
    sub = sub.set_index("Date")

    monthly_index = pd.date_range(start=START_DATE, end=END_DATE, freq="MS")
    sub = sub.reindex(monthly_index)

    # Fill city metadata
    if "RegionName" in sub.columns:
        sub["RegionName"] = sub["RegionName"].ffill().bfill()
    if "State" in sub.columns:
        sub["State"] = sub["State"].ffill().bfill()
    if "City" in sub.columns:
        sub["City"] = sub["City"].ffill().bfill()

    # Fill numeric values
    if "Price" in sub.columns:
        sub["Price"] = sub["Price"].ffill()
    if "Rent" in sub.columns:
        sub["Rent"] = sub["Rent"].ffill()

    sub = sub.reset_index().rename(columns={"index": "Date"})
    sub = sub.dropna(subset=["Price", "Rent"], how="all")

    return sub


# =========================================================
# Load data
# =========================================================
prices = pd.read_csv(PRICES_FILE)
rents = pd.read_csv(RENTS_FILE)

prices_long = reshape(prices, "Price")
rents_long = reshape(rents, "Rent")

# Normalize to month start to reduce mismatches before merge
prices_long["Date"] = prices_long["Date"].dt.to_period("M").dt.to_timestamp(how="start")
rents_long["Date"] = rents_long["Date"].dt.to_period("M").dt.to_timestamp(how="start")

# Outer merge is more robust than inner merge
df = pd.merge(
    prices_long,
    rents_long,
    on=["RegionName", "State", "Date"],
    how="outer"
)

# Sort and fill within each city
df = df.sort_values(["RegionName", "State", "Date"]).copy()
df["Price"] = df.groupby(["RegionName", "State"])["Price"].transform(lambda x: x.ffill())
df["Rent"] = df.groupby(["RegionName", "State"])["Rent"].transform(lambda x: x.ffill())

# Keep only the range you want
df = df[(df["Date"] >= START_DATE) & (df["Date"] <= END_DATE)].copy()

# Remove rows that still have no usable values
df = df.dropna(subset=["Price", "Rent"], how="all").copy()

# Create city label
df["City"] = df["RegionName"].astype(str).str.strip() + ", " + df["State"].astype(str).str.strip()

# Yield calculation
df["Yield"] = (df["Rent"] * 12 / df["Price"]) * 100
df = df.replace([float("inf"), -float("inf")], pd.NA)
df = df.dropna(subset=["Price", "Rent", "Yield"])

# =========================================================
# Build cleaned monthly city dataset
# =========================================================
city_dataset = {}
clean_city_frames = []

for city in sorted(df["City"].dropna().unique()):
    sub = df[df["City"] == city].copy()
    sub = prepare_city_timeseries(sub)

    if sub.empty:
        continue

    # Recompute yield after monthly cleanup
    sub["Yield"] = (sub["Rent"] * 12 / sub["Price"]) * 100
    sub = sub.replace([float("inf"), -float("inf")], pd.NA)
    sub = sub.dropna(subset=["Price", "Rent", "Yield"])

    if sub.empty:
        continue

    clean_city_frames.append(sub)

    city_dataset[city] = {
        "dates": sub["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "prices": sub["Price"].round(2).tolist(),
        "rents": sub["Rent"].round(2).tolist()
    }

if not clean_city_frames:
    raise ValueError("No city data available after cleaning.")

df_clean = pd.concat(clean_city_frames, ignore_index=True)

# Latest point for left scatter
latest_date = df_clean["Date"].max()
latest = df_clean[df_clean["Date"] == latest_date].copy()

if latest.empty:
    raise ValueError("No latest data available for scatter plot.")

# Default city for right panel
default_city = sorted(city_dataset.keys())[0]
default_data = city_dataset[default_city]
city_dataset_json = json.dumps(city_dataset)

# =========================================================
# Build figure
# =========================================================
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=(
        "City Comparison (Click a Point)",
        f"City Trends Over Time: {default_city}"
    ),
    horizontal_spacing=0.12,
    specs=[[{"secondary_y": False}, {"secondary_y": True}]]
)

# Left scatter
fig.add_trace(
    go.Scatter(
        x=latest["Price"],
        y=latest["Yield"],
        mode="markers",
        marker=dict(size=9),
        name="Cities",
        customdata=latest["City"],
        text=latest["City"],
        hovertemplate=(
            "<b>%{customdata}</b><br>"
            "Home Price: $%{x:,.0f}<br>"
            "Rent Yield: %{y:.2f}%<extra></extra>"
        )
    ),
    row=1,
    col=1
)

# Right panel - Home Price
fig.add_trace(
    go.Scatter(
        x=default_data["dates"],
        y=default_data["prices"],
        mode="lines",
        name="Home Price",
        hovertemplate="Date: %{x}<br>Home Price: $%{y:,.0f}<extra></extra>"
    ),
    row=1,
    col=2,
    secondary_y=False
)

# Right panel - Monthly Rent
fig.add_trace(
    go.Scatter(
        x=default_data["dates"],
        y=default_data["rents"],
        mode="lines",
        name="Monthly Rent",
        hovertemplate="Date: %{x}<br>Monthly Rent: $%{y:,.0f}<extra></extra>"
    ),
    row=1,
    col=2,
    secondary_y=True
)

# Axes
fig.update_xaxes(title_text="Home Price ($)", row=1, col=1)
fig.update_yaxes(title_text="Rent Yield (%)", row=1, col=1)

fig.update_xaxes(
    title_text="Date",
    range=[START_DATE.strftime("%Y-%m-%d"), END_DATE.strftime("%Y-%m-%d")],
    row=1,
    col=2
)
fig.update_yaxes(title_text="Home Price ($)", row=1, col=2, secondary_y=False)
fig.update_yaxes(title_text="Monthly Rent ($)", row=1, col=2, secondary_y=True)

# Layout
fig.update_layout(
    title="Housing Market Comparison and Trends",
    template="plotly_white",
    height=700,
    showlegend=True
)

# =========================================================
# Export HTML with reliable div id
# =========================================================
html_str = fig.to_html(
    include_plotlyjs="cdn",
    full_html=True,
    div_id="housing_plot"
)

# =========================================================
# Custom JS for linked interaction
# =========================================================
custom_js = f"""
<script>
window.addEventListener("load", function() {{
    const plot = document.getElementById("housing_plot");
    const dataset = {city_dataset_json};

    if (!plot) {{
        console.error("Plotly div not found.");
        return;
    }}

    plot.on("plotly_click", function(data) {{
        if (!data.points || !data.points.length) return;

        const point = data.points[0];
        const city = String(point.customdata || "").trim();

        // Only respond to clicks on the left scatter
        if (!city || !(city in dataset)) {{
            return;
        }}

        const cityData = dataset[city];

        if (!cityData || !cityData.dates || !cityData.prices || !cityData.rents) {{
            console.error("Incomplete data for city:", city);
            return;
        }}

        Plotly.restyle(plot, {{
            x: [cityData.dates],
            y: [cityData.prices]
        }}, [1]);

        Plotly.restyle(plot, {{
            x: [cityData.dates],
            y: [cityData.rents]
        }}, [2]);

        Plotly.relayout(plot, {{
            "annotations[1].text": "City Trends Over Time: " + city,
            "xaxis2.range": ["{START_DATE.strftime("%Y-%m-%d")}", "{END_DATE.strftime("%Y-%m-%d")}"]
        }});
    }});
}});
</script>
"""

html_str = html_str.replace("</body>", custom_js + "</body>")

# Ensure output folder exists
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(html_str)

print(f"Saved linked view to: {OUTPUT_FILE}")