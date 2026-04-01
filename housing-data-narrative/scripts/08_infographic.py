import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = BASE_DIR / "images"

PRICES_FILE = DATA_DIR / "zhvi.csv"
RENTS_FILE = DATA_DIR / "zori.csv"
OUTPUT_FILE = IMAGES_DIR / "market_types_infographic.png"


def detect_date_cols(df):
    cols = []
    for c in df.columns:
        try:
            pd.to_datetime(c)
            cols.append(c)
        except:
            continue
    return cols


def reshape(df, value_name):
    id_cols = ["RegionName", "State"]
    date_cols = detect_date_cols(df)

    df = df[id_cols + date_cols].melt(
        id_vars=id_cols,
        var_name="Date",
        value_name=value_name
    )

    df["Date"] = pd.to_datetime(df["Date"])
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    return df.dropna()


# Load + reshape
prices = reshape(pd.read_csv(PRICES_FILE), "Price")
rents = reshape(pd.read_csv(RENTS_FILE), "Rent")

df = pd.merge(prices, rents, on=["RegionName", "State", "Date"])
df = df[df["Date"] == df["Date"].max()].copy()

# Compute yield
df["Yield"] = (df["Rent"] * 12 / df["Price"]) * 100

# Create label
df["City"] = df["RegionName"] + ", " + df["State"]

# Select representative cities (you can tweak these)
targets = {
    "High-cost / Low-yield": "Washington, DC",
    "Middle market": "Chicago, IL",
    "Lower-cost / High-yield": "Columbus, GA"
}

selected = []

for label, city in targets.items():
    row = df[df["City"] == city]
    if not row.empty:
        r = row.iloc[0]
        selected.append({
            "type": label,
            "city": city,
            "price": r["Price"],
            "rent": r["Rent"],
            "yield": r["Yield"]
        })

# Plot infographic
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle("Three Types of Housing Markets", fontsize=18, fontweight="bold")

colors = ["#d73027", "#4575b4", "#1a9850"]

for i, (ax, item) in enumerate(zip(axes, selected)):
    ax.axis("off")

    ax.text(0.5, 0.85, item["type"], ha="center", fontsize=14, fontweight="bold", color=colors[i])
    ax.text(0.5, 0.7, item["city"], ha="center", fontsize=12)

    ax.text(0.5, 0.55, f"Home Value\n${item['price']:,.0f}", ha="center", fontsize=11)
    ax.text(0.5, 0.4, f"Monthly Rent\n${item['rent']:,.0f}", ha="center", fontsize=11)
    ax.text(0.5, 0.25, f"Rent Yield\n{item['yield']:.1f}%", ha="center", fontsize=11)

    # takeaway text
    if i == 0:
        msg = "High price, weaker income returns"
    elif i == 1:
        msg = "Balanced price and income"
    else:
        msg = "Lower price, stronger cash flow"

    ax.text(0.5, 0.1, msg, ha="center", fontsize=10, style="italic")

plt.tight_layout()
IMAGES_DIR.mkdir(exist_ok=True)
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved infographic to: {OUTPUT_FILE}")

