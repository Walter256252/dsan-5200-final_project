import pandas as pd
import matplotlib.pyplot as plt



# Load datasets

zhvi = pd.read_csv("data/zhvi.csv")
zori = pd.read_csv("data/zori.csv")
rates = pd.read_csv("data/mortgage_rates.csv")

print("ZHVI shape:", zhvi.shape)
print("ZORI shape:", zori.shape)
print("Rates shape:", rates.shape)

print("\nZHVI columns:")
print(zhvi.columns.tolist()[:20])

print("\nZORI columns:")
print(zori.columns.tolist()[:20])



# Helper function to reshape Zillow data
def melt_zillow(df, value_name):
    
    possible_id_vars = [
        "RegionID",
        "SizeRank",
        "RegionName",
        "RegionType",
        "StateName",
        "State",
        "City",
        "Metro",
        "CountyName"
    ]

    id_vars = [col for col in possible_id_vars if col in df.columns]

    # Everything else should be date columns
    value_vars = [col for col in df.columns if col not in id_vars]

    df_long = df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="date",
        value_name=value_name
    )

    df_long["date"] = pd.to_datetime(df_long["date"], format="%Y-%m-%d", errors="coerce")
    df_long = df_long.dropna(subset=["date"])

    return df_long



# Reshape Zillow data

zhvi_long = melt_zillow(zhvi, "home_price")
zori_long = melt_zillow(zori, "rent")

print("\nZHVI long sample:")
print(zhvi_long.head())

print("\nZORI long sample:")
print(zori_long.head())



# Filter cities

# ----------------------------
# 4. Filter target city-state pairs
# ----------------------------
target_places = [
    ("Washington", "DC"),
    ("Augusta", "GA"),
    ("Columbus", "GA"),
    ("Tacoma", "WA"),
    ("Austin", "TX"),
    ("Phoenix", "AZ"),
    ("Chicago", "IL")
]

target_df = pd.DataFrame(target_places, columns=["RegionName", "StateName"])

zhvi_long = zhvi_long.merge(target_df, on=["RegionName", "StateName"], how="inner")
zori_long = zori_long.merge(target_df, on=["RegionName", "StateName"], how="inner")

print("\nFiltered ZHVI city-state pairs:")
print(zhvi_long[["RegionName", "StateName"]].drop_duplicates().sort_values(["StateName", "RegionName"]))

print("\nFiltered ZORI city-state pairs:")
print(zori_long[["RegionName", "StateName"]].drop_duplicates().sort_values(["StateName", "RegionName"]))



# Filter date range

zhvi_long = zhvi_long[zhvi_long["date"] >= "2016-01-01"]
zori_long = zori_long[zori_long["date"] >= "2016-01-01"]



#  Merge prices + rent

merge_cols = [col for col in ["RegionName", "StateName", "date"] if col in zhvi_long.columns and col in zori_long.columns]

df = pd.merge(
    zhvi_long,
    zori_long,
    on=merge_cols,
    how="inner",
    suffixes=("_price", "_rent")
)

print("\nMerged shape:", df.shape)
print(df.head())



# Clean mortgage rates
print("\nMortgage rate columns:")
print(rates.columns.tolist())

# Rename columns if needed
if len(rates.columns) == 2:
    rates.columns = ["date", "mortgage_rate"]

rates["date"] = pd.to_datetime(rates["date"], errors="coerce")
rates["mortgage_rate"] = pd.to_numeric(rates["mortgage_rate"], errors="coerce")

df = pd.merge(df, rates, on="date", how="left")



# Create derived metrics

df["annual_rent"] = df["rent"] * 12
df["rent_yield"] = df["annual_rent"] / df["home_price"]

df = df.sort_values(["RegionName", "date"])
df["price_growth"] = df.groupby("RegionName")["home_price"].pct_change()
df["rent_growth"] = df.groupby("RegionName")["rent"].pct_change()



# Save final dataset
df.to_csv("data/final_dataset.csv", index=False)
print("\nFinal dataset saved to data/final_dataset.csv")



# Quick test plot for DC

dc = df[df["RegionName"] == "Washington"]

if not dc.empty:
    plt.figure(figsize=(10, 5))
    plt.plot(dc["date"], dc["home_price"])
    plt.title("Washington, DC Home Prices Over Time")
    plt.xlabel("Date")
    plt.ylabel("Home Price")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("images/dc_prices.png")
    print("Saved test plot to images/dc_prices.png")
else:
    print("No Washington rows found in merged dataset.")