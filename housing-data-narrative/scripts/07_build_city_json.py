import pandas as pd
import json

prices = pd.read_csv("data/zhvi.csv")
rents = pd.read_csv("data/zori.csv")

def reshape(df, val):
    id_cols = ["RegionName", "State"]
    date_cols = [c for c in df.columns if "-" in c]

    df = df[id_cols + date_cols].melt(
        id_vars=id_cols,
        var_name="Date",
        value_name=val
    )

    df["Date"] = pd.to_datetime(df["Date"])
    return df

p = reshape(prices, "Price")
r = reshape(rents, "Rent")

df = pd.merge(p, r, on=["RegionName", "State", "Date"])
df = df[df["Date"] >= "2016-01-01"]

df["City"] = df["RegionName"] + ", " + df["State"]

output = {}

for city in df["City"].unique():
    sub = df[df["City"] == city].sort_values("Date")

    output[city] = {
        "dates": sub["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "prices": sub["Price"].tolist(),
        "rents": sub["Rent"].tolist()
    }

with open("images/city_timeseries.json", "w") as f:
    json.dump(output, f)

print("Saved city_timeseries.json")