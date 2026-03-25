import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Load cleaned dataset
df = pd.read_csv("data/final_dataset.csv")
df["date"] = pd.to_datetime(df["date"])

# Filter DC
dc = df[(df["RegionName"] == "Washington") & (df["StateName"] == "DC")].copy()

# Sort by date
dc = dc.sort_values("date")

# Dollar formatter for y-axis
def dollar_formatter(x, pos):
    return f"${x:,.0f}"

# Home prices chart
plt.figure(figsize=(10, 5))
plt.plot(dc["date"], dc["home_price"])
plt.title("Washington, DC Home Prices Since 2016")
plt.xlabel("Date")
plt.ylabel("Typical Home Value")
plt.gca().yaxis.set_major_formatter(FuncFormatter(dollar_formatter))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("images/dc_home_prices.png", bbox_inches="tight")
plt.close()

# Rent chart
plt.figure(figsize=(10, 5))
plt.plot(dc["date"], dc["rent"])
plt.title("Washington, DC Rents Since 2016")
plt.xlabel("Date")
plt.ylabel("Typical Monthly Rent")
plt.gca().yaxis.set_major_formatter(FuncFormatter(dollar_formatter))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("images/dc_rents.png", bbox_inches="tight")
plt.close()

print("Saved intro charts to images/")