import pandas as pd
import plotly.graph_objects as go

# ----------------------------
# Load and prepare data
# ----------------------------
df = pd.read_csv("data/final_dataset.csv")
df["date"] = pd.to_datetime(df["date"])

df["city"] = df["RegionName"] + ", " + df["StateName"]
df["rent_yield_pct"] = df["rent_yield"] * 100

plot_df = df[["date", "city", "rent_yield_pct", "home_price", "rent"]].copy()
plot_df = plot_df.sort_values(["city", "date"])

city_order = [
    "Washington, DC",
    "Augusta, GA",
    "Columbus, GA",
    "Tacoma, WA",
    "Austin, TX",
    "Phoenix, AZ",
    "Chicago, IL",
]

available_cities = [c for c in city_order if c in plot_df["city"].unique()]

# ----------------------------
# Styling
# ----------------------------
dc_color = "#d62728"      # strong red
other_color = "#b8b8b8"   # muted gray

fig = go.Figure()

# ----------------------------
# Add traces
# ----------------------------
for city in available_cities:
    city_df = plot_df[plot_df["city"] == city].copy()

    is_dc = city == "Washington, DC"

    fig.add_trace(
        go.Scatter(
            x=city_df["date"],
            y=city_df["rent_yield_pct"],
            mode="lines",
            name=city,
            line=dict(
                color=dc_color if is_dc else other_color,
                width=4 if is_dc else 2,
            ),
            opacity=1.0 if is_dc else 0.45,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Date: %{x|%b %Y}<br>"
                "Rent Yield: %{y:.2f}%<br>"
                "Typical Home Value: $%{customdata[1]:,.0f}<br>"
                "Typical Monthly Rent: $%{customdata[2]:,.0f}"
                "<extra></extra>"
            ),
            customdata=city_df[["city", "home_price", "rent"]].to_numpy(),
            visible=True,
        )
    )

# ----------------------------
# Dropdown buttons
# ----------------------------
buttons = []

# Show all cities
buttons.append(
    dict(
        label="All cities",
        method="update",
        args=[
            {"visible": [True] * len(available_cities)},
            {
                "title": "Rent Yield Comparison Across Cities (2016–Present)",
                "showlegend": True,
            },
        ],
    )
)

# Show one city at a time
for city in available_cities:
    visible = [c == city for c in available_cities]
    buttons.append(
        dict(
            label=city,
            method="update",
            args=[
                {"visible": visible},
                {
                    "title": f"Rent Yield Over Time: {city}",
                    "showlegend": False,
                },
            ],
        )
    )

# ----------------------------
# Layout
# ----------------------------
fig.update_layout(
    title="Rent Yield Comparison Across Cities (2016–Present)",
    template="plotly_white",
    hovermode="x unified",
    width=1100,
    height=650,
    margin=dict(l=60, r=40, t=90, b=60),
    xaxis=dict(
        title="Date",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
    ),
    yaxis=dict(
        title="Rent Yield (%)",
        ticksuffix="%",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
    ),
    legend=dict(
        title="City",
        orientation="v",
        x=1.02,
        y=1,
        xanchor="left",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.85)",
    ),
    updatemenus=[
        dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=0.01,
            xanchor="left",
            y=1.16,
            yanchor="top",
            bgcolor="white",
            bordercolor="lightgray",
            borderwidth=1,
        )
    ],
    annotations=[
        dict(
            text="View:",
            x=0.0,
            xref="paper",
            y=1.145,
            yref="paper",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(size=12),
        )
    ],
)

# ----------------------------
# Save output
# ----------------------------
fig.write_html("images/rent_yield_interactive.html", include_plotlyjs="cdn")

print("Saved polished interactive chart to images/rent_yield_interactive.html")