from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from supply_optimizer import load_scenarios, run_epsilon_sweep
from supply_optimizer import solve_supply_plan_hc_stability


WORKSPACE = Path(__file__).parent
DEFAULT_DEMAND_CSV = WORKSPACE / "Project Mock Data - Demand and NIT (2).csv"
DEFAULT_CAPACITY_CSV = WORKSPACE / "Project Mock Data - Capacity (8).csv"

COLORS = {
    "blue": "#5B8DEF",
    "blue_fill": "rgba(91, 141, 239, 0.16)",
    "green": "#58B99D",
    "purple": "#9B7FEA",
    "lavender": "#C7B8F5",
    "teal": "#6CCBD1",
    "slate": "#667085",
    "grid": "#E5E7EB",
}
SITE_COLORS = ["#5B8DEF", "#58B99D", "#9B7FEA", "#6CCBD1", "#C7B8F5"]
INVENTORY_WEIGHT_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
DEFAULT_INVENTORY_WEIGHT = 2


st.set_page_config(
    page_title="Rough Cut Supply Plan Optimizer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --app-font: "Avenir Next", Avenir, "Helvetica Neue", Arial, sans-serif;
        --app-blue-25: #f6fbff;
        --app-blue-50: #edf7ff;
        --app-blue-100: #d7ecff;
        --app-blue-600: #2563a8;
        --app-blue-700: #1d4f86;
        --app-blue-900: #102f52;
        --app-border: #b9d7f2;
    }

    .stApp {
        background: var(--app-blue-25);
        color: var(--app-blue-900);
        font-family: var(--app-font);
    }

    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp h5,
    .stApp h6,
    .stApp p,
    .stApp label,
    .stApp input,
    .stApp textarea,
    .stApp select,
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stMetricLabel"],
    .stApp [data-testid="stMetricValue"],
    .stApp [data-baseweb="tab"] {
        font-family: var(--app-font);
    }

    .stApp span[class*="material"],
    .stApp i[class*="material"],
    .stApp [data-testid="stIconMaterial"] {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
        font-weight: normal;
        font-style: normal;
        line-height: 1;
        text-transform: none;
        letter-spacing: normal;
        white-space: nowrap;
        word-wrap: normal;
        direction: ltr;
    }

    header[data-testid="stHeader"] {
        background: var(--app-blue-25);
        border-bottom: 1px solid var(--app-border);
    }

    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"] {
        color: var(--app-blue-700);
    }

    [data-testid="stSidebar"] {
        background: var(--app-blue-50);
        border-right: 1px solid var(--app-border);
    }

    h1, h2, h3, h4, h5, h6,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {
        color: var(--app-blue-900);
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--app-border);
        border-radius: 8px;
        padding: 12px 14px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        background: var(--app-blue-50);
        border: 1px solid var(--app-border);
        border-radius: 10px 10px 0 0;
        color: var(--app-blue-700);
        min-width: 96px;
        padding: 10px 18px;
    }

    .stTabs [data-baseweb="tab"] p {
        font-size: 1rem;
        line-height: 1.2;
        white-space: nowrap;
    }

    .stTabs [aria-selected="true"] {
        background: white;
        color: var(--app-blue-900);
        border-bottom-color: white;
        font-weight: 650;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"],
    div[data-baseweb="textarea"] > div {
        background: white;
        border-color: var(--app-border);
        color: var(--app-blue-900);
    }

    div[data-baseweb="select"] span,
    div[data-baseweb="select"] svg,
    div[data-baseweb="input"] input,
    div[data-baseweb="base-input"] input {
        color: var(--app-blue-900);
        fill: var(--app-blue-900);
    }

    [data-testid="stFileUploader"] section {
        background: white;
        border: 1px solid var(--app-border);
        color: var(--app-blue-900);
    }

    [data-testid="stFileUploader"] section button {
        background: var(--app-blue-600);
        color: white;
        border-color: var(--app-blue-600);
    }

    [data-testid="stFileUploader"] section button p,
    [data-testid="stFileUploader"] section button span,
    [data-testid="stFileUploader"] section button svg {
        color: white;
        fill: white;
    }

    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span {
        color: var(--app-blue-700);
    }

    [data-testid="stSegmentedControl"] label,
    [data-testid="stSegmentedControl"] label div,
    [data-testid="stSegmentedControl"] label p {
        background: white;
        border-color: var(--app-border);
        color: var(--app-blue-900);
    }

    [data-testid="stSegmentedControl"] label[aria-checked="true"],
    [data-testid="stSegmentedControl"] label[aria-checked="true"] div,
    [data-testid="stSegmentedControl"] label[aria-checked="true"] p,
    [data-testid="stSegmentedControl"] label:has(input:checked),
    [data-testid="stSegmentedControl"] label:has(input:checked) div,
    [data-testid="stSegmentedControl"] label:has(input:checked) p {
        background: var(--app-blue-100);
        border-color: var(--app-blue-600);
        color: var(--app-blue-900);
    }

    [data-testid="stNumberInput"] div[data-baseweb="input"],
    [data-testid="stNumberInput"] div[data-baseweb="base-input"] {
        background: white;
        border-color: var(--app-border);
        color: var(--app-blue-900);
    }

    [data-testid="stNumberInput"] button {
        background: var(--app-blue-50);
        border-color: var(--app-border);
        color: var(--app-blue-700);
    }

    [data-testid="stNumberInput"] button:hover {
        background: var(--app-blue-100);
        color: var(--app-blue-900);
    }

    [data-testid="stNumberInput"] button p,
    [data-testid="stNumberInput"] button span,
    [data-testid="stNumberInput"] button svg {
        color: var(--app-blue-700);
        fill: var(--app-blue-700);
    }

    .stButton > button,
    [data-testid="stBaseButton-primary"],
    [data-testid="stBaseButton-secondary"] {
        background: var(--app-blue-600);
        border-color: var(--app-blue-600);
        color: white;
    }

    .stButton > button:hover,
    [data-testid="stBaseButton-primary"]:hover,
    [data-testid="stBaseButton-secondary"]:hover {
        background: var(--app-blue-700);
        border-color: var(--app-blue-700);
        color: white;
    }

    [data-testid="stDataFrame"],
    [data-testid="stDataEditor"] {
        background: white;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def soften_chart(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"color": "#111827"},
        legend={
            "bgcolor": "rgba(255,255,255,0)",
            "font": {"color": "#111827"},
            "title": {"font": {"color": "#111827"}},
            "orientation": "h",
            "x": 0,
            "xanchor": "left",
            "y": -0.24,
            "yanchor": "top",
        },
        xaxis={
            "gridcolor": COLORS["grid"],
            "zerolinecolor": COLORS["grid"],
            "tickfont": {"color": "#111827"},
            "title": {"font": {"color": "#111827"}},
        },
        yaxis={
            "gridcolor": COLORS["grid"],
            "zerolinecolor": COLORS["grid"],
            "tickfont": {"color": "#111827"},
            "title": {"font": {"color": "#111827"}},
        },
    )
    fig.update_xaxes(tickfont={"color": "#111827"}, title={"font": {"color": "#111827"}})
    fig.update_yaxes(tickfont={"color": "#111827"}, title={"font": {"color": "#111827"}})
    return fig


def format_number(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if pd.isna(number):
        return ""
    if abs(number - round(number)) < 1e-9:
        return f"{number:,.0f}"
    return f"{number:,.1f}"


def format_integer(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if pd.isna(number):
        return ""
    return f"{number:,.0f}"


def format_epsilon(value) -> str:
    return f"{float(value):.4f}"


def inventory_weight_label(epsilon: float) -> int:
    distances = [abs(float(epsilon) - value) for value in INVENTORY_WEIGHT_VALUES]
    return int(np.argmin(distances))


def display_df(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    for column in formatted.select_dtypes(include="number").columns:
        if "epsilon" in column.lower():
            formatted[column] = formatted[column].map(format_epsilon)
        else:
            formatted[column] = formatted[column].map(format_number)
    return formatted


def display_integer_df(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    for column in formatted.select_dtypes(include="number").columns:
        formatted[column] = formatted[column].map(format_integer)
    return formatted


def scenario_context_key(demand_name: str, capacity_name: str) -> str:
    return f"{demand_name}__{capacity_name}".replace(" ", "_")


def demand_input_table(demand_dict: dict) -> pd.DataFrame:
    raw = demand_dict["raw"].copy()
    first_quarter = raw["quarter"].iloc[0]
    return pd.DataFrame(
        [
            {
                "metric": "Starting FGI inventory",
                first_quarter: float(demand_dict["inventory_0"]),
            },
            {
                "metric": "Starting non-FGI inventory",
                first_quarter: float(demand_dict.get("non_fgi_inventory_0", 0.0)),
            },
        ]
    )


def capacity_input_table(capacity_dict: dict) -> pd.DataFrame:
    raw = capacity_dict["raw"].copy().sort_values(["site", "quarter"])
    metrics = [
        ("Max HC", "max_number_of_techs"),
        ("Minutes per tech", "minutes_per_tech"),
        ("Efficiency", "efficiency"),
        ("Time standard", "time_standard"),
        ("Minimum share", "min_share"),
    ]
    rows = []
    for site, site_df in raw.groupby("site", sort=True):
        site_df = site_df.set_index("quarter")
        for label, column in metrics:
            row = {"site": site, "metric": label}
            for quarter in capacity_dict["quarters"]:
                row[quarter] = float(site_df.loc[quarter, column])
            rows.append(row)

        row = {"site": site, "metric": "Starting HC"}
        for quarter in capacity_dict["quarters"]:
            row[quarter] = None
        row[capacity_dict["quarters"][0]] = float(capacity_dict["current_hc"].loc[site])
        rows.append(row)

    return pd.DataFrame(rows)


def editable_input_tables(
    demand_dict: dict,
    capacity_dict: dict,
    key: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    demand_key = f"demand_inputs_{key}"
    capacity_key = f"capacity_inputs_{key}"
    if demand_key not in st.session_state:
        st.session_state[demand_key] = demand_input_table(demand_dict)
    if capacity_key not in st.session_state:
        st.session_state[capacity_key] = capacity_input_table(capacity_dict)

    st.subheader("Demand inputs")
    demand_table = st.data_editor(
        st.session_state[demand_key],
        key=f"demand_editor_{key}",
        width="stretch",
        hide_index=True,
        column_config={"metric": st.column_config.TextColumn("Metric", disabled=True)},
    )
    st.session_state[demand_key] = demand_table

    st.subheader("Capacity inputs")
    capacity_table = st.data_editor(
        st.session_state[capacity_key],
        key=f"capacity_editor_{key}",
        width="stretch",
        hide_index=True,
        column_config={
            "site": st.column_config.TextColumn("Site", disabled=True),
            "metric": st.column_config.TextColumn("Metric", disabled=True),
        },
    )
    st.session_state[capacity_key] = capacity_table.sort_values(["site", "metric"])

    return demand_table, capacity_table


def editable_demand_inputs(demand_dict: dict, key: str) -> pd.DataFrame:
    demand_key = f"demand_inputs_{key}"
    fresh_defaults = demand_input_table(demand_dict)
    if demand_key not in st.session_state:
        st.session_state[demand_key] = fresh_defaults

    demand_table = st.session_state[demand_key].copy()
    if "Starting non-FGI inventory" not in set(demand_table["metric"]):
        demand_table = pd.concat(
            [
                demand_table,
                fresh_defaults[
                    fresh_defaults["metric"] == "Starting non-FGI inventory"
                ],
            ],
            ignore_index=True,
        )
    st.session_state[demand_key] = demand_table

    st.subheader("Demand inputs")
    demand_table = st.data_editor(
        st.session_state[demand_key],
        key=f"demand_editor_{key}",
        width="stretch",
        hide_index=True,
        column_config={"metric": st.column_config.TextColumn("Metric", disabled=True)},
    )
    st.session_state[demand_key] = demand_table
    return demand_table


def editable_capacity_inputs(capacity_dict: dict, key: str) -> pd.DataFrame:
    capacity_key = f"capacity_inputs_{key}"
    if capacity_key not in st.session_state:
        st.session_state[capacity_key] = capacity_input_table(capacity_dict)

    st.subheader("Capacity inputs")
    capacity_table = st.data_editor(
        st.session_state[capacity_key],
        key=f"capacity_editor_{key}",
        width="stretch",
        hide_index=True,
        column_config={
            "site": st.column_config.TextColumn("Site", disabled=True),
            "metric": st.column_config.TextColumn("Metric", disabled=True),
        },
    )
    st.session_state[capacity_key] = capacity_table.sort_values(["site", "metric"])
    return capacity_table


def apply_demand_inputs(demand_dict: dict, demand_table: pd.DataFrame) -> dict:
    adjusted = demand_dict.copy()
    quarters = list(demand_dict["quarters"])
    fgi_inventory_row = demand_table.loc[
        demand_table["metric"] == "Starting FGI inventory"
    ].iloc[0]
    non_fgi_inventory_row = demand_table.loc[
        demand_table["metric"] == "Starting non-FGI inventory"
    ].iloc[0]

    demand = np.asarray(demand_dict["demand"], dtype=float)
    target_basis_demand = pd.Series(demand).shift(-1).fillna(demand[-1]).to_numpy()
    raw = demand_dict["raw"].copy()
    raw.loc[raw.index[0], "inventory_0"] = float(fgi_inventory_row[quarters[0]])
    raw.loc[raw.index[0], "non_fgi_inventory_0"] = float(
        non_fgi_inventory_row[quarters[0]]
    )

    target_woh = raw["target_woh"].astype(float).to_numpy()
    adjusted["demand"] = demand
    adjusted["target_basis_demand"] = target_basis_demand
    adjusted["target_inventory"] = target_basis_demand * target_woh / 13
    adjusted["inventory_0"] = float(fgi_inventory_row[quarters[0]])
    adjusted["non_fgi_inventory_0"] = float(non_fgi_inventory_row[quarters[0]])
    adjusted["raw"] = raw
    return adjusted


def apply_capacity_inputs(capacity_dict: dict, capacity_table: pd.DataFrame) -> dict:
    adjusted = capacity_dict.copy()
    quarters = list(capacity_dict["quarters"])
    sites = list(capacity_dict["sites"])
    table = capacity_table.copy()

    metric_to_column = {
        "Max HC": "max_number_of_techs",
        "Minutes per tech": "minutes_per_tech",
        "Efficiency": "efficiency",
        "Time standard": "time_standard",
        "Minimum share": "min_share",
    }
    raw = capacity_dict["raw"].copy()
    for site in sites:
        for metric, column in metric_to_column.items():
            row = table[(table["site"] == site) & (table["metric"] == metric)].iloc[0]
            for quarter in quarters:
                raw.loc[(raw["site"] == site) & (raw["quarter"] == quarter), column] = float(
                    row[quarter]
                )

    current_hc = capacity_dict["current_hc"].copy()
    for site in sites:
        row = table[(table["site"] == site) & (table["metric"] == "Starting HC")].iloc[0]
        current_hc.loc[site] = float(row[quarters[0]])
        raw.loc[
            (raw["site"] == site) & (raw["quarter"] == quarters[0]),
            "current_hc",
        ] = current_hc.loc[site]

    max_hc = (
        raw.pivot(index="quarter", columns="site", values="max_number_of_techs")
        .reindex(index=quarters, columns=sites)
        .astype(float)
    )
    min_share = (
        raw.pivot(index="quarter", columns="site", values="min_share")
        .reindex(index=quarters, columns=sites)
        .fillna(0.0)
        .astype(float)
    )
    efficiency = raw["efficiency"].astype(float)
    raw["capacity_units"] = (
        raw["minutes_per_tech"].astype(float)
        * raw["max_number_of_techs"].astype(float)
        * efficiency
        / raw["time_standard"].astype(float)
    )
    capacity = (
        raw.pivot(index="quarter", columns="site", values="capacity_units")
        .reindex(index=quarters, columns=sites)
        .astype(float)
    )

    adjusted["raw"] = raw
    adjusted["max_hc"] = max_hc
    adjusted["min_share"] = min_share
    adjusted["capacity"] = capacity
    adjusted["current_hc"] = current_hc
    return adjusted


def inventory_chart(summary: pd.DataFrame) -> go.Figure:
    x = summary["Quarter"]
    hover = "%{x}<br>%{fullData.name}: %{y:,.1f}<extra></extra>"
    fig = go.Figure()
    fig.add_bar(
        x=x,
        y=summary["Ending_Inventory"],
        name="Ending FGI Inventory",
        marker_color=COLORS["lavender"],
        hovertemplate=hover,
    )
    fig.add_bar(
        x=x,
        y=summary["Non_FGI_Inventory"],
        name="Ending Non-FGI Inventory",
        marker_color=COLORS["teal"],
        opacity=0.72,
        hovertemplate=hover,
    )
    fig.add_scatter(
        x=x,
        y=summary["Demand"],
        mode="lines+markers",
        name="Demand",
        line={"color": COLORS["slate"]},
        marker={"color": COLORS["slate"]},
        hovertemplate=hover,
    )
    fig.add_scatter(
        x=x,
        y=summary["Total_Production"],
        mode="lines+markers",
        name="Started Build Qty",
        line={"color": COLORS["green"]},
        marker={"color": COLORS["green"]},
        hovertemplate=hover,
    )
    fig.add_scatter(
        x=x,
        y=summary["FGI_Production"],
        mode="lines+markers",
        name="FGI Production",
        line={"color": COLORS["blue"]},
        marker={"color": COLORS["blue"]},
        hovertemplate=hover,
    )
    fig.add_scatter(
        x=x,
        y=summary["Target_Inventory"],
        mode="lines+markers",
        line={"dash": "dash", "color": COLORS["purple"]},
        marker={"color": COLORS["purple"]},
        name="NIT",
        hovertemplate=hover,
    )
    fig.add_scatter(
        x=x,
        y=summary["Inventory_UB"],
        mode="lines",
        line={"width": 0, "color": COLORS["blue"]},
        showlegend=False,
        hoverinfo="skip",
    )
    fig.add_scatter(
        x=x,
        y=summary["Inventory_LB"],
        mode="lines",
        fill="tonexty",
        fillcolor=COLORS["blue_fill"],
        line={"width": 0, "color": COLORS["blue"]},
        name="NIT Band",
        hovertemplate=hover,
    )
    fig.update_layout(
        height=420,
        margin={"l": 12, "r": 12, "t": 24, "b": 94},
        yaxis_title="Units",
        hovermode="x unified",
        barmode="stack",
    )
    return soften_chart(fig)


def nit_settings_chart(settings: pd.DataFrame) -> go.Figure:
    hover = "%{x}<br>%{fullData.name}: %{y:,.1f}<extra></extra>"
    fig = go.Figure()
    fig.add_scatter(
        x=settings["quarter"],
        y=settings["inventory_ub"],
        mode="lines",
        line={"width": 0, "color": COLORS["blue"]},
        showlegend=False,
        hoverinfo="skip",
    )
    fig.add_scatter(
        x=settings["quarter"],
        y=settings["inventory_lb"],
        mode="lines",
        fill="tonexty",
        fillcolor=COLORS["blue_fill"],
        line={"width": 0, "color": COLORS["blue"]},
        name="NIT Band",
        hovertemplate=hover,
    )
    fig.add_scatter(
        x=settings["quarter"],
        y=settings["target_inventory"],
        mode="lines+markers",
        line={"dash": "dash", "color": COLORS["purple"]},
        marker={"color": COLORS["purple"]},
        name="NIT Target",
        hovertemplate=hover,
    )
    fig.add_scatter(
        x=settings["quarter"],
        y=settings["demand"],
        mode="lines+markers",
        line={"color": COLORS["green"]},
        marker={"color": COLORS["green"]},
        name="Demand",
        hovertemplate=hover,
    )
    fig.add_scatter(
        x=settings["quarter"],
        y=settings["inventory_basis_demand"],
        mode="lines+markers",
        line={"dash": "dot", "color": COLORS["slate"]},
        marker={"color": COLORS["slate"]},
        name="EOH Basis Demand",
        hovertemplate=hover,
    )
    fig.update_layout(
        height=340,
        margin={"l": 12, "r": 12, "t": 24, "b": 88},
        yaxis_title="Units",
        hovermode="x unified",
    )
    fig.update_yaxes(rangemode="tozero")
    return soften_chart(fig)


def headcount_chart(result: dict, capacity_dict: dict) -> go.Figure:
    hc = result["headcount_by_site"]
    max_hc = result["max_hc"]
    current_hc = result.get("current_hc", capacity_dict["current_hc"])
    hc_with_q0 = pd.concat(
        [
            current_hc.to_frame().T.rename(index={current_hc.name: "Q0"}),
            hc,
        ]
    )
    max_hc_with_q0 = pd.concat(
        [
            pd.DataFrame([[None] * len(max_hc.columns)], index=["Q0"], columns=max_hc.columns),
            max_hc,
        ]
    )
    fig = go.Figure()
    hover = "%{x}<br>%{fullData.name}: %{y:,.1f}<extra></extra>"
    for i, site in enumerate(hc_with_q0.columns):
        color = SITE_COLORS[i % len(SITE_COLORS)]
        fig.add_scatter(
            x=hc_with_q0.index,
            y=hc_with_q0[site],
            mode="lines+markers",
            name=f"{site} HC",
            line={"color": color},
            marker={"color": color},
            hovertemplate=hover,
        )
        fig.add_scatter(
            x=max_hc_with_q0.index,
            y=max_hc_with_q0[site],
            mode="lines",
            line={"dash": "dot", "color": color},
            name=f"{site} Max",
            hovertemplate=hover,
        )
    fig.update_layout(
        height=420,
        margin={"l": 12, "r": 12, "t": 24, "b": 104},
        yaxis_title="Headcount",
        hovermode="x unified",
    )
    y_max = max(hc_with_q0.to_numpy().max(), max_hc.to_numpy().max())
    fig.update_yaxes(range=[0, y_max * 1.12 if y_max else 1])
    return soften_chart(fig)


def tradeoff_chart(tradeoff_df: pd.DataFrame) -> go.Figure:
    x_col = (
        "normalized_stability_score"
        if "normalized_stability_score" in tradeoff_df
        else "stability_score"
    )
    y_col = (
        "normalized_nit_centering_score"
        if "normalized_nit_centering_score" in tradeoff_df
        else "nit_centering_score"
    )
    fig = go.Figure()
    fig.add_scatter(
        x=tradeoff_df[x_col],
        y=tradeoff_df[y_col],
        mode="lines+markers+text",
        text=[
            f"{int(value)}"
            for value in tradeoff_df.get(
                "inventory_weight",
                tradeoff_df["epsilon"].map(inventory_weight_label),
            )
        ],
        textposition="top center",
        name="Inventory weight sweep",
        line={"color": COLORS["purple"]},
        marker={"color": COLORS["blue"], "size": 9},
        textfont={"color": COLORS["slate"]},
        hovertemplate=(
            "Inventory weight: %{text}<br>"
            "Stability: %{x:,.1f}<br>"
            "NIT centering: %{y:,.1f}<extra></extra>"
        ),
    )
    fig.update_layout(
        height=390,
        margin={"l": 12, "r": 12, "t": 24, "b": 82},
        xaxis_title="Normalized Stability Score",
        yaxis_title="Normalized NIT Centering Score",
        hovermode="closest",
    )
    x_max = tradeoff_df[x_col].max()
    fig.update_xaxes(range=[0, x_max * 1.12 if x_max else 1])
    return soften_chart(fig)


@st.cache_data(show_spinner=False)
def load_from_paths(demand_file, capacity_file, demand_mtime_ns: int, capacity_mtime_ns: int):
    return load_scenarios(demand_file, capacity_file)

def feasibility_status(
    demand_dict: dict,
    capacity_dict: dict,
    solver: str,
    lead_time_weeks: int,
    assume_opening_pipeline: bool,
) -> tuple[bool, str]:
    try:
        result = solve_supply_plan_hc_stability(
            demand_dict=demand_dict,
            capacity_dict=capacity_dict,
            epsilon=0.1,
            smoothing_weight=1.0,
            solver=solver,
            lead_time_weeks=lead_time_weeks,
            assume_opening_pipeline=assume_opening_pipeline,
        )
    except Exception as exc:
        return False, str(exc)
    return True, result["status"]


def nit_rows_from_demand(demand_dict: dict) -> list[dict]:
    raw = demand_dict["raw"].copy()
    demand = raw["demand"].astype(float)
    inventory_basis_demand = pd.Series(
        demand_dict.get("target_basis_demand", demand),
        index=raw.index,
    ).astype(float)
    target_weeks = (
        raw["target_woh"].astype(float)
        if "target_woh" in raw
        else demand_dict["target_inventory"] / inventory_basis_demand * 13
    )
    lower_weeks = target_weeks * raw["nit_lb_pct"].astype(float)
    upper_weeks = target_weeks * raw["nit_ub_pct"].astype(float)

    return [
        {
            "quarter": str(quarter),
            "demand": float(demand_value),
            "inventory_basis_demand": float(basis_demand_value),
            "low_inventory_weeks": float(lower_value),
            "nit_inventory_weeks": float(target_value),
            "max_inventory_weeks": float(upper_value),
            "target_weeks": float(target_value),
            "lower_weeks": float(lower_value),
            "upper_weeks": float(upper_value),
        }
        for quarter, demand_value, basis_demand_value, target_value, lower_value, upper_value in zip(
            raw["quarter"],
            demand,
            inventory_basis_demand,
            target_weeks,
            lower_weeks,
            upper_weeks,
            strict=True,
        )
    ]


def editable_nit_settings(demand_dict: dict, key: str) -> pd.DataFrame:
    state_key = f"nit_rows_{key}"
    fresh_defaults = pd.DataFrame(nit_rows_from_demand(demand_dict))
    if state_key not in st.session_state:
        st.session_state[state_key] = fresh_defaults.to_dict("records")

    settings = pd.DataFrame(st.session_state[state_key])
    settings["quarter"] = fresh_defaults["quarter"]
    if "demand" not in settings:
        settings["demand"] = fresh_defaults["demand"]
    settings["inventory_basis_demand"] = (
        settings["demand"].astype(float).shift(-1).fillna(settings["demand"].astype(float).iloc[-1])
    )
    settings = settings[
        [
            "quarter",
            "demand",
            "inventory_basis_demand",
            "low_inventory_weeks",
            "nit_inventory_weeks",
            "max_inventory_weeks",
        ]
    ]

    edited = st.data_editor(
        settings,
        key=f"nit_editor_{key}",
        width="stretch",
        hide_index=True,
        column_config={
            "quarter": st.column_config.TextColumn("Quarter", disabled=True),
            "demand": st.column_config.NumberColumn(
                "Demand",
                min_value=0.0,
                step=1.0,
                format="%.1f",
            ),
            "inventory_basis_demand": st.column_config.NumberColumn(
                "EOH basis demand",
                disabled=True,
                help="Demand used to convert inventory weeks to units. This is next quarter's demand, except the final horizon quarter uses its own demand.",
            ),
            "low_inventory_weeks": st.column_config.NumberColumn(
                "Low inventory weeks",
                min_value=0.0,
                step=0.25,
                format="%.2f",
            ),
            "nit_inventory_weeks": st.column_config.NumberColumn(
                "NIT inventory weeks",
                min_value=0.0,
                step=0.25,
                format="%.2f",
            ),
            "max_inventory_weeks": st.column_config.NumberColumn(
                "Max inventory weeks",
                min_value=0.0,
                step=0.25,
                format="%.2f",
            ),
        },
    )
    edited["low_inventory_weeks"] = edited["low_inventory_weeks"].clip(lower=0)
    edited["demand"] = edited["demand"].astype(float).clip(lower=0)
    edited["inventory_basis_demand"] = (
        edited["demand"].shift(-1).fillna(edited["demand"].iloc[-1])
    )
    edited["nit_inventory_weeks"] = edited[
        ["nit_inventory_weeks", "low_inventory_weeks"]
    ].max(axis=1)
    edited["max_inventory_weeks"] = edited[
        ["max_inventory_weeks", "nit_inventory_weeks"]
    ].max(axis=1)

    st.session_state[state_key] = edited.to_dict("records")

    return nit_settings_from_rows(st.session_state[state_key])


def nit_settings_from_rows(rows: list[dict]) -> pd.DataFrame:
    settings = pd.DataFrame(rows)
    if "low_inventory_weeks" not in settings:
        settings["low_inventory_weeks"] = settings["lower_weeks"]
        settings["nit_inventory_weeks"] = settings["target_weeks"]
        settings["max_inventory_weeks"] = settings["upper_weeks"]
    if "inventory_basis_demand" not in settings:
        settings["inventory_basis_demand"] = settings["demand"]

    settings["low_inventory_weeks"] = settings["low_inventory_weeks"].clip(lower=0)
    settings["nit_inventory_weeks"] = settings[
        ["nit_inventory_weeks", "low_inventory_weeks"]
    ].max(axis=1)
    settings["max_inventory_weeks"] = settings[
        ["max_inventory_weeks", "nit_inventory_weeks"]
    ].max(axis=1)
    settings["lower_weeks"] = settings["low_inventory_weeks"]
    settings["target_weeks"] = settings["nit_inventory_weeks"]
    settings["upper_weeks"] = settings["max_inventory_weeks"]
    settings["target_inventory"] = (
        settings["inventory_basis_demand"] * settings["nit_inventory_weeks"] / 13
    )
    settings["inventory_lb"] = (
        settings["inventory_basis_demand"] * settings["low_inventory_weeks"] / 13
    )
    settings["inventory_ub"] = (
        settings["inventory_basis_demand"] * settings["max_inventory_weeks"] / 13
    )
    settings["lower_band"] = (
        settings["low_inventory_weeks"] / settings["nit_inventory_weeks"].replace(0, 1)
    )
    settings["upper_band"] = (
        settings["max_inventory_weeks"] / settings["nit_inventory_weeks"].replace(0, 1)
    )
    return settings


def apply_nit_settings(demand_dict: dict, settings: pd.DataFrame) -> dict:
    adjusted = demand_dict.copy()
    raw = demand_dict["raw"].copy()
    raw["demand"] = settings["demand"].astype(float).to_numpy()
    raw["target_woh"] = settings["target_weeks"].astype(float).to_numpy()
    raw["nit_lb_pct"] = settings["lower_band"].astype(float).to_numpy()
    raw["nit_ub_pct"] = settings["upper_band"].astype(float).to_numpy()

    adjusted["demand"] = raw["demand"].astype(float).to_numpy()
    adjusted["target_basis_demand"] = settings["inventory_basis_demand"].astype(float).to_numpy()
    adjusted["target_inventory"] = settings["target_inventory"].astype(float).to_numpy()
    adjusted["nit_lb_pct"] = raw["nit_lb_pct"].astype(float).to_numpy()
    adjusted["nit_ub_pct"] = raw["nit_ub_pct"].astype(float).to_numpy()
    adjusted["raw"] = raw
    return adjusted


st.title("Rough Cut Supply Plan Optimizer")

epsilons = INVENTORY_WEIGHT_VALUES
smoothing_weight = 1.0
solver = "OSQP"

if not DEFAULT_DEMAND_CSV.exists() or not DEFAULT_CAPACITY_CSV.exists():
    st.error("Default CSV files are missing.")
    st.stop()

demand_mtime_ns = DEFAULT_DEMAND_CSV.stat().st_mtime_ns
capacity_mtime_ns = DEFAULT_CAPACITY_CSV.stat().st_mtime_ns
scenario_data = load_from_paths(
    DEFAULT_DEMAND_CSV,
    DEFAULT_CAPACITY_CSV,
    demand_mtime_ns,
    capacity_mtime_ns,
)
data_source_key = (
    f"default_{DEFAULT_DEMAND_CSV.name}_{demand_mtime_ns}_"
    f"{DEFAULT_CAPACITY_CSV.name}_{capacity_mtime_ns}"
)

demand_names = list(scenario_data.demand_scenarios)
capacity_names = list(scenario_data.capacity_scenarios)
default_demand_index = (
    demand_names.index("seasonal_peak_q4") if "seasonal_peak_q4" in demand_names else 0
)
default_capacity_index = (
    capacity_names.index("two_site") if "two_site" in capacity_names else 0
)

selector_cols = st.columns(2)
with selector_cols[0]:
    demand_name = st.selectbox(
        "Demand scenario",
        demand_names,
        index=default_demand_index,
    )
with selector_cols[1]:
    capacity_name = st.selectbox(
        "Capacity scenario",
        capacity_names,
        index=default_capacity_index,
    )

demand_dict = scenario_data.demand_scenarios[demand_name]
capacity_dict = scenario_data.capacity_scenarios[capacity_name]
input_key = scenario_context_key(f"{data_source_key}_{demand_name}", capacity_name)

tab_overview, tab_details, tab_inputs, tab_about = st.tabs(
    ["Overview", "Details", "Inputs", "About"]
)

with tab_inputs:
    demand_inputs = editable_demand_inputs(demand_dict, input_key)

demand_dict = apply_demand_inputs(demand_dict, demand_inputs)

with tab_inputs:
    st.subheader("Shape NIT targets and bands")
    st.caption(
        "Edit the quarterly week values. The chart converts those weeks to inventory units using demand."
    )
    nit_settings = editable_nit_settings(demand_dict, input_key)
    st.plotly_chart(nit_settings_chart(nit_settings), width="stretch")

demand_dict = apply_nit_settings(demand_dict, nit_settings)

with tab_inputs:
    capacity_inputs = editable_capacity_inputs(capacity_dict, input_key)

capacity_dict = apply_capacity_inputs(capacity_dict, capacity_inputs)

if "lead_time_weeks" not in st.session_state:
    st.session_state.lead_time_weeks = 3
else:
    st.session_state.lead_time_weeks = int(round(st.session_state.lead_time_weeks))
if "assume_opening_pipeline" not in st.session_state:
    st.session_state.assume_opening_pipeline = True

is_feasible, feasibility_message = feasibility_status(
    demand_dict,
    capacity_dict,
    solver,
    st.session_state.lead_time_weeks,
    st.session_state.assume_opening_pipeline,
)

if not is_feasible:
    with tab_overview:
        st.error(f"Infeasible inputs · {feasibility_message}")
    with tab_inputs:
        st.error(f"Feasibility check failed: {feasibility_message}")
        st.caption("Inputs above feed the active scenario solve.")
    st.stop()

try:
    solve_started_at = perf_counter()
    results, tradeoff_df = run_epsilon_sweep(
        demand_dict=demand_dict,
        capacity_dict=capacity_dict,
        epsilons=epsilons,
        smoothing_weight=smoothing_weight,
        solver=solver,
        lead_time_weeks=st.session_state.lead_time_weeks,
        assume_opening_pipeline=st.session_state.assume_opening_pipeline,
    )
    solve_runtime = perf_counter() - solve_started_at
except Exception as exc:
    st.error(str(exc))
    st.stop()

tradeoff_df = tradeoff_df.copy()
tradeoff_df.insert(
    0,
    "inventory_weight",
    tradeoff_df["epsilon"].map(inventory_weight_label),
)

if (
    "selected_inventory_weight" not in st.session_state
    or st.session_state.selected_inventory_weight not in range(len(epsilons))
):
    st.session_state.selected_inventory_weight = DEFAULT_INVENTORY_WEIGHT

selected_epsilon = epsilons[st.session_state.selected_inventory_weight]
result = results[selected_epsilon]
summary = result["summary"]

with tab_overview:
    st.success(f"Feasible inputs · Solver status: {feasibility_message}")
    st.caption(
        f"Last rerun runtime: {solve_runtime:.2f}s · Lead time: {result['lead_time_weeks']:.1f} weeks"
    )

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.subheader("Inventory, demand, and build")
        st.plotly_chart(inventory_chart(summary), width="stretch")
        st.subheader("Inventory Weight")
        st.caption(
            "Higher inventory weight prioritizes tighter NIT adherence; lower inventory weight "
            "allows for increased flexibility within bands."
        )
        st.select_slider(
            "Inventory Weight",
            options=list(range(len(epsilons))),
            key="selected_inventory_weight",
            label_visibility="collapsed",
        )
        st.subheader("Lead time parameters")
        lead_cols = st.columns([1, 1.35])
        with lead_cols[0]:
            st.number_input(
                "Production lead time weeks",
                min_value=0,
                max_value=13,
                step=1,
                key="lead_time_weeks",
                help=(
                    "Production is assumed evenly spread across the quarter. "
                    "With a 3-week lead time, 10/13 of current-quarter production "
                    "counts toward same-quarter FGI inventory."
                ),
            )
        with lead_cols[1]:
            st.toggle(
                "Assume opening lead-time pipeline",
                key="assume_opening_pipeline",
                help=(
                    "When enabled, Q1 starts with a steady-state pipeline from pre-horizon "
                    "production. This avoids artificially starving FGI in the first quarter."
                ),
            )
    with chart_cols[1]:
        st.subheader("Headcount by site")
        st.plotly_chart(headcount_chart(result, capacity_dict), width="stretch")
        st.subheader("Production by site")
        st.dataframe(display_integer_df(result["production_by_site"].T), width="stretch")

    if len(epsilons) > 1:
        st.subheader("Stability vs NIT tradeoff")
        st.plotly_chart(tradeoff_chart(tradeoff_df), width="stretch")

with tab_details:
    st.subheader("Quarter summary")
    st.dataframe(display_df(summary), width="stretch")

    lead_cols = st.columns(3)
    lead_cols[0].metric(
        "Same-quarter FGI fraction",
        f"{result['same_quarter_fgi_fraction']:.1%}",
    )
    lead_cols[1].metric(
        "Next-quarter FGI fraction",
        f"{result['next_quarter_fgi_fraction']:.1%}",
    )
    lead_cols[2].metric(
        "Ending non-FGI inventory",
        format_number(summary["Non_FGI_Inventory"].iloc[-1]),
    )
    st.caption(
        f"Starting FGI inventory: {format_number(result['summary']['Ending_Inventory'].iloc[0] - result['summary']['FGI_Production'].iloc[0] + result['summary']['Demand'].iloc[0])} · "
        f"Starting non-FGI inventory: {format_number(result['non_fgi_inventory_0'])}"
    )

    st.subheader("Applied NIT settings")
    st.dataframe(
        display_df(
            nit_settings[
                [
                "quarter",
                "demand",
                "inventory_basis_demand",
                "low_inventory_weeks",
                "nit_inventory_weeks",
                "max_inventory_weeks",
                    "inventory_lb",
                    "target_inventory",
                    "inventory_ub",
                ]
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    table_cols = st.columns(2)
    with table_cols[0]:
        st.subheader("Production by site")
        st.dataframe(display_integer_df(result["production_by_site"].T), width="stretch")
    with table_cols[1]:
        st.subheader("Headcount by site")
        st.dataframe(display_df(result["headcount_by_site"].T), width="stretch")

    if len(epsilons) > 1:
        st.subheader("Inventory weight sweep")
        st.dataframe(display_df(tradeoff_df.drop(columns=["epsilon"])), width="stretch")

with tab_inputs:
    st.success(f"Feasibility check passed: {feasibility_message}")
    st.caption("Inputs above feed the active scenario solve.")

with tab_about:
    st.subheader("About")
    st.markdown(
        """
        The Rough Cut Supply Plan Optimizer uses convex optimization to quickly
        generate a range of plans that balance labor stability with inventory
        attainment. The tool aims to minimize a weighted sum of quadratic
        penalties for quarter-over-quarter hiring by site, and deviation from
        net inventory targets.

        The tool is designed for rapid scenario modeling of more complex supply
        chains, using aggregated measures to help supply chain leaders select
        directional plans. While existing heuristics-based planning systems are
        highly effective for detailed, constraint-aware network and node-level
        planning, they are not ideal for exploring high-level strategic
        tradeoffs due to their computational complexity and long runtimes.

        The inventory weight control exposes this tradeoff directly. The tool
        solves the same scenario across a fixed range of inventory weights,
        where lower values allow more flexibility within inventory bands and
        higher values prioritize tighter adherence to the NIT target. Because
        each solve runs nearly instantaneously, users can move between these
        alternatives and immediately see how inventory attainment, production
        timing, and labor stability change.

        This model is intended to work in tandem with advanced planning systems:
        it can pull demand, inventory, and capacity inputs, aggregate them,
        allow for targeted overrides, and generate a range of viable scenarios
        while quantifying tradeoffs. The resulting outputs can be used to select
        directional plans and define inputs to feed back into full planning
        models, reducing the need for time-intensive iterative runs of the
        network solver during early-stage planning.

        Since the simplified formulation is convex and runs at a coarse time
        granularity, tools like CVXPY can run scenarios nearly instantaneously.

        This model operates at an aggregated level and does not explicitly
        capture material constraints, multi-level BOM dependencies, or detailed
        routing, and is intended for directional planning rather than executable
        plan generation.
        """
    )
    st.subheader("Model formulation")
    st.latex(
        r"""
        \begin{aligned}
        \min_{H,p,I}\quad
        & \omega \frac{\sum_{s \in S}\sum_{t=1}^{T}
        (H_{s,t} - H_{s,t-1})^2}{\kappa_H}
        + \epsilon \frac{\sum_{t=1}^{T}(I_t - \bar{I}_t)^2}{\kappa_I} \\
        \text{s.t.}\quad
        & P_t = \sum_{s \in S} p_{s,t} && \forall t \\
        & A_t = \theta P_t + \delta P_{t-1} && \forall t \\
        & I_t = I_{t-1} + A_t - D_t && \forall t \\
        & \beta_t \bar{I}_t \le I_t \le \bar{\beta}_t \bar{I}_t && \forall t \\
        & 0 \le H_{s,t} \le H^{max}_{s,t} && \forall s,t \\
        & p_{s,t} = \gamma_{s,t} H_{s,t} && \forall s,t \\
        & p_{s,t} \ge \alpha_{s,t} P_t && \forall s,t \\
        & \theta = \frac{13-L}{13},\quad \delta = \frac{L}{13},\quad
        \gamma_{s,t} = \frac{\text{minutes per tech}_{s,t}\cdot
        \text{efficiency}_{s,t}}{\text{time standard}_{s,t}}
        \end{aligned}
        """
    )
    st.markdown(
        r"""
        **Variable and parameter glossary**

        $S$ is the set of production sites, and $T$ is the planning horizon in
        quarters. $H_{s,t}$ is headcount at site $s$ in quarter $t$, with
        $H_{s,0}$ equal to current starting headcount. $p_{s,t}$ is started
        production at site $s$ in quarter $t$, and $P_t$ is total started
        production across all sites.

        $I_t$ is ending finished goods inventory, $D_t$ is demand, and
        $\bar{I}_t$ is the target net inventory level. The lower and upper
        NIT bands are represented by $\beta_t$ and $\bar{\beta}_t$, so
        ending inventory must remain between $\beta_t\bar{I}_t$ and
        $\bar{\beta}_t\bar{I}_t$.

        $L$ is production lead time in weeks. Since each quarter is modeled as
        13 weeks, $\theta$ is the share of current-quarter starts that convert
        to finished goods in the same quarter, while $\delta$ is the share
        that remains in non-FGI pipeline and converts next quarter. $A_t$ is
        the total finished goods production available in quarter $t$ after
        applying that lead-time logic.

        $\gamma_{s,t}$ converts headcount into production units for each site
        and quarter. $H^{max}_{s,t}$ is maximum headcount, and $\alpha_{s,t}$
        is the minimum production share for a site when applicable. $\omega$
        is the headcount stability weight, $\epsilon$ is the inventory weight,
        and $\kappa_H$ and $\kappa_I$ normalize the two objective terms so the
        tradeoff is more robust to demand and capacity scale.
        """
    )
