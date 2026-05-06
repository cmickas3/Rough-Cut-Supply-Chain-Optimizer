from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO

import cvxpy as cp
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScenarioData:
    demand_scenarios: dict[str, dict]
    capacity_scenarios: dict[str, dict]


def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def _parse_efficiency(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.rstrip("%").astype(float)
    values.loc[values > 1] /= 100
    return values


def load_scenarios(demand_csv: str | BinaryIO, capacity_csv: str | BinaryIO) -> ScenarioData:
    ddf = _clean_cols(pd.read_csv(demand_csv))
    cdf = _clean_cols(pd.read_csv(capacity_csv))

    demand_scenarios = {}
    for scenario, sdf in ddf.groupby("scenario"):
        sdf = sdf.copy().reset_index(drop=True)
        demand = sdf["demand"].astype(float).to_numpy()
        target_basis_demand = np.roll(demand, -1)
        target_basis_demand[-1] = demand[-1]
        target_woh = sdf["target_woh"].astype(float).to_numpy()

        if not sdf["inventory_0"].notna().any():
            raise ValueError(f"No inventory_0 found for demand scenario: {scenario}")

        demand_scenarios[scenario] = {
            "quarters": sdf["quarter"].tolist(),
            "demand": demand,
            "target_basis_demand": target_basis_demand,
            "target_inventory": target_basis_demand * target_woh / 13,
            "nit_lb_pct": sdf["nit_lb_pct"].astype(float).to_numpy(),
            "nit_ub_pct": sdf["nit_ub_pct"].astype(float).to_numpy(),
            "inventory_0": float(sdf["inventory_0"].dropna().iloc[0]),
            "non_fgi_inventory_0": (
                float(sdf["non_fgi_inventory_0"].dropna().iloc[0])
                if "non_fgi_inventory_0" in sdf and sdf["non_fgi_inventory_0"].notna().any()
                else 0.0
            ),
            "raw": sdf,
        }

    cdf["efficiency"] = _parse_efficiency(cdf["efficiency"])

    capacity_scenarios = {}
    for scenario, sdf in cdf.groupby("scenario"):
        sdf = sdf.copy().reset_index(drop=True)
        quarters = list(dict.fromkeys(sdf["quarter"]))
        sites = list(dict.fromkeys(sdf["site"]))

        sdf["capacity_units"] = (
            sdf["minutes_per_tech"]
            * sdf["max_number_of_techs"]
            * sdf["efficiency"]
            / sdf["time_standard"]
        )

        capacity_scenarios[scenario] = {
            "quarters": quarters,
            "sites": sites,
            "capacity": _pivot(sdf, quarters, sites, "capacity_units"),
            "max_hc": _pivot(sdf, quarters, sites, "max_number_of_techs"),
            "min_share": _pivot(sdf, quarters, sites, "min_share").fillna(0.0),
            "current_hc": (
                sdf.dropna(subset=["current_hc"])
                .drop_duplicates("site")
                .set_index("site")["current_hc"]
                .reindex(sites)
                .fillna(0.0)
                .astype(float)
            ),
            "raw": sdf,
        }

    return ScenarioData(demand_scenarios, capacity_scenarios)


def _pivot(
    df: pd.DataFrame,
    quarters: list[str],
    sites: list[str],
    value: str,
) -> pd.DataFrame:
    return (
        df.pivot(index="quarter", columns="site", values=value)
        .reindex(index=quarters, columns=sites)
        .astype(float)
    )


def solve_supply_plan_hc_stability(
    demand_dict: dict,
    capacity_dict: dict,
    epsilon: float = 1.0,
    smoothing_weight: float = 1.0,
    solver: str = "OSQP",
    verbose: bool = False,
    normalize_objectives: bool = True,
    lead_time_weeks: float = 0.0,
    assume_opening_pipeline: bool = True,
) -> dict:
    quarters = list(demand_dict["quarters"])
    demand = np.asarray(demand_dict["demand"], dtype=float)
    target_inventory = np.asarray(demand_dict["target_inventory"], dtype=float)
    nit_lb_pct = np.asarray(demand_dict["nit_lb_pct"], dtype=float)
    nit_ub_pct = np.asarray(demand_dict["nit_ub_pct"], dtype=float)
    inventory_0 = float(demand_dict["inventory_0"])
    non_fgi_inventory_0 = float(demand_dict.get("non_fgi_inventory_0", 0.0))
    t_count = len(quarters)

    sites = list(capacity_dict["sites"])
    s_count = len(sites)

    if quarters != list(capacity_dict["quarters"]):
        raise ValueError(
            "Demand and capacity quarters do not match. "
            f"Demand quarters: {quarters}; capacity quarters: {capacity_dict['quarters']}"
        )

    max_hc = capacity_dict["max_hc"].reindex(index=quarters, columns=sites).astype(float)
    min_share = (
        capacity_dict["min_share"]
        .reindex(index=quarters, columns=sites)
        .fillna(0.0)
        .astype(float)
    )
    current_hc = capacity_dict["current_hc"].reindex(sites).fillna(0.0).astype(float)

    raw = _clean_cols(capacity_dict["raw"])
    raw["efficiency"] = _parse_efficiency(raw["efficiency"])
    raw["units_per_hc"] = (
        raw["minutes_per_tech"] * raw["efficiency"] / raw["time_standard"]
    )
    units_per_hc = _pivot(raw, quarters, sites, "units_per_hc")
    units_per_hc_vals = units_per_hc.to_numpy().T
    max_hc_vals = max_hc.to_numpy().T

    headcount = cp.Variable((s_count, t_count), nonneg=True)
    production = cp.Variable((s_count, t_count), nonneg=True)
    inventory = cp.Variable(t_count)
    total_production = cp.sum(production, axis=0)

    lead_time_weeks = float(np.clip(lead_time_weeks, 0.0, 13.0))
    same_quarter_fgi_fraction = (13.0 - lead_time_weeks) / 13.0
    next_quarter_fgi_fraction = lead_time_weeks / 13.0
    fgi_production = []
    for t in range(t_count):
        fgi_from_current = same_quarter_fgi_fraction * total_production[t]
        if t > 0:
            fgi_from_prior = next_quarter_fgi_fraction * total_production[t - 1]
        elif non_fgi_inventory_0 > 0:
            fgi_from_prior = non_fgi_inventory_0
        elif assume_opening_pipeline:
            fgi_from_prior = next_quarter_fgi_fraction * total_production[t]
        else:
            fgi_from_prior = 0.0
        fgi_production.append(fgi_from_current + fgi_from_prior)

    constraints = [inventory[0] == inventory_0 + fgi_production[0] - demand[0]]
    for t in range(1, t_count):
        constraints.append(
            inventory[t] == inventory[t - 1] + fgi_production[t] - demand[t]
        )

    nit_lb = nit_lb_pct * target_inventory
    nit_ub = nit_ub_pct * target_inventory
    constraints.extend([inventory >= nit_lb, inventory <= nit_ub])
    constraints.append(headcount <= max_hc_vals)
    constraints.append(production == cp.multiply(units_per_hc_vals, headcount))

    for s, site in enumerate(sites):
        constraints.append(
            production[s, :] >= cp.multiply(min_share[site].to_numpy(), total_production)
        )

    initial_hc = current_hc.to_numpy().reshape(s_count, 1)
    full_hc = cp.hstack([initial_hc, headcount])
    diff_matrix = np.zeros((t_count, t_count + 1))
    for t in range(t_count):
        diff_matrix[t, t] = -1
        diff_matrix[t, t + 1] = 1

    hc_diffs = full_hc @ diff_matrix.T
    stability_objective = cp.sum_squares(hc_diffs)
    nit_centering_objective = cp.sum_squares(inventory - target_inventory)

    stability_scale = float(np.sum(np.maximum(max_hc_vals, 1.0) ** 2))
    nit_scale = float(np.sum(np.maximum(target_inventory, 1.0) ** 2))

    if normalize_objectives:
        stability_term = stability_objective / stability_scale
        nit_term = nit_centering_objective / nit_scale
    else:
        stability_term = stability_objective
        nit_term = nit_centering_objective

    problem = cp.Problem(
        cp.Minimize(
            smoothing_weight * stability_term + epsilon * nit_term
        ),
        constraints,
    )
    problem.solve(solver=solver, verbose=verbose)

    if problem.status not in {"optimal", "optimal_inaccurate"}:
        raise RuntimeError(f"Solve failed with status: {problem.status}")

    production_by_site = pd.DataFrame(production.value.T, index=quarters, columns=sites)
    headcount_by_site = pd.DataFrame(headcount.value.T, index=quarters, columns=sites)
    max_capacity_units = units_per_hc * max_hc
    total_production_value = np.asarray(total_production.value).ravel()
    fgi_production_value = np.array(
        [
            same_quarter_fgi_fraction * total_production_value[t]
            + (
                next_quarter_fgi_fraction * total_production_value[t - 1]
                if t > 0
                else (
                    non_fgi_inventory_0
                    if non_fgi_inventory_0 > 0
                    else (
                        next_quarter_fgi_fraction * total_production_value[t]
                        if assume_opening_pipeline
                        else 0.0
                    )
                )
            )
            for t in range(t_count)
        ]
    )
    non_fgi_inventory_value = next_quarter_fgi_fraction * total_production_value

    summary = pd.DataFrame(index=quarters)
    summary["Quarter"] = quarters
    summary["Demand"] = demand
    summary["Target_Inventory"] = target_inventory
    summary["Inventory_LB"] = nit_lb
    summary["Inventory_UB"] = nit_ub
    summary["Total_Production"] = total_production_value
    summary["FGI_Production"] = fgi_production_value
    summary["Non_FGI_Inventory"] = non_fgi_inventory_value
    summary["Ending_Inventory"] = np.asarray(inventory.value).ravel()
    summary["NIT_Deviation"] = summary["Ending_Inventory"] - summary["Target_Inventory"]
    summary["NIT_Abs_Deviation"] = summary["NIT_Deviation"].abs()
    summary["Total_HC"] = headcount_by_site.sum(axis=1)
    summary["Total_Max_HC"] = max_hc.sum(axis=1)
    summary["Total_Max_Capacity_Units"] = max_capacity_units.sum(axis=1)
    summary["Capacity_Utilization"] = (
        summary["Total_Production"] / summary["Total_Max_Capacity_Units"]
    )

    stability_score = float(stability_objective.value)
    nit_centering_score = float(nit_centering_objective.value)
    normalized_stability_score = stability_score / stability_scale
    normalized_nit_centering_score = nit_centering_score / nit_scale

    return {
        "status": problem.status,
        "objective_value": problem.value,
        "epsilon": epsilon,
        "normalize_objectives": normalize_objectives,
        "lead_time_weeks": lead_time_weeks,
        "assume_opening_pipeline": assume_opening_pipeline,
        "non_fgi_inventory_0": non_fgi_inventory_0,
        "same_quarter_fgi_fraction": same_quarter_fgi_fraction,
        "next_quarter_fgi_fraction": next_quarter_fgi_fraction,
        "stability_score": stability_score,
        "nit_centering_score": nit_centering_score,
        "normalized_stability_score": normalized_stability_score,
        "normalized_nit_centering_score": normalized_nit_centering_score,
        "weighted_stability_score": float(
            smoothing_weight
            * (
                normalized_stability_score
                if normalize_objectives
                else stability_score
            )
        ),
        "weighted_nit_score": float(
            epsilon
            * (
                normalized_nit_centering_score
                if normalize_objectives
                else nit_centering_score
            )
        ),
        "stability_scale": stability_scale,
        "nit_scale": nit_scale,
        "summary": summary,
        "production_by_site": production_by_site,
        "headcount_by_site": headcount_by_site,
        "current_hc": current_hc,
        "units_per_hc": units_per_hc,
        "max_hc": max_hc,
        "max_capacity_units": max_capacity_units,
        "min_share": min_share,
    }


def run_epsilon_sweep(
    demand_dict: dict,
    capacity_dict: dict,
    epsilons: list[float],
    smoothing_weight: float = 1.0,
    solver: str = "OSQP",
    normalize_objectives: bool = True,
    lead_time_weeks: float = 0.0,
    assume_opening_pipeline: bool = True,
) -> tuple[dict[float, dict], pd.DataFrame]:
    results = {}
    rows = []

    for epsilon in epsilons:
        result = solve_supply_plan_hc_stability(
            demand_dict=demand_dict,
            capacity_dict=capacity_dict,
            epsilon=epsilon,
            smoothing_weight=smoothing_weight,
            solver=solver,
            normalize_objectives=normalize_objectives,
            lead_time_weeks=lead_time_weeks,
            assume_opening_pipeline=assume_opening_pipeline,
        )
        results[epsilon] = result
        rows.append(
            {
                "epsilon": epsilon,
                "objective_value": result["objective_value"],
                "stability_score": result["stability_score"],
                "nit_centering_score": result["nit_centering_score"],
                "normalized_stability_score": result["normalized_stability_score"],
                "normalized_nit_centering_score": result[
                    "normalized_nit_centering_score"
                ],
                "weighted_stability_score": result["weighted_stability_score"],
                "weighted_nit_score": result["weighted_nit_score"],
                "avg_abs_nit_deviation": result["summary"]["NIT_Abs_Deviation"].mean(),
                "max_abs_nit_deviation": result["summary"]["NIT_Abs_Deviation"].max(),
                "ending_nit_deviation": result["summary"]["NIT_Deviation"].iloc[-1],
            }
        )

    return results, pd.DataFrame(rows)
