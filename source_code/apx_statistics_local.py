"""
Local Overfitting Statistics

Quantifies the "forced sacrifice" of local overfitting:
For each model, finds the per-group optimal configuration (complexity or epoch),
then reports how much R² is lost on other groups at the globally chosen optimum.
"""

import os
import numpy as np
import pandas as pd


def _load_complexity_data(project_path, model_name):
    path = os.path.join(project_path, f"{model_name}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _load_epoch_data(project_path, model_name):
    path = os.path.join(project_path, f"{model_name}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def compute_complexity_sacrifice(project_path, model_name, strategies=None):
    """
    For a model's complexity sweep, find each strategy's optimal complexity
    and report the R² each other strategy achieves at that point.

    Returns DataFrame with columns:
        optimized_for, optimal_param, and R² columns for each strategy.
    """
    if strategies is None:
        strategies = ["Sample", "Site", "Grid"]

    df = _load_complexity_data(project_path, model_name)
    if df is None:
        return None

    param_col = "parameter_number"
    r2_col = "r2_score"

    avg = df.groupby([param_col, "strategy"])[r2_col].mean().reset_index()
    pivot = avg.pivot(index=param_col, columns="strategy", values=r2_col).dropna()

    rows = []
    for target_strat in strategies:
        if target_strat not in pivot.columns:
            continue
        best_param = pivot[target_strat].idxmax()
        row = {"model": model_name, "optimized_for": target_strat, "optimal_param": best_param}
        for s in strategies:
            if s in pivot.columns:
                row[f"R2_{s}"] = pivot.loc[best_param, s]
        rows.append(row)

    return pd.DataFrame(rows)


def compute_epoch_sacrifice(project_path, model_name, strategies=None):
    """
    For a model's epoch sweep, find each strategy's optimal epoch
    and report the R² each other strategy achieves at that point.
    """
    if strategies is None:
        strategies = ["Sample", "Site", "Grid"]

    df = _load_epoch_data(project_path, model_name)
    if df is None:
        return None

    r2_col = "val_r2"
    epoch_col = "epoch"

    avg = df.groupby([epoch_col, "strategy"])[r2_col].mean().reset_index()
    pivot = avg.pivot(index=epoch_col, columns="strategy", values=r2_col).dropna()

    rows = []
    for target_strat in strategies:
        if target_strat not in pivot.columns:
            continue
        best_epoch = pivot[target_strat].idxmax()
        row = {"model": model_name, "optimized_for": target_strat, "optimal_epoch": best_epoch}
        for s in strategies:
            if s in pivot.columns:
                row[f"R2_{s}"] = pivot.loc[best_epoch, s]
        rows.append(row)

    return pd.DataFrame(rows)


def sacrifice_summary(df_sacrifice, strategies=None):
    """
    From a sacrifice table, compute the forced loss:
    For each strategy, the difference between its own optimal R²
    and the R² it gets when another strategy is optimized.

    Returns a summary DataFrame.
    """
    if strategies is None:
        strategies = ["Sample", "Site", "Grid"]

    strategy_display = {"Sample": "Time-wise", "Site": "Site-wise", "Grid": "Region-wise"}

    rows = []
    for _, row in df_sacrifice.iterrows():
        opt_for = row["optimized_for"]
        for s in strategies:
            col = f"R2_{s}"
            if col not in row or s == opt_for:
                continue
            own_opt = df_sacrifice.loc[
                df_sacrifice["optimized_for"] == s, col
            ]
            if own_opt.empty:
                continue
            own_best = own_opt.values[0]
            forced_r2 = row[col]
            loss = own_best - forced_r2
            pct_loss = loss / own_best * 100 if own_best != 0 else 0

            rows.append({
                "model": row["model"],
                "optimized_for": strategy_display.get(opt_for, opt_for),
                "evaluated_on": strategy_display.get(s, s),
                "R2_at_own_optimum": own_best,
                "R2_at_forced_optimum": forced_r2,
                "R2_loss": loss,
                "pct_loss": pct_loss,
            })

    return pd.DataFrame(rows)


def full_local_overfit_report(
    complexity_path, epoch_path,
    baseline_models=None, og_models=None,
    strategies=None
):
    """
    Generate a complete local overfitting report for baseline and OG models.

    Parameters
    ----------
    complexity_path : str
        Path to 5CV_validation_vs_parameter directory.
    epoch_path : str
        Path to epoch_analysis directory.
    baseline_models : list
        e.g. ["mlp", "resnet", "transformer", "lightgbm", "catboost", "xgboost"]
    og_models : list
        e.g. ["og_mlp", "og_resnet", "og_transformer"]
    strategies : list
        e.g. ["Sample", "Site", "Grid"]
    """
    if baseline_models is None:
        baseline_models = ["mlp", "resnet", "transformer", "lightgbm", "catboost", "xgboost"]
    if og_models is None:
        og_models = ["og_mlp", "og_resnet", "og_transformer"]
    if strategies is None:
        strategies = ["Sample", "Site", "Grid"]

    all_results = []

    print("=" * 80)
    print("LOCAL OVERFITTING SACRIFICE REPORT")
    print("=" * 80)

    for dim_label, path, compute_fn, param_key in [
        ("COMPLEXITY", complexity_path, compute_complexity_sacrifice, "optimal_param"),
        ("EPOCH", epoch_path, compute_epoch_sacrifice, "optimal_epoch"),
    ]:
        print(f"\n{'─' * 40}")
        print(f"  {dim_label} DIMENSION")
        print(f"{'─' * 40}")

        for group_label, models in [("Baseline", baseline_models), ("OG", og_models)]:
            print(f"\n  [{group_label} models]")
            for model in models:
                sac = compute_fn(path, model, strategies)
                if sac is None:
                    print(f"    {model}: no data")
                    continue

                summary = sacrifice_summary(sac, strategies)
                if summary.empty:
                    continue

                summary["dimension"] = dim_label
                summary["group"] = group_label
                all_results.append(summary)

                max_loss = summary.loc[summary["pct_loss"].idxmax()]
                print(f"    {model:20s}  max sacrifice: "
                      f"{max_loss['optimized_for']}→{max_loss['evaluated_on']}  "
                      f"R²={max_loss['R2_at_own_optimum']:.3f}→{max_loss['R2_at_forced_optimum']:.3f}  "
                      f"loss={max_loss['pct_loss']:.1f}%")

    if all_results:
        full_df = pd.concat(all_results, ignore_index=True)

        print(f"\n{'=' * 80}")
        print("SUMMARY")
        print(f"{'=' * 80}")

        for group in ["Baseline", "OG"]:
            sub = full_df[full_df["group"] == group]
            if sub.empty:
                continue
            print(f"\n  {group} models:")
            print(f"    Avg forced sacrifice:  {sub['pct_loss'].mean():.1f}%")
            print(f"    Max forced sacrifice:  {sub['pct_loss'].max():.1f}% "
                  f"({sub.loc[sub['pct_loss'].idxmax(), 'model']})")
            print(f"    Min forced sacrifice:  {sub['pct_loss'].min():.1f}% "
                  f"({sub.loc[sub['pct_loss'].idxmin(), 'model']})")

        return full_df
    return pd.DataFrame()
