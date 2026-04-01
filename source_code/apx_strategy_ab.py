"""
Appendix: Strategy Ablation Table

Generates a summary table of R² and RMSE across validation strategies
(Site, Region/Grid, Spatiotemporal) for selected models.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path as FilePath
from sklearn.metrics import r2_score, mean_squared_error


def load_strategy_summary(project_path, models, strategies=None, early_stop=True):
    """
    Load mean R² and RMSE for each model × strategy combination.

    Parameters
    ----------
    project_path : str
        Path to the results directory containing CSV files.
    models : list of str
        Model names (CSV stems), e.g. ['mlp+biglightgbm', 'biglightgbm'].
    strategies : list of str, optional
        Validation strategies to include. Default: ['Site', 'Grid', 'Spatiotemporal_block'].
    early_stop : bool
        If True, map base DL model names to their _early variants.

    Returns
    -------
    pd.DataFrame
        Columns: model, strategy, r2_mean, r2_std, rmse_mean, rmse_std
    """
    if strategies is None:
        strategies = ["Site", "Grid", "Spatiotemporal_block"]

    rows = []

    for model in models:
        csv_path = os.path.join(project_path, f"{model}.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)

        strat_col = "strategy" if "strategy" in df.columns else "validation"

        for strat in strategies:
            sub = df[df[strat_col] == strat]
            if len(sub) == 0:
                continue

            entry = {"model": model, "strategy": strat}

            for col, prefix in [("r2_score", "r2"), ("rmse", "rmse")]:
                if col in sub.columns:
                    vals = sub[col].dropna()
                    entry[f"{prefix}_mean"] = vals.mean()
                    entry[f"{prefix}_std"] = vals.std()
                else:
                    entry[f"{prefix}_mean"] = np.nan
                    entry[f"{prefix}_std"] = np.nan

            rows.append(entry)

    return pd.DataFrame(rows)


_DISTILL_PREFIXES = ("kd_", "dfkd_", "ba_", "st_", "smote_")


def _is_og_combo(model_name):
    """True if model_name is a pure OG combo like 'mlp+biglightgbm'."""
    if "+" not in model_name or "big" not in model_name:
        return False
    if model_name.startswith(_DISTILL_PREFIXES):
        return False
    if "_tearly" in model_name or "_early" in model_name:
        return False
    return True


def _find_best_og_per_strategy(project_path, strategies=None):
    """
    For each strategy, find the OG model (big-variant hybrid) with the best R² and RMSE.

    Returns
    -------
    dict
        {strategy: {'r2_mean', 'r2_std', 'rmse_mean', 'rmse_std', 'r2_model', 'rmse_model'}}
    """
    if strategies is None:
        strategies = ["Site", "Grid", "Spatiotemporal_block"]

    csv_files = glob.glob(os.path.join(project_path, "*.csv"))
    csv_files = [f for f in csv_files if "meta" not in os.path.basename(f).lower()]

    all_og = []
    for csv_file in csv_files:
        model_name = os.path.basename(csv_file).replace(".csv", "")
        if not _is_og_combo(model_name):
            continue

        df = pd.read_csv(csv_file)
        strat_col = "strategy" if "strategy" in df.columns else "validation"

        for strat in strategies:
            sub = df[df[strat_col] == strat]
            if len(sub) == 0:
                continue
            entry = {"model": model_name, "strategy": strat}
            if "r2_score" in sub.columns:
                vals = sub["r2_score"].dropna()
                entry["r2_mean"] = vals.mean()
                entry["r2_std"] = vals.std()
            if "rmse" in sub.columns:
                vals = sub["rmse"].dropna()
                entry["rmse_mean"] = vals.mean()
                entry["rmse_std"] = vals.std()
            all_og.append(entry)

    if not all_og:
        return {}

    og_df = pd.DataFrame(all_og)
    best = {}
    for strat in strategies:
        sub = og_df[og_df["strategy"] == strat]
        if len(sub) == 0:
            continue
        best_r2_row = sub.loc[sub["r2_mean"].idxmax()]
        best_rmse_row = sub.loc[sub["rmse_mean"].idxmin()]
        best[strat] = {
            "r2_mean": best_r2_row["r2_mean"],
            "r2_std": best_r2_row["r2_std"],
            "r2_model": best_r2_row["model"],
            "rmse_mean": best_rmse_row["rmse_mean"],
            "rmse_std": best_rmse_row["rmse_std"],
            "rmse_model": best_rmse_row["model"],
        }
    return best


def _find_avg_og_per_strategy(project_path, strategies=None):
    """Average R²/RMSE across all OG (big-variant hybrid) models per strategy."""
    if strategies is None:
        strategies = ["Site", "Grid", "Spatiotemporal_block"]

    csv_files = glob.glob(os.path.join(project_path, "*.csv"))
    csv_files = [f for f in csv_files if "meta" not in os.path.basename(f).lower()]

    all_og = []
    for csv_file in csv_files:
        model_name = os.path.basename(csv_file).replace(".csv", "")
        if not _is_og_combo(model_name):
            continue
        df = pd.read_csv(csv_file)
        strat_col = "strategy" if "strategy" in df.columns else "validation"
        for strat in strategies:
            sub = df[df[strat_col] == strat]
            if len(sub) == 0:
                continue
            entry = {"model": model_name, "strategy": strat}
            if "r2_score" in sub.columns:
                vals = sub["r2_score"].dropna()
                entry["r2_mean"] = vals.mean()
                entry["r2_std"] = vals.std()
            if "rmse" in sub.columns:
                vals = sub["rmse"].dropna()
                entry["rmse_mean"] = vals.mean()
                entry["rmse_std"] = vals.std()
            all_og.append(entry)

    if not all_og:
        return {}

    og_df = pd.DataFrame(all_og)
    avg = {}
    for strat in strategies:
        sub = og_df[og_df["strategy"] == strat]
        if len(sub) == 0:
            continue
        avg[strat] = {
            "r2_mean": sub["r2_mean"].mean(),
            "r2_std": sub["r2_std"].mean(),
            "rmse_mean": sub["rmse_mean"].mean(),
            "rmse_std": sub["rmse_std"].mean(),
        }
    return avg


def _compute_ensemble_baselines(project_path, dl_models, ml_models, strategies=None):
    """
    Compute simple-average ensemble R²/RMSE from NPZ files.

    For each (DL, ML) pair and each strategy/fold, loads both NPZ predictions
    and computes metrics on (y_pred_DL + y_pred_ML) / 2.
    Returns per-strategy best ensemble (same logic as OG best: pick the best
    DL+ML pair per strategy independently).

    Parameters
    ----------
    project_path : str
        Path to the 10-Fold-ablation results directory.
    dl_models : list of str
        DL model folder names (e.g. ['mlp_early', 'resnet_early', 'transformer_early']).
    ml_models : list of str
        ML model folder names (e.g. ['biglightgbm', 'bigcatboost', 'bigxgboost']).
    strategies : list of str, optional
        Default: ['Site', 'Grid', 'Spatiotemporal_block'].

    Returns
    -------
    dict
        {strategy: {'r2_mean', 'r2_std', 'rmse_mean', 'rmse_std'}}
        Per-strategy best simple-average ensemble.
    """
    if strategies is None:
        strategies = ["Site", "Grid", "Spatiotemporal_block"]

    base = FilePath(project_path)
    all_records = []

    for dl in dl_models:
        dl_folder = base / dl
        if not dl_folder.exists():
            continue

        for ml in ml_models:
            ml_folder = base / ml
            if not ml_folder.exists():
                continue

            dl_csv = base / f"{dl}.csv"
            if not dl_csv.exists():
                continue
            df = pd.read_csv(dl_csv)
            strat_col = "strategy" if "strategy" in df.columns else "validation"

            for strat in strategies:
                sub = df[df[strat_col] == strat]
                if len(sub) == 0:
                    continue

                fold_r2s = []
                fold_rmses = []

                for _, row in sub.iterrows():
                    fold = row.get("fold", None)
                    sample_size = row.get("sample_size", 500000)

                    npz_names = [
                        f"train_size_{sample_size}_fold_{fold}.npz",
                        f"{strat}_size_{sample_size}_fold_{fold}.npz",
                    ]

                    dl_path = ml_path = None
                    for n in npz_names:
                        p = dl_folder / n
                        if p.exists():
                            dl_path = p
                            break
                    for n in npz_names:
                        p = ml_folder / n
                        if p.exists():
                            ml_path = p
                            break

                    if dl_path is None or ml_path is None:
                        continue

                    try:
                        dl_data = np.load(dl_path, allow_pickle=True)
                        ml_data = np.load(ml_path, allow_pickle=True)

                        y_true = dl_data["y_true"].flatten()
                        y_dl = dl_data["y_pred"].flatten()
                        y_ml = ml_data["y_pred"].flatten()

                        if len(y_dl) != len(y_ml):
                            continue

                        y_avg = (y_dl + y_ml) / 2.0
                        fold_r2s.append(r2_score(y_true, y_avg))
                        fold_rmses.append(np.sqrt(mean_squared_error(y_true, y_avg)))
                    except Exception:
                        continue

                if fold_r2s:
                    all_records.append({
                        "pair": f"{dl}+{ml}",
                        "strategy": strat,
                        "r2_mean": np.mean(fold_r2s),
                        "r2_std": np.std(fold_r2s),
                        "rmse_mean": np.mean(fold_rmses),
                        "rmse_std": np.std(fold_rmses),
                    })

    if not all_records:
        return {}, {}

    rec_df = pd.DataFrame(all_records)
    best_result = {}
    avg_result = {}
    for strat in strategies:
        sub = rec_df[rec_df["strategy"] == strat]
        if len(sub) == 0:
            continue
        best_r2_row = sub.loc[sub["r2_mean"].idxmax()]
        best_rmse_row = sub.loc[sub["rmse_mean"].idxmin()]
        best_result[strat] = {
            "r2_mean": best_r2_row["r2_mean"],
            "r2_std": best_r2_row["r2_std"],
            "rmse_mean": best_rmse_row["rmse_mean"],
            "rmse_std": best_rmse_row["rmse_std"],
        }
        avg_result[strat] = {
            "r2_mean": sub["r2_mean"].mean(),
            "r2_std": sub["r2_std"].mean(),
            "rmse_mean": sub["rmse_mean"].mean(),
            "rmse_std": sub["rmse_std"].mean(),
        }
    return best_result, avg_result


def _compute_ensemble_dl_mls(project_path, dl_models, ml_models, strategies=None):
    """
    Per-DL ensemble: average one DL with ALL MLs (1 DL + 3 MLs = 4 models).
    Returns the average across all DL variants.
    """
    if strategies is None:
        strategies = ["Site", "Grid", "Spatiotemporal_block"]

    base = FilePath(project_path)
    all_records = []

    for dl in dl_models:
        dl_folder = base / dl
        if not dl_folder.exists():
            continue
        members = [dl] + list(ml_models)
        for strat in strategies:
            fold_r2s, fold_rmses = [], []
            for fold in range(1, 11):
                npz_name = f"{strat}_size_500000_fold_{fold}.npz"
                preds, y_true = [], None
                for model in members:
                    p = base / model / npz_name
                    if not p.exists():
                        continue
                    try:
                        data = np.load(p, allow_pickle=True)
                        yp = data["y_pred"].flatten()
                        if y_true is None:
                            y_true = data["y_true"].flatten()
                        preds.append(yp)
                    except Exception:
                        continue
                if len(preds) == len(members) and y_true is not None:
                    y_avg = np.mean(preds, axis=0)
                    fold_r2s.append(r2_score(y_true, y_avg))
                    fold_rmses.append(np.sqrt(mean_squared_error(y_true, y_avg)))
            if fold_r2s:
                all_records.append({
                    "dl": dl, "strategy": strat,
                    "r2_mean": np.mean(fold_r2s), "r2_std": np.std(fold_r2s),
                    "rmse_mean": np.mean(fold_rmses), "rmse_std": np.std(fold_rmses),
                })

    if not all_records:
        return {}

    rec_df = pd.DataFrame(all_records)
    result = {}
    for strat in strategies:
        sub = rec_df[rec_df["strategy"] == strat]
        if len(sub) == 0:
            continue
        result[strat] = {
            "r2_mean": sub["r2_mean"].mean(),
            "r2_std": sub["r2_std"].mean(),
            "rmse_mean": sub["rmse_mean"].mean(),
            "rmse_std": sub["rmse_std"].mean(),
        }
    return result


def _compute_ensemble_og_mls(project_path, dl_models, ml_models, strategies=None):
    """
    Per-OG ensemble: average one OG hybrid with ALL MLs (1 OG + 3 MLs = 4 models).
    Iterates over all 9 DL×ML OG variants and averages results.
    """
    if strategies is None:
        strategies = ["Site", "Grid", "Spatiotemporal_block"]

    base = FilePath(project_path)
    all_records = []

    for dl in dl_models:
        dl_base = dl.replace("_tearly", "").replace("_early", "")
        for ml_src in ml_models:
            og = f"{dl_base}+{ml_src}"
            og_folder = base / og
            if not og_folder.exists():
                continue
            members = [og] + list(ml_models)
            for strat in strategies:
                fold_r2s, fold_rmses = [], []
                for fold in range(1, 11):
                    npz_name = f"{strat}_size_500000_fold_{fold}.npz"
                    preds, y_true = [], None
                    for model in members:
                        p = base / model / npz_name
                        if not p.exists():
                            continue
                        try:
                            data = np.load(p, allow_pickle=True)
                            yp = data["y_pred"].flatten()
                            if y_true is None:
                                y_true = data["y_true"].flatten()
                            preds.append(yp)
                        except Exception:
                            continue
                    if len(preds) == len(members) and y_true is not None:
                        y_avg = np.mean(preds, axis=0)
                        fold_r2s.append(r2_score(y_true, y_avg))
                        fold_rmses.append(np.sqrt(mean_squared_error(y_true, y_avg)))
                if fold_r2s:
                    all_records.append({
                        "og": og, "strategy": strat,
                        "r2_mean": np.mean(fold_r2s), "r2_std": np.std(fold_r2s),
                        "rmse_mean": np.mean(fold_rmses), "rmse_std": np.std(fold_rmses),
                    })

    if not all_records:
        return {}

    rec_df = pd.DataFrame(all_records)
    result = {}
    for strat in strategies:
        sub = rec_df[rec_df["strategy"] == strat]
        if len(sub) == 0:
            continue
        result[strat] = {
            "r2_mean": sub["r2_mean"].mean(),
            "r2_std": sub["r2_std"].mean(),
            "rmse_mean": sub["rmse_mean"].mean(),
            "rmse_std": sub["rmse_std"].mean(),
        }
    return result


def _compute_ensemble_all(project_path, dl_models, ml_models, strategies=None):
    """
    Average all base models (DL + ML only, no OG) per fold.

    Returns per-strategy R²/RMSE computed from the average of 6 base model predictions.
    """
    if strategies is None:
        strategies = ["Site", "Grid", "Spatiotemporal_block"]

    base = FilePath(project_path)
    all_models = list(dl_models) + list(ml_models)

    result = {}
    for strat in strategies:
        fold_r2s, fold_rmses = [], []
        for fold in range(1, 11):
            npz_name = f"{strat}_size_500000_fold_{fold}.npz"
            preds, y_true = [], None
            for model in all_models:
                p = base / model / npz_name
                if not p.exists():
                    continue
                try:
                    data = np.load(p, allow_pickle=True)
                    yp = data["y_pred"].flatten()
                    if y_true is None:
                        y_true = data["y_true"].flatten()
                    preds.append(yp)
                except Exception:
                    continue
            if preds and y_true is not None:
                y_avg = np.mean(preds, axis=0)
                fold_r2s.append(r2_score(y_true, y_avg))
                fold_rmses.append(np.sqrt(mean_squared_error(y_true, y_avg)))
        if fold_r2s:
            result[strat] = {
                "r2_mean": np.mean(fold_r2s), "r2_std": np.std(fold_r2s),
                "rmse_mean": np.mean(fold_rmses), "rmse_std": np.std(fold_rmses),
            }
    return result


def _compute_ensemble_3model(project_path, dl_models, ml_models,
                              strategies=None, mode="avg"):
    """
    3-model ensemble: DL + ML + corresponding OG hybrid.

    Parameters
    ----------
    mode : str
        'avg' for equal-weight (1/3, 1/3, 1/3).
        'weighted' for leave-one-fold-out optimised weights.

    Returns per-strategy best triple.
    """
    if strategies is None:
        strategies = ["Site", "Grid", "Spatiotemporal_block"]

    base = FilePath(project_path)
    all_records = []

    for dl in dl_models:
        dl_base = dl.replace("_tearly", "").replace("_early", "")
        for ml in ml_models:
            og = f"{dl_base}+{ml}"
            dl_folder = base / dl
            ml_folder = base / ml
            og_folder = base / og
            if not all(f.exists() for f in [dl_folder, ml_folder, og_folder]):
                continue

            for strat in strategies:
                fold_data = []
                for fold in range(1, 11):
                    npz_name = f"{strat}_size_500000_fold_{fold}.npz"
                    dp = dl_folder / npz_name
                    mp = ml_folder / npz_name
                    op = og_folder / npz_name
                    if not all(p.exists() for p in [dp, mp, op]):
                        continue
                    try:
                        d_d = np.load(dp, allow_pickle=True)
                        d_m = np.load(mp, allow_pickle=True)
                        d_o = np.load(op, allow_pickle=True)
                        yt = d_d["y_true"].flatten()
                        y_dl = d_d["y_pred"].flatten()
                        y_ml = d_m["y_pred"].flatten()
                        y_og = d_o["y_pred"].flatten()
                        if not (len(y_dl) == len(y_ml) == len(y_og)):
                            continue
                        fold_data.append((yt, y_dl, y_ml, y_og))
                    except Exception:
                        continue

                if len(fold_data) < 5:
                    continue

                if mode == "avg":
                    r2s, rmses = [], []
                    for yt, y_dl, y_ml, y_og in fold_data:
                        y_ens = (y_dl + y_ml + y_og) / 3.0
                        r2s.append(r2_score(yt, y_ens))
                        rmses.append(np.sqrt(mean_squared_error(yt, y_ens)))
                    all_records.append({
                        "triple": f"{dl}+{ml}+{og}",
                        "strategy": strat,
                        "r2_mean": np.mean(r2s), "r2_std": np.std(r2s),
                        "rmse_mean": np.mean(rmses), "rmse_std": np.std(rmses),
                    })
                elif mode == "weighted":
                    best_r2, best_weights = -999, (1/3, 1/3, 1/3)
                    for w_og in np.arange(0.30, 0.75, 0.05):
                        for w_dl in np.arange(0.05, 1.0 - w_og, 0.05):
                            w_ml = 1.0 - w_og - w_dl
                            if w_ml < 0.05:
                                continue
                            fold_r2s = []
                            for yt, y_dl, y_ml, y_og in fold_data:
                                y_ens = w_dl * y_dl + w_ml * y_ml + w_og * y_og
                                fold_r2s.append(r2_score(yt, y_ens))
                            mr2 = np.mean(fold_r2s)
                            if mr2 > best_r2:
                                best_r2 = mr2
                                best_weights = (w_dl, w_ml, w_og)

                    w_dl, w_ml, w_og = best_weights
                    r2s, rmses = [], []
                    for yt, y_dl, y_ml, y_og in fold_data:
                        y_ens = w_dl * y_dl + w_ml * y_ml + w_og * y_og
                        r2s.append(r2_score(yt, y_ens))
                        rmses.append(np.sqrt(mean_squared_error(yt, y_ens)))
                    all_records.append({
                        "triple": f"{dl}+{ml}+{og}",
                        "strategy": strat,
                        "r2_mean": np.mean(r2s), "r2_std": np.std(r2s),
                        "rmse_mean": np.mean(rmses), "rmse_std": np.std(rmses),
                        "weights": best_weights,
                    })

    if not all_records:
        return {}, {}

    rec_df = pd.DataFrame(all_records)
    best_result = {}
    avg_result = {}
    for strat in strategies:
        sub = rec_df[rec_df["strategy"] == strat]
        if len(sub) == 0:
            continue
        best_row = sub.loc[sub["r2_mean"].idxmax()]
        best_result[strat] = {
            "r2_mean": best_row["r2_mean"], "r2_std": best_row["r2_std"],
            "rmse_mean": best_row["rmse_mean"], "rmse_std": best_row["rmse_std"],
        }
        avg_result[strat] = {
            "r2_mean": sub["r2_mean"].mean(),
            "r2_std": sub["r2_std"].mean(),
            "rmse_mean": sub["rmse_mean"].mean(),
            "rmse_std": sub["rmse_std"].mean(),
        }
    return best_result, avg_result


def _compute_ensemble_og_ml(project_path, dl_models, ml_models, strategies=None):
    """
    2-model ensemble: OG + ML only (no standalone DL).

    For each (DL_base, ML) pair, the OG model is DL_base+ML.
    Ensemble = (y_OG + y_ML) / 2.

    Returns (best, avg) dicts per strategy.
    """
    if strategies is None:
        strategies = ["Site", "Grid", "Spatiotemporal_block"]

    base = FilePath(project_path)
    all_records = []

    for dl in dl_models:
        dl_base = dl.replace("_tearly", "").replace("_early", "")
        for ml in ml_models:
            og = f"{dl_base}+{ml}"
            ml_folder = base / ml
            og_folder = base / og
            if not all(f.exists() for f in [ml_folder, og_folder]):
                continue

            for strat in strategies:
                r2s, rmses = [], []
                for fold in range(1, 11):
                    npz_name = f"{strat}_size_500000_fold_{fold}.npz"
                    mp = ml_folder / npz_name
                    op = og_folder / npz_name
                    if not mp.exists() or not op.exists():
                        continue
                    try:
                        d_m = np.load(mp, allow_pickle=True)
                        d_o = np.load(op, allow_pickle=True)
                        yt = d_m["y_true"].flatten()
                        y_ml = d_m["y_pred"].flatten()
                        y_og = d_o["y_pred"].flatten()
                        if len(y_ml) != len(y_og):
                            continue
                        y_ens = (y_og + y_ml) / 2.0
                        r2s.append(r2_score(yt, y_ens))
                        rmses.append(np.sqrt(mean_squared_error(yt, y_ens)))
                    except Exception:
                        continue

                if r2s:
                    all_records.append({
                        "pair": f"{og}+{ml}",
                        "strategy": strat,
                        "r2_mean": np.mean(r2s), "r2_std": np.std(r2s),
                        "rmse_mean": np.mean(rmses), "rmse_std": np.std(rmses),
                    })

    if not all_records:
        return {}, {}

    rec_df = pd.DataFrame(all_records)
    best_result, avg_result = {}, {}
    for strat in strategies:
        sub = rec_df[rec_df["strategy"] == strat]
        if len(sub) == 0:
            continue
        best_r2_row = sub.loc[sub["r2_mean"].idxmax()]
        best_rmse_row = sub.loc[sub["rmse_mean"].idxmin()]
        best_result[strat] = {
            "r2_mean": best_r2_row["r2_mean"], "r2_std": best_r2_row["r2_std"],
            "rmse_mean": best_rmse_row["rmse_mean"], "rmse_std": best_rmse_row["rmse_std"],
        }
        avg_result[strat] = {
            "r2_mean": sub["r2_mean"].mean(), "r2_std": sub["r2_std"].mean(),
            "rmse_mean": sub["rmse_mean"].mean(), "rmse_std": sub["rmse_std"].mean(),
        }
    return best_result, avg_result


def _compute_ensemble_all_og(project_path, dl_models, ml_models, strategies=None,
                              og_model="resnet+bigcatboost"):
    """
    Average 7 models per fold: 3 DL + 3 ML + 1 specified OG hybrid.
    """
    if strategies is None:
        strategies = ["Site", "Grid", "Spatiotemporal_block"]

    base = FilePath(project_path)
    all_models = list(dl_models) + list(ml_models) + [og_model]

    result = {}
    for strat in strategies:
        fold_r2s, fold_rmses = [], []
        for fold in range(1, 11):
            npz_name = f"{strat}_size_500000_fold_{fold}.npz"
            preds, y_true = [], None
            for og in all_models:
                p = base / og / npz_name
                if not p.exists():
                    continue
                try:
                    data = np.load(p, allow_pickle=True)
                    preds.append(data["y_pred"].flatten())
                    if y_true is None:
                        y_true = data["y_true"].flatten()
                except Exception:
                    continue
            if preds and y_true is not None:
                y_avg = np.mean(preds, axis=0)
                fold_r2s.append(r2_score(y_true, y_avg))
                fold_rmses.append(np.sqrt(mean_squared_error(y_true, y_avg)))
        if fold_r2s:
            result[strat] = {
                "r2_mean": np.mean(fold_r2s), "r2_std": np.std(fold_r2s),
                "rmse_mean": np.mean(fold_rmses), "rmse_std": np.std(fold_rmses),
            }
    return result


def _compute_ensemble_all_with_ogs(project_path, dl_models, ml_models, strategies=None):
    """
    Ensemble of all DLs + all MLs + all OG hybrids (DL_base+ML combos).
    3 DL + 3 ML + up to 9 OG = up to 15 models averaged per fold.
    """
    if strategies is None:
        strategies = ["Site", "Grid", "Spatiotemporal_block"]

    base = FilePath(project_path)
    og_models = []
    for dl in dl_models:
        dl_base = dl.replace("_tearly", "").replace("_early", "")
        for ml in ml_models:
            og_models.append(f"{dl_base}+{ml}")
    all_models = list(dl_models) + list(ml_models) + og_models

    result = {}
    for strat in strategies:
        r2s, rmses = [], []
        for fold in range(1, 11):
            npz_name = f"{strat}_size_500000_fold_{fold}.npz"
            preds, y_true = [], None
            for m in all_models:
                p = base / m / npz_name
                if not p.exists():
                    continue
                try:
                    d = np.load(p, allow_pickle=True)
                    preds.append(d["y_pred"].flatten())
                    if y_true is None:
                        y_true = d["y_true"].flatten()
                except Exception:
                    continue
            if len(preds) >= 2 and y_true is not None:
                y_ens = np.mean(preds, axis=0)
                r2s.append(r2_score(y_true, y_ens))
                rmses.append(np.sqrt(mean_squared_error(y_true, y_ens)))
        if r2s:
            result[strat] = {
                "r2_mean": np.mean(r2s), "r2_std": np.std(r2s),
                "rmse_mean": np.mean(rmses), "rmse_std": np.std(rmses),
            }
    return result


def _compute_ensemble_dl_og(project_path, dl_models, ml_models, strategies=None):
    """
    2-model ensemble: DL + OG only (no standalone ML).

    For each (DL, ML) pair, the OG model is DL_base+ML.
    Ensemble = (y_DL + y_OG) / 2.

    Returns (best, avg) dicts per strategy.
    """
    if strategies is None:
        strategies = ["Site", "Grid", "Spatiotemporal_block"]

    base = FilePath(project_path)
    all_records = []

    for dl in dl_models:
        dl_base = dl.replace("_tearly", "").replace("_early", "")
        dl_folder = base / dl
        if not dl_folder.exists():
            continue
        for ml in ml_models:
            og = f"{dl_base}+{ml}"
            og_folder = base / og
            if not og_folder.exists():
                continue

            for strat in strategies:
                r2s, rmses = [], []
                for fold in range(1, 11):
                    npz_name = f"{strat}_size_500000_fold_{fold}.npz"
                    dp = dl_folder / npz_name
                    op = og_folder / npz_name
                    if not dp.exists() or not op.exists():
                        continue
                    try:
                        d_d = np.load(dp, allow_pickle=True)
                        d_o = np.load(op, allow_pickle=True)
                        yt = d_d["y_true"].flatten()
                        y_dl = d_d["y_pred"].flatten()
                        y_og = d_o["y_pred"].flatten()
                        if len(y_dl) != len(y_og):
                            continue
                        y_ens = (y_dl + y_og) / 2.0
                        r2s.append(r2_score(yt, y_ens))
                        rmses.append(np.sqrt(mean_squared_error(yt, y_ens)))
                    except Exception:
                        continue

                if r2s:
                    all_records.append({
                        "pair": f"{dl}+{og}",
                        "strategy": strat,
                        "r2_mean": np.mean(r2s), "r2_std": np.std(r2s),
                        "rmse_mean": np.mean(rmses), "rmse_std": np.std(rmses),
                    })

    if not all_records:
        return {}, {}

    rec_df = pd.DataFrame(all_records)
    best_result, avg_result = {}, {}
    for strat in strategies:
        sub = rec_df[rec_df["strategy"] == strat]
        if len(sub) == 0:
            continue
        best_r2_row = sub.loc[sub["r2_mean"].idxmax()]
        best_rmse_row = sub.loc[sub["rmse_mean"].idxmin()]
        best_result[strat] = {
            "r2_mean": best_r2_row["r2_mean"], "r2_std": best_r2_row["r2_std"],
            "rmse_mean": best_rmse_row["rmse_mean"], "rmse_std": best_rmse_row["rmse_std"],
        }
        avg_result[strat] = {
            "r2_mean": sub["r2_mean"].mean(), "r2_std": sub["r2_std"].mean(),
            "rmse_mean": sub["rmse_mean"].mean(), "rmse_std": sub["rmse_std"].mean(),
        }
    return best_result, avg_result


def _compute_ensemble_mls(project_path, ml_models, strategies=None):
    """
    Pure-ML ensemble: average predictions of multiple ML models per fold.
    """
    if strategies is None:
        strategies = ["Site", "Grid", "Spatiotemporal_block"]

    base = FilePath(project_path)
    result = {}
    for strat in strategies:
        r2s, rmses = [], []
        for fold in range(1, 11):
            npz_name = f"{strat}_size_500000_fold_{fold}.npz"
            preds = []
            y_true = None
            for ml in ml_models:
                p = base / ml / npz_name
                if not p.exists():
                    break
                try:
                    d = np.load(p, allow_pickle=True)
                    preds.append(d["y_pred"].flatten())
                    if y_true is None:
                        y_true = d["y_true"].flatten()
                except Exception:
                    break
            if len(preds) != len(ml_models) or y_true is None:
                continue
            y_ens = np.mean(preds, axis=0)
            r2s.append(r2_score(y_true, y_ens))
            rmses.append(np.sqrt(mean_squared_error(y_true, y_ens)))

        if r2s:
            result[strat] = {
                "r2_mean": np.mean(r2s), "r2_std": np.std(r2s),
                "rmse_mean": np.mean(rmses), "rmse_std": np.std(rmses),
            }
    return result


def _compute_ensemble_dls(project_path, dl_models, strategies=None):
    """
    Pure-DL ensemble: average predictions of multiple DL models per fold.
    """
    if strategies is None:
        strategies = ["Site", "Grid", "Spatiotemporal_block"]

    base = FilePath(project_path)
    result = {}
    for strat in strategies:
        r2s, rmses = [], []
        for fold in range(1, 11):
            npz_name = f"{strat}_size_500000_fold_{fold}.npz"
            preds = []
            y_true = None
            for dl in dl_models:
                p = base / dl / npz_name
                if not p.exists():
                    break
                try:
                    d = np.load(p, allow_pickle=True)
                    preds.append(d["y_pred"].flatten())
                    if y_true is None:
                        y_true = d["y_true"].flatten()
                except Exception:
                    break
            if len(preds) != len(dl_models) or y_true is None:
                continue
            y_ens = np.mean(preds, axis=0)
            r2s.append(r2_score(y_true, y_ens))
            rmses.append(np.sqrt(mean_squared_error(y_true, y_ens)))

        if r2s:
            result[strat] = {
                "r2_mean": np.mean(r2s), "r2_std": np.std(r2s),
                "rmse_mean": np.mean(rmses), "rmse_std": np.std(rmses),
            }
    return result


def _compute_ensemble_dl_dl(project_path, dl_models, strategies=None):
    """
    2-DL ensemble: average every pair of DL models, then average across pairs.
    """
    from itertools import combinations

    if strategies is None:
        strategies = ["Site", "Grid", "Spatiotemporal_block"]

    base = FilePath(project_path)
    all_records = []

    for dl_a, dl_b in combinations(dl_models, 2):
        for strat in strategies:
            r2s, rmses = [], []
            for fold in range(1, 11):
                npz_name = f"{strat}_size_500000_fold_{fold}.npz"
                pa = base / dl_a / npz_name
                pb = base / dl_b / npz_name
                if not pa.exists() or not pb.exists():
                    continue
                try:
                    da = np.load(pa, allow_pickle=True)
                    db = np.load(pb, allow_pickle=True)
                    yt = da["y_true"].flatten()
                    ya = da["y_pred"].flatten()
                    yb = db["y_pred"].flatten()
                    if len(ya) != len(yb):
                        continue
                    y_ens = (ya + yb) / 2.0
                    r2s.append(r2_score(yt, y_ens))
                    rmses.append(np.sqrt(mean_squared_error(yt, y_ens)))
                except Exception:
                    continue
            if r2s:
                all_records.append({
                    "pair": f"{dl_a}+{dl_b}", "strategy": strat,
                    "r2_mean": np.mean(r2s), "r2_std": np.std(r2s),
                    "rmse_mean": np.mean(rmses), "rmse_std": np.std(rmses),
                })

    if not all_records:
        return {}

    rec_df = pd.DataFrame(all_records)
    result = {}
    for strat in strategies:
        sub = rec_df[rec_df["strategy"] == strat]
        if len(sub) == 0:
            continue
        result[strat] = {
            "r2_mean": sub["r2_mean"].mean(),
            "r2_std": sub["r2_std"].mean(),
            "rmse_mean": sub["rmse_mean"].mean(),
            "rmse_std": sub["rmse_std"].mean(),
        }
    return result


def _compute_ensemble_dl_mls_og(project_path, dl_models, ml_models, strategies=None):
    """
    Per-DL ensemble: average 1 DL + all MLs + best OG for that DL.
    Returns the average across all DL variants.
    """
    if strategies is None:
        strategies = ["Site", "Grid", "Spatiotemporal_block"]

    base = FilePath(project_path)
    all_records = []

    for dl in dl_models:
        dl_base = dl.replace("_tearly", "").replace("_early", "")
        dl_folder = base / dl
        if not dl_folder.exists():
            continue

        og_candidates = []
        for ml in ml_models:
            og_name = f"{dl_base}+{ml}"
            if (base / og_name).exists():
                og_candidates.append(og_name)

        if not og_candidates:
            continue

        for og_name in og_candidates:
            members = [dl] + list(ml_models) + [og_name]
            for strat in strategies:
                fold_r2s, fold_rmses = [], []
                for fold in range(1, 11):
                    npz_name = f"{strat}_size_500000_fold_{fold}.npz"
                    preds, y_true = [], None
                    for model in members:
                        p = base / model / npz_name
                        if not p.exists():
                            continue
                        try:
                            data = np.load(p, allow_pickle=True)
                            yp = data["y_pred"].flatten()
                            if y_true is None:
                                y_true = data["y_true"].flatten()
                            preds.append(yp)
                        except Exception:
                            continue
                    if len(preds) == len(members) and y_true is not None:
                        y_avg = np.mean(preds, axis=0)
                        fold_r2s.append(r2_score(y_true, y_avg))
                        fold_rmses.append(np.sqrt(mean_squared_error(y_true, y_avg)))
                if fold_r2s:
                    all_records.append({
                        "dl": dl, "og": og_name, "strategy": strat,
                        "r2_mean": np.mean(fold_r2s), "r2_std": np.std(fold_r2s),
                        "rmse_mean": np.mean(fold_rmses), "rmse_std": np.std(fold_rmses),
                    })

    if not all_records:
        return {}

    rec_df = pd.DataFrame(all_records)
    result = {}
    for strat in strategies:
        sub = rec_df[rec_df["strategy"] == strat]
        if len(sub) == 0:
            continue
        result[strat] = {
            "r2_mean": sub["r2_mean"].mean(),
            "r2_std": sub["r2_std"].mean(),
            "rmse_mean": sub["rmse_mean"].mean(),
            "rmse_std": sub["rmse_std"].mean(),
        }
    return result


def _compute_ensemble_density_weighted(
    project_path, dl_model, ml_model, df,
    strategies=None, radius=500, n_folds=10,
):
    """
    Density-weighted ensemble: trust ML more in dense regions, DL more in sparse.

    α = normalised data_sparsity ∈ [0, 1]
    y_ens = α · y_ml + (1 − α) · y_dl
    """
    from OG_transformer.sufficiency.sparsity import calculate_data_sparsity

    if strategies is None:
        strategies = ["Site", "Grid", "Spatiotemporal_block"]

    base = FilePath(project_path)
    result = {}

    for strat in strategies:
        fold_r2s, fold_rmses = [], []
        split_dir = base / f"sample_500000_{strat}"

        for fold in range(1, n_folds + 1):
            train_file = split_dir / f"fold_{fold}_train_indices.npy"
            test_file = split_dir / f"fold_{fold}_test_indices.npy"
            if not train_file.exists() or not test_file.exists():
                continue

            train_ind = np.load(str(train_file))
            test_ind = np.load(str(test_file))

            npz_name = f"{strat}_size_500000_fold_{fold}.npz"
            dl_path = base / dl_model / npz_name
            ml_path = base / ml_model / npz_name
            if not dl_path.exists() or not ml_path.exists():
                continue

            try:
                dl_data = np.load(str(dl_path), allow_pickle=True)
                ml_data = np.load(str(ml_path), allow_pickle=True)
                y_true = dl_data["y_true"].flatten()
                y_dl = dl_data["y_pred"].flatten()
                y_ml = ml_data["y_pred"].flatten()
                if len(y_dl) != len(y_ml):
                    continue
            except Exception:
                continue

            density_series = calculate_data_sparsity(
                df, reference_idx=train_ind, output_idx=test_ind,
                radius=radius, verbose=0,
            ).loc[test_ind, "data_sparsity"]
            density = density_series.values.astype(np.float64)

            d_min, d_max = density.min(), density.max()
            if d_max > d_min:
                alpha = (density - d_min) / (d_max - d_min)
            else:
                alpha = np.full_like(density, 0.5)

            y_ens = alpha * y_ml + (1.0 - alpha) * y_dl
            fold_r2s.append(r2_score(y_true, y_ens))
            fold_rmses.append(np.sqrt(mean_squared_error(y_true, y_ens)))

        if fold_r2s:
            result[strat] = {
                "r2_mean": np.mean(fold_r2s),
                "r2_std": np.std(fold_r2s),
                "rmse_mean": np.mean(fold_rmses),
                "rmse_std": np.std(fold_rmses),
            }

    return result


DISPLAY_NAMES = {
    "Site": "Site",
    "Grid": "Region",
    "Spatiotemporal_block": "Spatiotemporal",
}

MODEL_DISPLAY = {
    "biglightgbm": "LightGBM",
    "bigxgboost": "XGBoost",
    "bigcatboost": "CatBoost",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "mlp_early": "MLP",
    "mlp_tearly": "MLP",
    "mlp": "MLP",
    "resnet_early": "ResNet",
    "resnet_tearly": "ResNet",
    "resnet": "ResNet",
    "transformer_early": "Transformer",
    "transformer_tearly": "Transformer",
    "transformer": "Transformer",
    "linear_regression": "LinearRegression",
    "random_forest": "RandomForest",
    "decision_tree": "DecisionTree",
    "elasticnet": "ElasticNet",
    "pls": "PLS",
    "extra_trees": "ExtraTrees",
    "ridge": "Ridge",
}


def _fmt_metric(mean, std, show_std):
    """Format a metric value as 'mean ± std' or 'mean' string."""
    if np.isnan(mean):
        return "—"
    if show_std and not np.isnan(std):
        return f"{mean:.4f} ± {std:.4f}" if mean < 1 else f"{mean:.2f} ± {std:.2f}"
    return f"{mean:.4f}" if mean < 1 else f"{mean:.2f}"


def _render_mpl_table(cell_text, header_row0, header_row1, n_cols, n_strat,
                      section_label_rows, bold_cells, has_cost, cost_extra):
    """Render a matplotlib table figure used by both strategy and distillation tables."""
    all_rows = [header_row0, header_row1] + cell_text
    n_total = len(all_rows)
    n_models = len(cell_text)

    n_extra = n_cols - (1 + cost_extra + 2 * n_strat)
    fig_w = 2.6 * n_strat + 3.2 + (1.4 if has_cost else 0) + 1.0 * n_extra
    fig_h = 0.36 * n_total + 0.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    if has_cost:
        model_w = 0.18
        cost_w = 0.08
        extra_w = 0.05 * n_extra
        data_w = (1.0 - model_w - cost_w - extra_w) / (2 * n_strat) if n_strat else 0.1
        col_widths = [model_w, cost_w] + [data_w] * (2 * n_strat) + [extra_w / max(n_extra, 1)] * n_extra
    else:
        extra_w = 0.05 * n_extra
        data_w = (1.0 - 0.22 - extra_w) / (2 * n_strat) if n_strat else 0.1
        col_widths = [0.22] + [data_w] * (2 * n_strat) + [extra_w / max(n_extra, 1)] * n_extra

    tbl = ax.table(
        cellText=all_rows,
        colWidths=col_widths,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.4)

    hdr_color = "#4472C4"
    for j in range(n_cols):
        c0 = tbl[0, j]
        c1 = tbl[1, j]
        c0.set_facecolor(hdr_color)
        c0.set_text_props(fontweight="bold", fontsize=9.5, color="white")
        c0.set_edgecolor(hdr_color)
        c0.set_linewidth(0)
        c1.set_facecolor("#5b9bd5")
        c1.set_text_props(fontweight="bold", fontsize=8.5, color="white")
        c1.set_edgecolor("#5b9bd5")
        c1.set_linewidth(0)

    for j in range(n_cols):
        tbl[0, j].set_edgecolor("white")
        tbl[0, j].set_linewidth(0)

    strat_j_start = 1 + cost_extra
    for idx in range(n_strat):
        j_r2 = strat_j_start + idx * 2
        j_rmse = j_r2 + 1
        c_rmse = tbl[0, j_rmse]
        c_rmse.get_text().set_text("")

    tbl[0, 0].get_text().set_text("")
    if has_cost:
        tbl[0, 1].get_text().set_text("")

    data_row_counter = 0
    for i in range(n_models):
        tbl_row = i + 2
        is_section = i in section_label_rows

        for j in range(n_cols):
            cell = tbl[tbl_row, j]

            if is_section:
                cell.set_facecolor("#d6dce4")
                cell.set_edgecolor("#999999")
                cell.set_linewidth(0.8)
                if j == 0:
                    cell.set_text_props(fontweight="bold", fontsize=9,
                                        fontstyle="italic")
                else:
                    cell.get_text().set_text("")
            else:
                cell.set_edgecolor("#dddddd")
                cell.set_linewidth(0.5)

                if data_row_counter % 2 == 0:
                    cell.set_facecolor("#f2f5fa")
                else:
                    cell.set_facecolor("white")

                if j == 0:
                    cell.set_text_props(fontweight="bold", fontsize=8.5)
                    cell._loc = "left"
                elif (i, j) in bold_cells:
                    cell.set_text_props(fontweight="bold", color="#c00000",
                                        fontsize=8.5)
                else:
                    cell.set_text_props(fontsize=8.5)

        if not is_section:
            data_row_counter += 1

    plt.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    return fig


def _save_table(save_path, raw_df, latex):
    """Save raw DataFrame as .csv and LaTeX as .tex."""
    if not save_path:
        return
    import os
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    base, _ = os.path.splitext(save_path)
    raw_df.to_csv(f"{base}.csv")
    with open(f"{base}.tex", "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"Saved: {base}.csv, {base}.tex")


def _build_bold_cells(cell_text, sections, n_cols, cost_extra):
    """Find cells that should be bolded (best per section per column)."""
    bold_cells = set()
    data_start = 1 + cost_extra
    for j in range(data_start, n_cols):
        higher_better = ((j - data_start) % 2 == 0)
        for sec_indices in sections:
            bv, bi = None, None
            for i in sec_indices:
                v = cell_text[i][j]
                if not v or v == "—":
                    continue
                val = float(v.split(" ±")[0])
                if bv is None or \
                   (higher_better and val > bv) or \
                   (not higher_better and val < bv):
                    bv = val
                    bi = i
            if bi is not None:
                bold_cells.add((bi, j))
    return bold_cells


def _build_sections(table_rows, section_label_rows):
    """Group data row indices into sections separated by section labels."""
    sections = []
    current_section = []
    for i, row in enumerate(table_rows):
        if i in section_label_rows:
            if current_section:
                sections.append(current_section)
            current_section = []
        else:
            current_section.append(i)
    if current_section:
        sections.append(current_section)
    return sections


def _build_latex(table_rows, strategies, section_label_rows, has_cost,
                 cost_dict, bold_best, best_per_section, cost_extra,
                 caption, label):
    """Build a LaTeX table string from table rows."""
    n_strat = len(strategies)
    strat_names = [DISPLAY_NAMES.get(s, s) for s in strategies]

    header_top = " & ".join(
        ["\\multicolumn{2}{c}{" + s + "}" for s in strat_names]
    )
    strat_col_start = 2 + cost_extra
    cmidrules = " ".join(
        [f"\\cmidrule(lr){{{strat_col_start + 2*i}-{strat_col_start + 2*i + 1}}}"
         for i in range(n_strat)]
    )
    header_sub = " & ".join(["R$^2$ & RMSE"] * n_strat)

    if has_cost:
        tex_hdr1 = f"Method & Est. Cost & {header_top} \\\\"
        tex_hdr2 = f" & & {header_sub} \\\\"
        col_spec = "lc" + "cc" * n_strat
    else:
        tex_hdr1 = f"Method & {header_top} \\\\"
        tex_hdr2 = f" & {header_sub} \\\\"
        col_spec = "l" + "cc" * n_strat

    def _tex(val_str, col_name, row_idx):
        if val_str == "—":
            return "—"
        txt = val_str.replace("±", "$\\pm$")
        if bold_best and (row_idx, col_name) in best_per_section:
            txt = "\\textbf{" + txt + "}"
        return txt

    body_lines = []
    for i, row in enumerate(table_rows):
        if i in section_label_rows:
            ncol = 1 + cost_extra + 2 * n_strat
            line = f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{{row['Model']}}}}}" + " \\\\"
        else:
            cells = [row["Model"]]
            if has_cost:
                c = cost_dict.get(row["Model"]) if cost_dict else None
                cells.append(f"{c:.1f}$\\times$" if c is not None else "—")
            for strat in strategies:
                sd = DISPLAY_NAMES.get(strat, strat)
                cells.append(_tex(row.get(f"{sd} R²", "—"), f"{sd} R²", i))
                cells.append(_tex(row.get(f"{sd} RMSE", "—"), f"{sd} RMSE", i))
            line = " & ".join(cells) + " \\\\"
        body_lines.append(line)

    return f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\small
\\begin{{tabular}}{{{col_spec}}}
\\toprule
{tex_hdr1}
{cmidrules}
{tex_hdr2}
\\midrule
{chr(10).join(body_lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


def _build_cell_text(table_rows, strategies, section_label_rows, has_cost,
                     cost_dict):
    """Build cell_text matrix for matplotlib rendering."""
    cell_text = []
    for i, row in enumerate(table_rows):
        r = [row["Model"]]
        if has_cost:
            if i in section_label_rows:
                r.append("")
            else:
                c = cost_dict.get(row["Model"]) if cost_dict else None
                r.append(f"{c:.2f}x" if c is not None else "—")
        for strat in strategies:
            sd = DISPLAY_NAMES.get(strat, strat)
            r.append(row.get(f"{sd} R²", "—"))
            r.append(row.get(f"{sd} RMSE", "—"))
        cell_text.append(r)
    return cell_text


def _build_mpl_headers(strategies, has_cost, cost_extra):
    """Build header rows for the matplotlib table."""
    strat_names = [DISPLAY_NAMES.get(s, s) for s in strategies]
    n_strat = len(strategies)
    header_row0 = [""]
    if has_cost:
        header_row0.append("")
    for sname in strat_names:
        header_row0.append(sname)
        header_row0.append("")
    header_row1 = ["Method"]
    if has_cost:
        header_row1.append("Est. Cost")
    header_row1 += ["R²", "RMSE"] * n_strat
    return header_row0, header_row1


def strategy_ablation_table(project_path, models=None, strategies=None,
                            early_stop=True, show_std=True,
                            bold_best=True, cost_dict=None, save_path=None):
    """
    Build an Ensemble strategy comparison table.

    Rows: Ensemble (MLs), Ensemble (DLs), Ensemble (DL+ML),
          Ensemble (ML+OG), Ensemble (DL+OG), Ensemble (DL+ML+OG).
    All ensembles are simple averaging.

    Parameters
    ----------
    project_path : str
        Path to the 10-Fold-ablation results directory.
    models : list of str, optional
        Model CSV stems for DL/ML. If None, uses a default set.
    strategies : list of str, optional
        Default: ['Site', 'Grid', 'Spatiotemporal_block'].
    early_stop : bool
        Use early-stop DL models (_tearly).
    show_std : bool
        If True, format as "mean ± std".
    bold_best : bool
        If True, bold the best value per column.
    cost_dict : dict or None
        If provided, adds an "Est. Cost" column.
    save_path : str or None
        If provided, saves .csv and .tex to this base path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    latex : str
    raw_df : pd.DataFrame
    """
    if strategies is None:
        strategies = ["Site", "Grid", "Spatiotemporal_block"]

    early_suffix = "_tearly" if early_stop else ""

    if models is None:
        dl = [f"mlp{early_suffix}", f"resnet{early_suffix}", f"transformer{early_suffix}"] \
            if early_stop else ["mlp", "resnet", "transformer"]
        models = dl + ["biglightgbm", "bigcatboost", "bigxgboost"]

    dl_list = [m for m in models if m.replace("_tearly", "").replace("_early", "") in
               {"mlp", "bigmlp", "resnet", "transformer", "gtransformer"}]
    ml_list = [m for m in models if m in
               {"biglightgbm", "bigxgboost", "bigcatboost", "lightgbm", "xgboost", "catboost"}]

    _ens_2_best, ens_2_avg = _compute_ensemble_baselines(
        project_path, dl_list, ml_list, strategies)
    ens_mls = _compute_ensemble_mls(project_path, ml_list, strategies)
    _ens_ogml_best, ens_ogml_avg = _compute_ensemble_og_ml(
        project_path, dl_list, ml_list, strategies)
    _ens_dlog_best, ens_dlog_avg = _compute_ensemble_dl_og(
        project_path, dl_list, ml_list, strategies)
    _ens_og_best, ens_og_avg = _compute_ensemble_3model(
        project_path, dl_list, ml_list, strategies, mode="avg")
    ens_og_mls = _compute_ensemble_og_mls(
        project_path, dl_list, ml_list, strategies)
    ens_dl_mls = _compute_ensemble_dl_mls(
        project_path, dl_list, ml_list, strategies)
    ens_dl_dl = _compute_ensemble_dl_dl(project_path, dl_list, strategies)
    ens_dl_mls_og = _compute_ensemble_dl_mls_og(
        project_path, dl_list, ml_list, strategies)

    def _add_row(label, data_dict, rows):
        if not data_dict:
            return
        row = {"Model": label}
        for strat in strategies:
            sd = DISPLAY_NAMES.get(strat, strat)
            if strat in data_dict:
                b = data_dict[strat]
                row[f"{sd} R²"] = _fmt_metric(b["r2_mean"], b["r2_std"], show_std)
                row[f"{sd} RMSE"] = _fmt_metric(b["rmse_mean"], b["rmse_std"], show_std)
            else:
                row[f"{sd} R²"] = "—"
                row[f"{sd} RMSE"] = "—"
        rows.append(row)

    table_rows = []
    section_label_rows = set()

    _add_row("Ensemble (MLs)", ens_mls, table_rows)
    _add_row("Ensemble (DL+ML)", ens_2_avg, table_rows)
    _add_row("Ensemble (MLs+DL)", ens_dl_mls, table_rows)
    _add_row("Ensemble (ML+OG)", ens_ogml_avg, table_rows)
    _add_row("Ensemble (MLs+OG)", ens_og_mls, table_rows)
    _add_row("Ensemble (DL+DL)", ens_dl_dl, table_rows)
    _add_row("Ensemble (DL+OG)", ens_dlog_avg, table_rows)
    _add_row("Ensemble (DL+ML+OG)", ens_og_avg, table_rows)
    _add_row("Ensemble (DL+MLs+OG)", ens_dl_mls_og, table_rows)

    raw_df = pd.DataFrame(table_rows).set_index("Model")

    has_cost = cost_dict is not None
    n_strat = len(strategies)
    cost_extra = 1 if has_cost else 0
    n_cols = 1 + cost_extra + 2 * n_strat

    sections = _build_sections(table_rows, section_label_rows)

    best_per_section = {}
    if bold_best:
        for strat in strategies:
            sd = DISPLAY_NAMES.get(strat, strat)
            for col_name, higher_better in [(f"{sd} R²", True), (f"{sd} RMSE", False)]:
                best_val, best_i = None, None
                for i in range(len(table_rows)):
                    v = table_rows[i].get(col_name, "—")
                    if not v or v == "—":
                        continue
                    val = float(v.split(" ±")[0])
                    if best_val is None or \
                       (higher_better and val > best_val) or \
                       (not higher_better and val < best_val):
                        best_val = val
                        best_i = i
                if best_i is not None:
                    best_per_section[(best_i, col_name)] = True

    latex = _build_latex(
        table_rows, strategies, section_label_rows, has_cost,
        cost_dict, bold_best, best_per_section, cost_extra,
        caption="Ensemble strategy comparison. Best values are in \\textbf{bold}.",
        label="tab:ensemble_strategy",
    )

    cell_text = _build_cell_text(
        table_rows, strategies, section_label_rows, has_cost, cost_dict)
    bold_cells = _build_bold_cells(
        cell_text, sections, n_cols, cost_extra) if bold_best else set()
    header_row0, header_row1 = _build_mpl_headers(strategies, has_cost, cost_extra)

    fig = _render_mpl_table(
        cell_text, header_row0, header_row1, n_cols, n_strat,
        section_label_rows, bold_cells, has_cost, cost_extra)

    _save_table(save_path, raw_df, latex)

    return fig, latex, raw_df


# ---------------------------------------------------------------------------
# Helpers: group aggregates and Δ% columns
# ---------------------------------------------------------------------------

def _group_aggregate(raw_df, csv_names, strategies, show_std=True):
    """
    Compute avg and best R²/RMSE across a group of models per strategy.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Output of ``load_strategy_summary``.
    csv_names : list of str
        CSV stems in the group.
    strategies : list of str

    Returns
    -------
    avg_dict, best_dict : dict
        ``{strategy: {r2_mean, r2_std, rmse_mean, rmse_std}}``
    """
    avg_dict, best_dict = {}, {}
    for strat in strategies:
        sub = raw_df[(raw_df["model"].isin(csv_names)) &
                     (raw_df["strategy"] == strat)]
        if len(sub) == 0:
            continue
        avg_dict[strat] = {
            "r2_mean": sub["r2_mean"].mean(),
            "r2_std": sub["r2_std"].mean(),
            "rmse_mean": sub["rmse_mean"].mean() if "rmse_mean" in sub else np.nan,
            "rmse_std": sub["rmse_std"].mean() if "rmse_std" in sub else np.nan,
        }
        best_row = sub.loc[sub["r2_mean"].idxmax()]
        best_dict[strat] = {
            "r2_mean": best_row["r2_mean"],
            "r2_std": best_row["r2_std"],
            "rmse_mean": best_row.get("rmse_mean", np.nan),
            "rmse_std": best_row.get("rmse_std", np.nan),
        }
    return avg_dict, best_dict


def _add_delta_columns(table_rows, section_label_rows):
    """
    For each non-section row that has Site R² and Region/Spatiotemporal R²,
    compute Δ_Region = (Site - Region) / Site × 100 and
    Δ_ST = (Site - Spatiotemporal) / Site × 100.

    Modifies ``table_rows`` in-place by adding 'Δ Region' and 'Δ ST' keys.
    """
    for i, row in enumerate(table_rows):
        if i in section_label_rows:
            row["Δ Region"] = ""
            row["Δ ST"] = ""
            continue

        site_str = row.get("Site R²", "—")
        region_str = row.get("Region R²", "—")
        st_str = row.get("Spatiotemporal R²", "—")

        def _parse(s):
            if not s or s == "—":
                return None
            return float(s.split(" ±")[0])

        site_val = _parse(site_str)
        region_val = _parse(region_str)
        st_val = _parse(st_str)

        if site_val is not None and site_val > 0 and region_val is not None:
            row["Δ Region"] = f"{(site_val - region_val) / site_val * 100:.1f}%"
        else:
            row["Δ Region"] = "—"

        if site_val is not None and site_val > 0 and st_val is not None:
            row["Δ ST"] = f"{(site_val - st_val) / site_val * 100:.1f}%"
        else:
            row["Δ ST"] = "—"


# ---------------------------------------------------------------------------
# OG + Baseline Detail Table
# ---------------------------------------------------------------------------

def og_baseline_table(
    project_path,
    strategies=None,
    show_std=True,
    bold_best=True,
    early_stop=True,
    cost_dict=None,
    save_path=None,
    extra_ml_models=None,
):
    """
    Build a comprehensive table listing every individual baseline model,
    every OG combination (DL_base+ML), plus group aggregates and Δ columns.

    Row order:
        --- DL Baselines ---
        MLP, ResNet, Transformer
        --- DL Aggregates ---
        DL_avg, DL_best
        --- Boosting Baselines ---
        LightGBM, CatBoost, XGBoost
        --- Boosting Aggregates ---
        Boost_avg, Boost_best
        --- Other ML Baselines ---
        LinearRegression, RandomForest, DecisionTree, ...
        --- OG Combinations (9 pairs) ---
        OG (MLP+LightGBM), ...
        --- OG Aggregates ---
        OG_avg, OG_best

    Parameters
    ----------
    project_path : str
        Data directory containing all model CSVs (e.g. ``…/10-Fold-ablation``).
    extra_ml_models : list of str, optional
        Additional ML model CSV stems to include (loaded from the same
        ``project_path``).  Models whose CSV does not exist are silently
        skipped.  Default includes linear_regression, random_forest, etc.
    strategies : list of str, optional
    show_std, bold_best, early_stop, cost_dict, save_path :
        Same as before.

    Returns (fig, latex, raw_df).
    """
    if strategies is None:
        strategies = ["Site", "Grid", "Spatiotemporal_block"]

    if extra_ml_models is None:
        extra_ml_models = [
            "linear_regression", "random_forest", "decision_tree",
            "elasticnet", "pls", "extra_trees", "ridge",
        ]

    early_suffix = "_tearly" if early_stop else ""

    dl_names = [f"mlp{early_suffix}", f"resnet{early_suffix}",
                f"transformer{early_suffix}"]
    boost_names = ["biglightgbm", "bigcatboost", "bigxgboost"]

    dl_bases = [d.replace("_tearly", "").replace("_early", "")
                for d in dl_names]
    og_names = [f"{dlb}+{ml}" for dlb in dl_bases for ml in boost_names]

    OG_DISPLAY = {}
    for dlb in dl_bases:
        for ml in boost_names:
            dl_nice = MODEL_DISPLAY.get(dlb, dlb)
            ml_nice = MODEL_DISPLAY.get(ml, ml.replace("big", ""))
            OG_DISPLAY[f"{dlb}+{ml}"] = f"OG ({dl_nice}+{ml_nice})"

    all_csv = dl_names + boost_names + og_names + extra_ml_models
    raw = load_strategy_summary(project_path, all_csv, strategies)

    og_best = _find_best_og_per_strategy(project_path, strategies)
    og_avg = _find_avg_og_per_strategy(project_path, strategies)

    dl_avg, dl_best = _group_aggregate(raw, dl_names, strategies, show_std)
    boost_avg, boost_best = _group_aggregate(raw, boost_names, strategies, show_std)

    # --- row builders ---
    def _add_csv_row(label, csv_name, rows):
        row = {"Model": label}
        for strat in strategies:
            sub = raw[(raw["model"] == csv_name) & (raw["strategy"] == strat)]
            sd = DISPLAY_NAMES.get(strat, strat)
            if len(sub) > 0:
                s = sub.iloc[0]
                row[f"{sd} R²"] = _fmt_metric(s["r2_mean"], s["r2_std"],
                                               show_std)
                row[f"{sd} RMSE"] = _fmt_metric(
                    s.get("rmse_mean", np.nan),
                    s.get("rmse_std", np.nan), show_std)
            else:
                row[f"{sd} R²"] = "—"
                row[f"{sd} RMSE"] = "—"
        rows.append(row)

    def _add_dict_row(label, data_dict, rows):
        if not data_dict:
            return
        row = {"Model": label}
        for strat in strategies:
            sd = DISPLAY_NAMES.get(strat, strat)
            if strat in data_dict:
                b = data_dict[strat]
                row[f"{sd} R²"] = _fmt_metric(b["r2_mean"], b["r2_std"],
                                               show_std)
                row[f"{sd} RMSE"] = _fmt_metric(
                    b.get("rmse_mean", np.nan),
                    b.get("rmse_std", np.nan), show_std)
            else:
                row[f"{sd} R²"] = "—"
                row[f"{sd} RMSE"] = "—"
        rows.append(row)

    # --- Build table rows ---
    table_rows = []
    section_label_rows = set()

    # DL Baselines
    section_label_rows.add(len(table_rows))
    table_rows.append({"Model": "DL Baselines"})
    for csv_name in dl_names:
        display = MODEL_DISPLAY.get(csv_name, csv_name)
        _add_csv_row(display, csv_name, table_rows)

    # DL Aggregates
    section_label_rows.add(len(table_rows))
    table_rows.append({"Model": "DL Aggregates"})
    _add_dict_row("DL_avg", dl_avg, table_rows)
    _add_dict_row("DL_best", dl_best, table_rows)

    # Boosting Baselines
    section_label_rows.add(len(table_rows))
    table_rows.append({"Model": "Boosting Baselines"})
    for csv_name in boost_names:
        display = MODEL_DISPLAY.get(csv_name, csv_name)
        _add_csv_row(display, csv_name, table_rows)

    # Boosting Aggregates
    section_label_rows.add(len(table_rows))
    table_rows.append({"Model": "Boosting Aggregates"})
    _add_dict_row("Boost_avg", boost_avg, table_rows)
    _add_dict_row("Boost_best", boost_best, table_rows)

    # Other ML Baselines (only those with data)
    other_ml_with_data = [m for m in extra_ml_models
                          if len(raw[raw["model"] == m]) > 0]
    if other_ml_with_data:
        section_label_rows.add(len(table_rows))
        table_rows.append({"Model": "Other ML Baselines"})
        for csv_name in other_ml_with_data:
            display = MODEL_DISPLAY.get(csv_name, csv_name)
            _add_csv_row(display, csv_name, table_rows)

    # OG Combinations
    section_label_rows.add(len(table_rows))
    table_rows.append({"Model": "OG Combinations"})
    for og_csv in og_names:
        display = OG_DISPLAY.get(og_csv, og_csv)
        _add_csv_row(display, og_csv, table_rows)

    # OG Aggregates
    section_label_rows.add(len(table_rows))
    table_rows.append({"Model": "OG Aggregates"})
    _add_dict_row("OG_avg", og_avg, table_rows)
    _add_dict_row("OG_best", og_best, table_rows)

    # --- Delta columns ---
    _add_delta_columns(table_rows, section_label_rows)

    # --- Build raw DataFrame ---
    raw_df = pd.DataFrame(table_rows)
    raw_df = raw_df[~raw_df.index.isin(section_label_rows)]
    raw_df = raw_df.set_index("Model")

    # --- Formatting for output ---
    has_cost = cost_dict is not None
    n_strat = len(strategies)
    cost_extra = 1 if has_cost else 0

    # Column layout: Model | (Cost) | R² RMSE Δ | R² RMSE Δ | R² RMSE Δ
    # But we use the standard R²/RMSE build path and append Δ columns
    # in the CSV/LaTeX outputs separately.

    # For mpl table and LaTeX, use the standard pipeline then add Δ cols
    n_cols_base = 1 + cost_extra + 2 * n_strat

    sections = _build_sections(table_rows, section_label_rows)

    best_per_section = {}
    if bold_best:
        all_data = [i for i in range(len(table_rows))
                    if i not in section_label_rows]
        for strat in strategies:
            sd = DISPLAY_NAMES.get(strat, strat)
            for col_name, higher_better in [(f"{sd} R²", True),
                                            (f"{sd} RMSE", False)]:
                best_val, best_i = None, None
                for i in all_data:
                    v = table_rows[i].get(col_name, "—")
                    if not v or v == "—":
                        continue
                    val = float(v.split(" ±")[0])
                    if best_val is None or \
                       (higher_better and val > best_val) or \
                       (not higher_better and val < best_val):
                        best_val = val
                        best_i = i
                if best_i is not None:
                    best_per_section[(best_i, col_name)] = True

    latex = _build_latex(
        table_rows, strategies, section_label_rows, has_cost,
        cost_dict, bold_best, best_per_section, cost_extra,
        caption="Individual model and OG combination performance. "
                "Best values are in \\textbf{bold}.",
        label="tab:og_baseline_detail",
    )

    cell_text = _build_cell_text(
        table_rows, strategies, section_label_rows, has_cost, cost_dict)

    # Append Δ columns to cell_text
    for i, row in enumerate(table_rows):
        cell_text[i].append(row.get("Δ Region", "—"))
        cell_text[i].append(row.get("Δ ST", "—"))

    n_cols = n_cols_base + 2  # +2 for Δ Region and Δ ST

    bold_cells = _build_bold_cells(
        cell_text, sections, n_cols_base, cost_extra) if bold_best else set()

    # Bold lowest Δ (best = lowest degradation)
    if bold_best:
        for delta_j in [n_cols_base, n_cols_base + 1]:
            for sec_indices in sections:
                bv, bi = None, None
                for i in sec_indices:
                    v = cell_text[i][delta_j]
                    if not v or v == "—" or v == "":
                        continue
                    val = float(v.replace("%", ""))
                    if bv is None or val < bv:
                        bv = val
                        bi = i
                if bi is not None:
                    bold_cells.add((bi, delta_j))

    header_row0, header_row1 = _build_mpl_headers(
        strategies, has_cost, cost_extra)
    header_row0 += ["", ""]
    header_row1 += ["Δ Region", "Δ ST"]

    fig = _render_mpl_table(
        cell_text, header_row0, header_row1, n_cols, n_strat,
        section_label_rows, bold_cells, has_cost, cost_extra)

    _save_table(save_path, raw_df, latex)

    return fig, latex, raw_df


# ---------------------------------------------------------------------------
# Distillation Ablation Table
# ---------------------------------------------------------------------------

DISTILL_DISPLAY = {
    "kd": "Vanilla KD",
    "ba": "Born Again",
    "st": "Self-Training",
    "dfkd": "Data-Free KD",
    "smote": "SMOTE-R",
}


def _compute_cost_from_timing(timing, smote_data_ratio=1.2):
    """Compute relative cost dict from raw timing primitives."""
    t_nn = timing["nn_epochs"] * timing["t_nn_per_epoch"]
    t_ml = timing["t_ml_train"]
    t_ml_inf = timing["t_ml_infer"]
    n_ep = timing["nn_epochs"]
    t_og_epoch = timing["t_nn_per_epoch"] + t_ml_inf
    t_og_total = t_ml + n_ep * t_og_epoch

    return {
        "MLP":                  1.0,
        "CatBoost":             t_ml / t_nn,
        "OG":                   t_og_total / t_nn,
        "Vanilla KD":           (t_ml + t_nn) / t_nn,
        "Born Again":           2.0,
        "Self-Training":        3.0,
        "Data-Free KD":         (t_ml + t_nn) / t_nn,
        "SMOTE-R":              smote_data_ratio,
        "Ensemble (DLs)":       3.0,
        "Ensemble (MLs)":       (3 * t_ml) / t_nn,
        "Ensemble (DLs+MLs)":    (3 * t_nn + 3 * t_ml) / t_nn,
        "Ensemble (DLs+MLs+OG)": (3 * t_nn + 3 * t_ml + t_og_total) / t_nn,
        "Ensemble (DL+ML)":     (t_ml + t_nn) / t_nn,
        "Ensemble (ML+OG)":     (2 * t_ml + t_nn + n_ep * t_ml_inf) / t_nn,
        "Ensemble (DL+OG)":     (t_ml + t_nn + t_og_total) / t_nn,
        "Ensemble (DL+ML+OG)":  (t_ml + t_nn + t_og_total) / t_nn,
        "Ensemble (MLs+OG)":    (3 * t_ml + t_og_total) / t_nn,
        "Ensemble (MLs+DL)":    (3 * t_ml + t_nn) / t_nn,
        "Ensemble (DL+DL)":     2.0,
        "Ensemble (DL+MLs+OG)": (t_nn + 3 * t_ml + t_og_total) / t_nn,
        "Ensemble (DL+ML+Vanilla KD)":   (t_ml + t_nn + t_ml + t_nn) / t_nn,
        "Ensemble (DL+ML+Born Again)":   (t_ml + t_nn + 2 * t_nn) / t_nn,
        "Ensemble (DL+ML+Self-Training)": (t_ml + t_nn + 3 * t_nn) / t_nn,
        "Ensemble (DL+ML+Data-Free KD)": (t_ml + t_nn + t_ml + t_nn) / t_nn,
        "Ensemble (DL+ML+SMOTE-R)":      (t_ml + t_nn + smote_data_ratio * t_nn) / t_nn,
    }


def estimate_training_cost(data_path=None, X=None, y=None, n_samples=500_000,
                           nn_epochs=200, n_benchmark_epochs=5,
                           smote_data_ratio=1.2, load_cache=True):
    """
    Benchmark NN and ML training speed, then compute estimated relative
    training cost for every method in the distillation ablation table.

    Cache is saved as ``benchmark_timing.json`` in the same directory
    as *data_path*.  When *load_cache* is True and the cache exists,
    it is loaded directly without re-running the benchmark.

    Parameters
    ----------
    data_path : str, optional
        Path to the pkl data file.  Cache is stored next to this file.
        Defaults to ``local_overfit/data/training/…_2019.pkl``.
    X, y : array-like, optional
        Training data (used instead of *data_path* if given).
    n_samples : int
        Samples to draw for benchmarking (default 500 000).
    nn_epochs : int
        Assumed total NN epochs per training pass (default 200).
    n_benchmark_epochs : int
        MLP epochs for per-epoch timing (default 5).
    smote_data_ratio : float
        Data expansion factor for SMOTE-R.
    load_cache : bool
        If True (default), return cached timing when available.
        Set False to force a fresh benchmark.

    Returns
    -------
    cost_dict : dict
        ``{display_name: relative_cost}`` for all table rows.
    timing_info : dict
        Raw timing data in seconds.
    """
    import time, json, sys
    from pathlib import Path as _P

    _root = str(_P(__file__).resolve().parents[3])
    if _root not in sys.path:
        sys.path.insert(0, _root)

    if data_path is None and X is None:
        data_path = str(_P(__file__).resolve().parents[1] / "data" / "training"
                        / "Site_Observation_with_feature_2019.pkl")

    cache_path = _P(data_path).resolve().parent / "benchmark_timing.json" \
        if data_path else None

    # ---- try cache first ----
    if load_cache and cache_path and cache_path.exists():
        with open(cache_path) as f:
            timing = json.load(f)
        cost = _compute_cost_from_timing(timing, smote_data_ratio)
        t_nn_total = timing["nn_epochs"] * timing["t_nn_per_epoch"]
        print(f"Loaded timing cache from {cache_path}")
        print(f"  MLP:      {timing['t_nn_per_epoch']:.2f} s/epoch  ->  "
              f"{timing['nn_epochs']} ep = {t_nn_total:.1f} s")
        print(f"  MLP inference:      {timing['t_nn_infer']:.2f} s")
        print(f"  CatBoost train:     {timing['t_ml_train']:.2f} s  "
              f"({timing['t_ml_train']/t_nn_total:.1%} of MLP)")
        print(f"  CatBoost inference: {timing['t_ml_infer']:.2f} s")
        print(f"\nRelative costs (MLP = 1.0x):")
        for name, c in cost.items():
            print(f"  {name:25s} {c:.2f}x")
        return cost, timing

    # ---- run benchmark ----
    from OG_transformer.model.mlp import MLPRegressor as _MLP
    from catboost import CatBoostRegressor as _CB

    if data_path is not None:
        import pickle as pkl
        from OG_transformer import general_feature_engineering
        print(f"Loading data from {data_path} ...")
        df = pkl.load(open(data_path, 'rb'))
        df["time"] = pd.to_datetime(df["time"])
        for col in df.select_dtypes(include=["float64", "int64"]).columns:
            df[col] = df[col].astype("float32")
        feature_list = [
            'time', 'TROPOMI_ozone', 'OMI_ozone', 'tco3', 'blh', 't2m',
            'sp', 'WS10', 'population', 'no2', 'DSR', 'strd', 'r_1000',
            'lai_hv', 'pev', 'ssro', 't_975', 't_925', 'tp', 'tsn',
            'stl1', 'sampling_height', 'latitude', 'longitude',
        ]
        X, y = general_feature_engineering(df, feature_list, "Ozone",
                                           weight_dir=None)
        del df
    elif X is None or y is None:
        raise ValueError("Provide (X, y) or data_path.")

    X_np = (X.values if hasattr(X, 'values') else np.asarray(X)).astype(np.float32)
    y_np = (y.values.ravel() if hasattr(y, 'values')
            else np.asarray(y).ravel()).astype(np.float32)

    if n_samples is not None and len(X_np) > n_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_np), n_samples, replace=False)
        X_np = X_np[idx]
        y_np = y_np[idx]

    n_feat = X_np.shape[1]
    print(f"Benchmarking on {len(X_np):,} samples x {n_feat} features ...")

    print("[1/5] GPU warmup ...")
    _w = _MLP(num_features=n_feat, hidden_dims=[512, 256, 128],
              dropout=0.1, lr=1e-3, epochs=1, batch_size=1024)
    _w.fit(X_np, y_np, progress_bar=False, early_stopping=False,
           monitor_curve=False)

    print(f"[2/5] MLP benchmark ({n_benchmark_epochs} epochs) ...")
    _b = _MLP(num_features=n_feat, hidden_dims=[512, 256, 128],
              dropout=0.1, lr=1e-3, epochs=n_benchmark_epochs, batch_size=1024)
    t0 = time.perf_counter()
    _b.fit(X_np, y_np, progress_bar=True, early_stopping=False,
           monitor_curve=False)
    t_nn = time.perf_counter() - t0
    t_nn_per_epoch = t_nn / n_benchmark_epochs

    print("[3/5] MLP inference benchmark ...")
    t0 = time.perf_counter()
    _b.predict(X_np)
    t_nn_infer = time.perf_counter() - t0

    print("[4/5] CatBoost training benchmark ...")
    _cb = _CB(iterations=1000, depth=7, learning_rate=0.1, l2_leaf_reg=2,
              border_count=64, task_type="GPU", verbose=0, random_state=42)
    t0 = time.perf_counter()
    _cb.fit(X_np, y_np)
    t_ml = time.perf_counter() - t0

    print("[5/5] CatBoost inference benchmark ...")
    t0 = time.perf_counter()
    _cb.predict(X_np)
    t_ml_infer = time.perf_counter() - t0

    t_nn_total = nn_epochs * t_nn_per_epoch

    print(f"\n  MLP:      {t_nn_per_epoch:.2f} s/epoch  ->  {nn_epochs} ep = {t_nn_total:.1f} s")
    print(f"  MLP inference:      {t_nn_infer:.2f} s")
    print(f"  CatBoost train:     {t_ml:.2f} s  ({t_ml/t_nn_total:.1%} of MLP)")
    print(f"  CatBoost inference: {t_ml_infer:.2f} s")

    timing_info = {
        "t_nn_per_epoch": t_nn_per_epoch,
        "t_nn_infer": t_nn_infer,
        "t_ml_train": t_ml,
        "t_ml_infer": t_ml_infer,
        "nn_epochs": nn_epochs,
        "n_samples": len(X_np),
        "n_features": n_feat,
    }

    cost = _compute_cost_from_timing(timing_info, smote_data_ratio)

    print("\nEstimated relative costs (MLP = 1.0x):")
    for name, c in cost.items():
        print(f"  {name:25s} {c:.2f}x")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(timing_info, f, indent=2)
        print(f"\nTiming cached to {cache_path}")

    return cost, timing_info


def distillation_ablation_table(
    project_path,
    nn_name="mlp",
    ml_name="bigcatboost",
    strategies=None,
    show_std=True,
    bold_best=True,
    early_stop=True,
    cost_dict=None,
    save_path=None,
):
    """
    Build a single-model comparison table of distillation / rebalancing
    methods against OG and base models.

    Rows shown (in order):
        1. Base DL model (e.g. MLP)
        2. Base ML model (e.g. CatBoost)
        3. Vanilla KD
        4. Born Again
        5. Self-Training
        6. Data-Free KD
        7. SMOTE-R
        8. OG (best)

    Parameters
    ----------
    project_path : str
        Path to the 10-Fold-ablation results directory.
    nn_name : str
        Base NN model name (default 'mlp').
    ml_name : str
        Base ML model name (default 'bigcatboost').
    strategies : list of str, optional
        Default: ['Site', 'Grid', 'Spatiotemporal_block'].
    show_std : bool
        If True, format as "mean +/- std".
    bold_best : bool
        If True, bold the best value per strategy column.
    early_stop : bool
        If True, use '_early' variant for DL models.
    cost_dict : dict or None
        If provided, adds an "Est. Cost" column.  Keys are display names
        (e.g. ``"MLP"``, ``"OG"``), values are relative costs (float).
        Generate with :func:`estimate_training_cost`.

    Returns
    -------
    fig : matplotlib.figure.Figure
    latex : str
    raw_df : pd.DataFrame
    """
    if strategies is None:
        strategies = ["Site", "Grid", "Spatiotemporal_block"]

    early_suffix = "_tearly" if early_stop else ""
    nn_csv = f"{nn_name}{early_suffix}"
    ml_csv = ml_name

    distill_prefixes = ["kd", "ba", "st", "dfkd", "smote"]
    distill_csv_names = []
    for p in distill_prefixes:
        if p in ("kd", "dfkd"):
            base = f"{p}_{nn_name}+{ml_name}"
        else:
            base = f"{p}_{nn_name}"
        distill_csv_names.append(f"{base}{early_suffix}" if early_stop else base)

    all_csv = [nn_csv, ml_csv] + distill_csv_names
    raw = load_strategy_summary(project_path, all_csv, strategies)

    og_best = _find_best_og_per_strategy(project_path, strategies)

    def _add_row(label, data_dict, rows):
        if not data_dict:
            return
        row = {"Model": label}
        for strat in strategies:
            sd = DISPLAY_NAMES.get(strat, strat)
            if strat in data_dict:
                b = data_dict[strat]
                row[f"{sd} R²"] = _fmt_metric(b["r2_mean"], b["r2_std"], show_std)
                row[f"{sd} RMSE"] = _fmt_metric(b["rmse_mean"], b["rmse_std"], show_std)
            else:
                row[f"{sd} R²"] = "—"
                row[f"{sd} RMSE"] = "—"
        rows.append(row)

    def _add_csv_row(label, csv_name, rows):
        row = {"Model": label}
        for strat in strategies:
            sub = raw[(raw["model"] == csv_name) & (raw["strategy"] == strat)]
            sd = DISPLAY_NAMES.get(strat, strat)
            if len(sub) > 0:
                s = sub.iloc[0]
                row[f"{sd} R²"] = _fmt_metric(s["r2_mean"], s["r2_std"], show_std)
                row[f"{sd} RMSE"] = _fmt_metric(s["rmse_mean"], s["rmse_std"], show_std)
            else:
                row[f"{sd} R²"] = "—"
                row[f"{sd} RMSE"] = "—"
        rows.append(row)

    table_rows = []
    section_label_rows = set()

    nn_display = MODEL_DISPLAY.get(nn_csv, nn_name.upper())
    ml_display = MODEL_DISPLAY.get(ml_csv, ml_name.replace("big", "").capitalize())

    # ---- Single Model only (no section header — flat table) ----
    _add_csv_row(nn_display, nn_csv, table_rows)
    _add_csv_row(ml_display, ml_csv, table_rows)
    for prefix, csv_name in zip(distill_prefixes, distill_csv_names):
        label = DISTILL_DISPLAY.get(prefix, prefix)
        _add_csv_row(label, csv_name, table_rows)
    _add_row("OG", og_best, table_rows)

    raw_df = pd.DataFrame(table_rows).set_index("Model")

    has_cost = cost_dict is not None
    n_strat = len(strategies)
    cost_extra = 1 if has_cost else 0
    n_cols = 1 + cost_extra + 2 * n_strat

    sections = _build_sections(table_rows, section_label_rows)

    best_per_section = {}
    if bold_best:
        all_indices = list(range(len(table_rows)))
        for strat in strategies:
            sd = DISPLAY_NAMES.get(strat, strat)
            for col_name, higher_better in [(f"{sd} R²", True), (f"{sd} RMSE", False)]:
                best_val, best_i = None, None
                for i in all_indices:
                    v = table_rows[i].get(col_name, "—")
                    if not v or v == "—":
                        continue
                    val = float(v.split(" ±")[0])
                    if best_val is None or \
                       (higher_better and val > best_val) or \
                       (not higher_better and val < best_val):
                        best_val = val
                        best_i = i
                if best_i is not None:
                    best_per_section[(best_i, col_name)] = True

    latex = _build_latex(
        table_rows, strategies, section_label_rows, has_cost,
        cost_dict, bold_best, best_per_section, cost_extra,
        caption="Comparison of knowledge distillation and data augmentation methods against OG. "
                "All methods use MLP as the student model. Best values are in \\textbf{bold}.",
        label="tab:distillation_comparison",
    )

    cell_text = _build_cell_text(
        table_rows, strategies, section_label_rows, has_cost, cost_dict)
    bold_cells = _build_bold_cells(
        cell_text, sections, n_cols, cost_extra) if bold_best else set()
    header_row0, header_row1 = _build_mpl_headers(strategies, has_cost, cost_extra)

    fig = _render_mpl_table(
        cell_text, header_row0, header_row1, n_cols, n_strat,
        section_label_rows, bold_cells, has_cost, cost_extra)

    _save_table(save_path, raw_df, latex)

    return fig, latex, raw_df


# ---------------------------------------------------------------------------
# OG Ablation Table
# ---------------------------------------------------------------------------

OG_ABL_DISPLAY = {
    "mlp_tearly": "Baseline LV (MLP)",
    "mlp_bigcatboost_200": "Baseline LV (200 ep)",
    "bigcatboost_hvhv": "HV → HV",
    "mlp_bigcatboost_hvlv": "HV → LV",
    "mlp_bigcatboost_density": "HV → LV + density",
    "mlp_bigcatboost_noise": "HV → LV + noise",
}


def OG_ablation_table(
    project_path,
    strategies=None,
    show_std=True,
    bold_best=True,
    early_stop=True,
    cost_dict=None,
    save_path=None,
):
    """
    Build an OG component ablation table.

    Rows:
        1. Baseline LV (MLP)
        2. Baseline LV (200 ep)        — rules out "just training longer"
        3. HV → HV                     — teacher-student, both HV
        4. HV → LV                     — pseudo-labels only
        5. HV → LV + density           — + density-aware sampling
        6. HV → LV + noise             — + noise injection
        7. Full OG (best)               — all components

    Parameters
    ----------
    project_path : str
        Path to the 10-Fold-ablation results directory.
    strategies, show_std, bold_best, early_stop, cost_dict, save_path :
        Same semantics as ``distillation_ablation_table``.

    Returns
    -------
    fig, latex, raw_df
    """
    if strategies is None:
        strategies = ["Site", "Grid", "Spatiotemporal_block"]

    early_suffix = "_tearly" if early_stop else ""
    mlp_csv = f"mlp{early_suffix}"

    ablation_csv_names = [
        mlp_csv,
        "mlp_bigcatboost_200",
        "bigcatboost_hvhv",
        "mlp_bigcatboost_hvlv",
        "mlp_bigcatboost_density",
        "mlp_bigcatboost_noise",
    ]

    raw = load_strategy_summary(project_path, ablation_csv_names, strategies)
    og_best = _find_best_og_per_strategy(project_path, strategies)

    def _add_csv_row(label, csv_name, rows):
        row = {"Model": label}
        for strat in strategies:
            sub = raw[(raw["model"] == csv_name) & (raw["strategy"] == strat)]
            sd = DISPLAY_NAMES.get(strat, strat)
            if len(sub) > 0:
                s = sub.iloc[0]
                row[f"{sd} R²"] = _fmt_metric(s["r2_mean"], s["r2_std"], show_std)
                row[f"{sd} RMSE"] = _fmt_metric(
                    s.get("rmse_mean", np.nan), s.get("rmse_std", np.nan), show_std)
            else:
                row[f"{sd} R²"] = "—"
                row[f"{sd} RMSE"] = "—"
        rows.append(row)

    def _add_dict_row(label, data_dict, rows):
        if not data_dict:
            return
        row = {"Model": label}
        for strat in strategies:
            sd = DISPLAY_NAMES.get(strat, strat)
            if strat in data_dict:
                b = data_dict[strat]
                row[f"{sd} R²"] = _fmt_metric(b["r2_mean"], b["r2_std"], show_std)
                row[f"{sd} RMSE"] = _fmt_metric(
                    b.get("rmse_mean", np.nan), b.get("rmse_std", np.nan), show_std)
            else:
                row[f"{sd} R²"] = "—"
                row[f"{sd} RMSE"] = "—"
        rows.append(row)

    table_rows = []
    section_label_rows = set()

    for csv_name in ablation_csv_names:
        label = OG_ABL_DISPLAY.get(csv_name, csv_name)
        _add_csv_row(label, csv_name, table_rows)
    _add_dict_row("Full OG", og_best, table_rows)

    raw_df = pd.DataFrame(table_rows).set_index("Model")

    has_cost = cost_dict is not None
    n_strat = len(strategies)
    cost_extra = 1 if has_cost else 0
    n_cols = 1 + cost_extra + 2 * n_strat

    sections = _build_sections(table_rows, section_label_rows)

    best_per_section = {}
    if bold_best:
        all_indices = list(range(len(table_rows)))
        for strat in strategies:
            sd = DISPLAY_NAMES.get(strat, strat)
            for col_name, higher_better in [(f"{sd} R²", True), (f"{sd} RMSE", False)]:
                best_val, best_i = None, None
                for i in all_indices:
                    v = table_rows[i].get(col_name, "—")
                    if not v or v == "—":
                        continue
                    val = float(v.split(" ±")[0])
                    if best_val is None or \
                       (higher_better and val > best_val) or \
                       (not higher_better and val < best_val):
                        best_val = val
                        best_i = i
                if best_i is not None:
                    best_per_section[(best_i, col_name)] = True

    latex = _build_latex(
        table_rows, strategies, section_label_rows, has_cost,
        cost_dict, bold_best, best_per_section, cost_extra,
        caption="OG component ablation. Each row adds one component "
                "to the teacher--student pipeline. Best values are in \\textbf{bold}.",
        label="tab:og_ablation",
    )

    cell_text = _build_cell_text(
        table_rows, strategies, section_label_rows, has_cost, cost_dict)
    bold_cells = _build_bold_cells(
        cell_text, sections, n_cols, cost_extra) if bold_best else set()
    header_row0, header_row1 = _build_mpl_headers(strategies, has_cost, cost_extra)

    fig = _render_mpl_table(
        cell_text, header_row0, header_row1, n_cols, n_strat,
        section_label_rows, bold_cells, has_cost, cost_extra)

    _save_table(save_path, raw_df, latex)

    return fig, latex, raw_df
