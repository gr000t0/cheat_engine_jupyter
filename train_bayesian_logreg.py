import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import joblib

from pymc.sampling.jax import sample_numpyro_nuts

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
)


INPUT_CSV = "moves_all_sus.csv"
MODEL_OUT = "bayesian_logreg_model.joblib"

ENGINE_FEATURES = [
    # Stockfish v2 features
    "stockfishiness_clean",
    "sf_cp_loss_clean",
    "sf_rank_topk_clean",
    "sf_top1",
    "sf_gap_best_second",

    # Maia feature
    "log_maia_prob_played",
]

META_FEATURES = [
    "player_elo_clean",
    "move_number",
]

CHESS_NUMERIC_FEATURES = [
    "is_capture",
    "gives_check",
    "is_castling",
    "is_promotion",
    "legal_moves_before",
    "material_balance",
]

CHESS_ONEHOT_FEATURES = [
    # moved piece
    # Referenzkategorie weggelassen: piece_pawn
    "piece_knight",
    "piece_bishop",
    "piece_rook",
    "piece_queen",
    "piece_king",

    # captured piece
    # Referenzkategorie weggelassen: captured_none
    "captured_pawn",
    "captured_knight",
    "captured_bishop",
    "captured_rook",
    "captured_queen",

    # color
    # Referenzkategorie weggelassen: color_white
    "color_black",
]

FEATURES = (
    ENGINE_FEATURES
    + META_FEATURES
    + CHESS_NUMERIC_FEATURES
    + CHESS_ONEHOT_FEATURES
)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ensure_clean_features(df):
    """
    Falls einzelne Clean-Spalten noch nicht existieren,
    werden sie hier erzeugt.
    """

    if "stockfishiness_clean" not in df.columns and "stockfishiness" in df.columns:
        df["stockfishiness_clean"] = df["stockfishiness"].fillna(0.0).clip(0.0, 1.0)

    if "sf_cp_loss_clean" not in df.columns and "sf_cp_loss" in df.columns:
        df["sf_cp_loss_clean"] = df["sf_cp_loss"].fillna(1000).clip(0, 1000)

    if "sf_rank_topk_clean" not in df.columns and "sf_rank_topk" in df.columns:
        # Falls sf_rank_topk fehlt, wird später mit Median/0 gefüllt.
        df["sf_rank_topk_clean"] = df["sf_rank_topk"].fillna(df["sf_rank_topk"].max())

    if "sf_gap_best_second" in df.columns:
        df["sf_gap_best_second"] = df["sf_gap_best_second"].fillna(0)

    if "sf_top1" in df.columns:
        df["sf_top1"] = df["sf_top1"].fillna(0).astype(int)

    if "log_maia_prob_played" not in df.columns and "maia_prob_played" in df.columns:
        df["log_maia_prob_played"] = np.log(
            df["maia_prob_played"].fillna(0).clip(lower=1e-9)
        )

    if "player_elo_clean" not in df.columns and "player_elo" in df.columns:
        df["player_elo_clean"] = df["player_elo"].where(
            (df["player_elo"].notna()) & (df["player_elo"] >= 0),
            1500,
        )

    return df


def load_data():
    df = pd.read_csv(INPUT_CSV)

    df = ensure_clean_features(df)

    # Opening optional rausnehmen
    if "source" in df.columns:
        df = df[df["source"] != "lichess_opening"].copy()

    required = ["label", "game_id"]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    # Falls manche One-Hot-Spalten in einem Sample fehlen, als 0 ergänzen.
    for col in FEATURES:
        if col not in df.columns:
            print(f"Warning: missing feature column {col}, filling with 0")
            df[col] = 0

    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)

    for col in FEATURES:
        if df[col].isna().any():
            median = df[col].median()
            if pd.isna(median):
                median = 0
            df[col] = df[col].fillna(median)

    df["label"] = df["label"].astype(int)

    return df


def main():
    df = load_data()

    print("\n=== Dataset check ===")
    print("Rows:", len(df))
    print("Games:", df["game_id"].nunique())
    print("\nMoves per game:")
    print(df.groupby("game_id").size().describe())

    if "color_black" in df.columns:
        print("\nMoves per game/color:")
        print(df.groupby(["game_id", "color_black"]).size().describe())

    print("Using features:")
    for feature in FEATURES:
        print(f"  - {feature}")

    X = df[FEATURES].values
    y = df["label"].values
    groups = df["game_id"].values

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42,
    )

    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train_raw = X[train_idx]
    X_test_raw = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    n_features = X_train.shape[1]

    with pm.Model() as model:
        beta0 = pm.Normal("beta0", mu=0, sigma=2)
        beta = pm.Normal("beta", mu=0, sigma=1, shape=n_features)

        eta = beta0 + pm.math.dot(X_train, beta)
        p = pm.math.sigmoid(eta)

        pm.Bernoulli("y_obs", p=p, observed=y_train)

        trace = sample_numpyro_nuts(
                draws=1000,
                tune=1000,
                chains=4,
                target_accept=0.9,
                random_seed=42,
                )
    print("\n=== Posterior Summary ===")
    print(az.summary(trace, var_names=["beta0", "beta"]))

    beta0_samples = trace.posterior["beta0"].values.reshape(-1)
    beta_samples = trace.posterior["beta"].values.reshape(-1, n_features)

    logits_samples = beta0_samples[:, None] + beta_samples @ X_test.T
    prob_samples = sigmoid(logits_samples)

    y_prob = prob_samples.mean(axis=0)
    y_pred = (y_prob >= 0.5).astype(int)

    print("\n=== Move-Level Evaluation ===")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, zero_division=0))
    print("Recall   :", recall_score(y_test, y_pred, zero_division=0))
    print("F1       :", f1_score(y_test, y_pred, zero_division=0))
    print("ROC-AUC  :", roc_auc_score(y_test, y_prob))
    print("Brier    :", brier_score_loss(y_test, y_prob))

    coef_df = pd.DataFrame({
        "feature": FEATURES,
        "beta_mean": beta_samples.mean(axis=0),
        "beta_q05": np.quantile(beta_samples, 0.05, axis=0),
        "beta_q95": np.quantile(beta_samples, 0.95, axis=0),
    }).sort_values("beta_mean", ascending=False)

    print("\n=== Coefficients ===")
    print(coef_df.to_string(index=False))

    test_out = df.iloc[test_idx].copy()
    test_out["p_engine_mean"] = y_prob
    test_out["p_engine_q05"] = np.quantile(prob_samples, 0.05, axis=0)
    test_out["p_engine_q95"] = np.quantile(prob_samples, 0.95, axis=0)
    test_out.to_csv("bayesian_logreg_test_predictions.csv", index=False)

    joblib.dump(
        {
            "features": FEATURES,
            "scaler": scaler,
            "trace": trace,
            "beta0_samples": beta0_samples,
            "beta_samples": beta_samples,
        },
        MODEL_OUT,
    )

    print(f"\nSaved model to {MODEL_OUT}")
    print("Saved predictions to bayesian_logreg_test_predictions.csv")


if __name__ == "__main__":
    main()