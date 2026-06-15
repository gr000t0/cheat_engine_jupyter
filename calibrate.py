import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

INPUT_CSV = "game_color_scores.csv"
OUTPUT_MODEL = "bayesian_game_calibrator.joblib"

MIN_MOVES = 10
EPS = 1e-4
RANDOM_STATE = 42


def logit(x):
    x = np.clip(x, EPS, 1 - EPS)
    return np.log(x / (1 - x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


df = pd.read_csv(INPUT_CSV)
df = df[df["n_moves"] >= MIN_MOVES].copy()

train_idx, test_idx = train_test_split(
    df.index,
    test_size=0.3,
    random_state=RANDOM_STATE,
)

train = df.loc[train_idx].copy()
test = df.loc[test_idx].copy()

x_train = logit(train["pred_engine_rate"].values)
y_train = logit(train["true_engine_rate"].values)

x_test = logit(test["pred_engine_rate"].values)
y_test_true = test["true_engine_rate"].values

with pm.Model() as cal_model:
    a = pm.Normal("a", mu=0, sigma=1)
    b = pm.Normal("b", mu=1, sigma=1)
    sigma = pm.HalfNormal("sigma", sigma=1)

    mu = a + b * x_train

    pm.Normal("obs", mu=mu, sigma=sigma, observed=y_train)

    trace = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,
        target_accept=0.9,
        random_seed=42,
    )

print(az.summary(trace, var_names=["a", "b", "sigma"]))

a_samples = trace.posterior["a"].values.reshape(-1)
b_samples = trace.posterior["b"].values.reshape(-1)

z_pred_samples = a_samples[:, None] + b_samples[:, None] * x_test[None, :]
p_pred_samples = sigmoid(z_pred_samples)

test["bayes_cal_mean"] = p_pred_samples.mean(axis=0)
test["bayes_cal_q05"] = np.quantile(p_pred_samples, 0.05, axis=0)
test["bayes_cal_q95"] = np.quantile(p_pred_samples, 0.95, axis=0)

print("\nRaw:")
print("MAE:", mean_absolute_error(y_test_true, test["pred_engine_rate"]))
print("R2 :", r2_score(y_test_true, test["pred_engine_rate"]))

print("\nBayesian logit calibrated:")
print("MAE:", mean_absolute_error(y_test_true, test["bayes_cal_mean"]))
print("R2 :", r2_score(y_test_true, test["bayes_cal_mean"]))

test.to_csv("game_color_scores_bayes_cal_test.csv", index=False)

joblib.dump(
    {
        "trace": trace,
        "eps": EPS,
        "min_moves": MIN_MOVES,
    },
    OUTPUT_MODEL,
)

print(f"Saved {OUTPUT_MODEL}")