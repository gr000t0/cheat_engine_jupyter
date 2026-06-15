#!/usr/bin/env python3
"""
Predict engine-likeness for a new PGN.

Input:
  - PGN file with one or more games
  - trained Bayesian move-level model: bayesian_logreg_model.joblib
  - optional Bayesian game-level calibrator: bayesian_game_calibrator.joblib

Output:
  - move-level predictions CSV
  - game/color-level engine-rate summary CSV

Example:
  python predict_pgn_engine.py --pgn my_game.pgn \
      --model bayesian_logreg_model.joblib \
      --calibrator bayesian_game_calibrator.joblib
"""

import argparse
import io
import json
import math
import os
import re
import subprocess
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import joblib
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from tqdm import tqdm


# -----------------------------
# Defaults from your pipeline
# -----------------------------
STOCKFISH_PATH_DEFAULT = "/usr/games/stockfish"
LC0_PATH_DEFAULT = os.path.expanduser("~/.local/bin/lc0")
MAIA_WEIGHTS_DIR_DEFAULT = os.path.expanduser("~/Games/maia-chess/maia_weights")

AVAILABLE_MAIA_ELOS = [1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]

OPENING_MOVES_PER_COLOR = 10
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}
PIECE_NAMES = {
    0: "none",
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}
PIECE_CATEGORIES = ["none", "pawn", "knight", "bishop", "rook", "queen", "king"]


# -----------------------------
# Numeric helpers
# -----------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logit(x, eps=1e-4):
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x))


# -----------------------------
# PGN + chess features
# -----------------------------
def parse_elo(value, default=1500):
    try:
        if value is None or value == "?":
            return default
        v = int(value)
        if v < 0:
            return default
        return v
    except Exception:
        return default


def material_score(board, color):
    return sum(len(board.pieces(piece_type, color)) * value for piece_type, value in PIECE_VALUES.items())


def extract_chess_features(board, move):
    if move not in board.legal_moves:
        return {
            "piece_type": "none",
            "captured_piece_type": "none",
            "is_capture": np.nan,
            "gives_check": np.nan,
            "is_castling": np.nan,
            "is_promotion": np.nan,
            "legal_moves_before": np.nan,
            "material_balance": np.nan,
        }

    color = board.turn
    opponent = not color

    piece = board.piece_at(move.from_square)
    piece_type_id = piece.piece_type if piece else 0

    if board.is_en_passant(move):
        captured_piece_type_id = chess.PAWN
    else:
        captured_piece = board.piece_at(move.to_square)
        captured_piece_type_id = captured_piece.piece_type if captured_piece else 0

    return {
        "piece_type": PIECE_NAMES[piece_type_id],
        "captured_piece_type": PIECE_NAMES[captured_piece_type_id],
        "is_capture": int(board.is_capture(move)),
        "gives_check": int(board.gives_check(move)),
        "is_castling": int(board.is_castling(move)),
        "is_promotion": int(move.promotion is not None),
        "legal_moves_before": board.legal_moves.count(),
        "material_balance": material_score(board, color) - material_score(board, opponent),
    }


def pgn_file_to_moves(pgn_path, default_white_elo=1500, default_black_elo=1500):
    rows = []
    with open(pgn_path, "r", encoding="utf-8", errors="replace") as f:
        game_id = 0
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            white_elo = parse_elo(game.headers.get("WhiteElo"), default_white_elo)
            black_elo = parse_elo(game.headers.get("BlackElo"), default_black_elo)
            result = game.headers.get("Result", "*")

            board = game.board()
            white_move_i = 0
            black_move_i = 0

            for ply, move in enumerate(game.mainline_moves(), start=1):
                color = "white" if board.turn == chess.WHITE else "black"
                fen_before = board.fen()
                move_uci = move.uci()
                move_san = board.san(move)

                chess_features = extract_chess_features(board, move)

                if color == "white":
                    player_elo = white_elo
                    opponent_elo = black_elo
                    source = "opening" if white_move_i < OPENING_MOVES_PER_COLOR else "unknown"
                    white_move_i += 1
                else:
                    player_elo = black_elo
                    opponent_elo = white_elo
                    source = "opening" if black_move_i < OPENING_MOVES_PER_COLOR else "unknown"
                    black_move_i += 1

                row = {
                    "game_id": game_id,
                    "ply": ply,
                    "color_name": color,
                    "color": color,  # one-hot encoded later; color_name is kept for grouping
                    "move_number": board.fullmove_number,
                    "fen_before": fen_before,
                    "move_uci": move_uci,
                    "move_san": move_san,
                    "player_elo": player_elo,
                    "opponent_elo": opponent_elo,
                    "result": result,
                    "source": source,
                }
                row.update(chess_features)
                rows.append(row)
                board.push(move)

            game_id += 1

    if not rows:
        raise ValueError(f"No games found in PGN: {pgn_path}")

    df = pd.DataFrame(rows)
    return add_one_hot_columns(df)


def add_one_hot_columns(df):
    df["piece_type"] = pd.Categorical(df["piece_type"], categories=PIECE_CATEGORIES)
    df["captured_piece_type"] = pd.Categorical(df["captured_piece_type"], categories=PIECE_CATEGORIES)
    df["color"] = pd.Categorical(df["color"], categories=["white", "black"])

    return pd.get_dummies(
        df,
        columns=["piece_type", "captured_piece_type", "color"],
        prefix=["piece", "captured", "color"],
        dtype=int,
    )


# -----------------------------
# Stockfish features
# -----------------------------
def score_to_cp(score, pov_color):
    return score.pov(pov_color).score(mate_score=100000)


def stockfishiness_from_cp_loss(cp_loss, scale=100):
    if cp_loss is None or pd.isna(cp_loss):
        return None
    cp_loss = max(0, min(float(cp_loss), 1000))
    return float(math.exp(-cp_loss / scale))


def empty_stockfish_features():
    return {
        "stockfishiness": None,
        "sf_cp_loss": None,
        "sf_rank_topk": None,
        "sf_top1": None,
        "sf_gap_best_second": None,
        "sf_eval_best": None,
        "sf_eval_played": None,
    }


def stockfish_features(engine, fen_before, move_uci, depth=10, multipv=5):
    try:
        board = chess.Board(fen_before)
        played_move = chess.Move.from_uci(move_uci)

        if played_move not in board.legal_moves:
            return empty_stockfish_features()

        color = board.turn

        infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
        if isinstance(infos, dict):
            infos = [infos]

        top_moves = []
        for info in infos:
            pv = info.get("pv")
            if not pv:
                continue
            candidate_move = pv[0]
            candidate_score = score_to_cp(info["score"], color)
            top_moves.append((candidate_move, candidate_score))

        if not top_moves:
            return empty_stockfish_features()

        best_move, best_score = top_moves[0]
        second_score = top_moves[1][1] if len(top_moves) > 1 else None

        rank = multipv + 1
        for i, (candidate_move, _) in enumerate(top_moves, start=1):
            if candidate_move == played_move:
                rank = i
                break

        played_info = engine.analyse(
            board,
            chess.engine.Limit(depth=depth),
            root_moves=[played_move],
        )
        played_score = score_to_cp(played_info["score"], color)

        cp_loss = max(0, best_score - played_score)
        return {
            "stockfishiness": stockfishiness_from_cp_loss(cp_loss),
            "sf_cp_loss": cp_loss,
            "sf_rank_topk": rank,
            "sf_top1": int(played_move == best_move),
            "sf_gap_best_second": None if second_score is None else best_score - second_score,
            "sf_eval_best": best_score,
            "sf_eval_played": played_score,
        }
    except Exception as e:
        out = empty_stockfish_features()
        out["sf_error"] = str(e)
        return out


def add_stockfish_features(df, stockfish_path, depth=10, multipv=5, threads=4, hash_mb=512):
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Threads": threads, "Hash": hash_mb})

    features = []
    try:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Stockfish"):
            features.append(stockfish_features(engine, row["fen_before"], row["move_uci"], depth=depth, multipv=multipv))
    finally:
        engine.quit()

    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(features).reset_index(drop=True)], axis=1)
    out["stockfishiness_clean"] = out["stockfishiness"].fillna(0.0).clip(0.0, 1.0)
    out["sf_cp_loss_clean"] = out["sf_cp_loss"].fillna(1000).clip(0, 1000)
    out["sf_rank_topk_clean"] = out["sf_rank_topk"].fillna(multipv + 1)
    return out


# -----------------------------
# Maia features
# -----------------------------
def pick_maia_elo(player_elo):
    if pd.isna(player_elo) or int(player_elo) < 0:
        return 1500
    player_elo = int(player_elo)
    return min(AVAILABLE_MAIA_ELOS, key=lambda e: abs(e - player_elo))


def send(proc, cmd):
    proc.stdin.write(cmd + "\n")
    proc.stdin.flush()


def read_until(proc, marker):
    lines = []
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        line = line.rstrip()
        lines.append(line)
        if marker in line:
            break
    return lines


def start_maia_process(lc0_path, weights_path):
    proc = subprocess.Popen(
        [lc0_path, f"--weights={weights_path}", "--verbose-move-stats", "--threads=1"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    send(proc, "uci")
    read_until(proc, "uciok")
    send(proc, "isready")
    read_until(proc, "readyok")
    return proc


def stop_maia_process(proc):
    try:
        send(proc, "quit")
    except Exception:
        pass
    try:
        proc.terminate()
    except Exception:
        pass


def get_maia_prob_for_move(proc, fen, move_uci):
    send(proc, f"position fen {fen}")
    send(proc, "go nodes 1")

    pattern = re.compile(rf"info string\s+{re.escape(move_uci)}\s+.*?\(P:\s*([0-9.]+)%\)")
    prob = None

    while True:
        line = proc.stdout.readline()
        if not line:
            break
        line = line.rstrip()

        match = pattern.search(line)
        if match:
            prob = float(match.group(1)) / 100.0

        if line.startswith("bestmove"):
            break

    return prob


def add_maia_features(df, lc0_path, maia_weights_dir):
    df = df.copy()
    df["maia_elo_bucket"] = df["player_elo"].apply(pick_maia_elo)
    df["maia_prob_played"] = np.nan

    for maia_elo, group in df.groupby("maia_elo_bucket"):
        weights_path = os.path.join(maia_weights_dir, f"maia-{maia_elo}.pb.gz")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Maia weights not found: {weights_path}")

        print(f"Using Maia {maia_elo}: {weights_path} for {len(group)} moves")
        proc = start_maia_process(lc0_path, weights_path)
        try:
            for idx, row in tqdm(group.iterrows(), total=len(group), desc=f"Maia {maia_elo}"):
                prob = get_maia_prob_for_move(proc, row["fen_before"], row["move_uci"])
                df.at[idx, "maia_prob_played"] = prob
        finally:
            stop_maia_process(proc)

    df["maia_prob_missing"] = df["maia_prob_played"].isna().astype(int)
    df["humanlikeness"] = df["maia_prob_played"]
    df["humanlikeness_clean"] = df["humanlikeness"].fillna(0.0).clip(0.0, 1.0)
    df["log_maia_prob_played"] = np.log(df["humanlikeness_clean"].clip(lower=1e-9))
    return df


# -----------------------------
# Model feature prep + prediction
# -----------------------------
def ensure_model_features(df, features):
    df = df.copy()

    if "player_elo_clean" not in df.columns and "player_elo" in df.columns:
        df["player_elo_clean"] = df["player_elo"].where((df["player_elo"].notna()) & (df["player_elo"] >= 0), 1500)

    if "opponent_elo_clean" not in df.columns and "opponent_elo" in df.columns:
        df["opponent_elo_clean"] = df["opponent_elo"].where((df["opponent_elo"].notna()) & (df["opponent_elo"] >= 0), 1500)

    if "elo_diff" not in df.columns and {"player_elo_clean", "opponent_elo_clean"}.issubset(df.columns):
        df["elo_diff"] = df["player_elo_clean"] - df["opponent_elo_clean"]

    # Derived combined features, if model expects them.
    if "engine_human_gap" in features and "engine_human_gap" not in df.columns:
        df["engine_human_gap"] = df.get("stockfishiness_clean", 0) - df.get("humanlikeness_clean", 0)

    if "stockfish_low_human" in features and "stockfish_low_human" not in df.columns:
        df["stockfish_low_human"] = df.get("stockfishiness_clean", 0) * (1.0 - df.get("humanlikeness_clean", 0))

    # Add missing one-hot/model columns as zeros.
    for col in features:
        if col not in df.columns:
            print(f"Warning: missing model feature {col}; filling with 0")
            df[col] = 0

    df[features] = df[features].replace([np.inf, -np.inf], np.nan)
    for col in features:
        if df[col].isna().any():
            median = df[col].median()
            if pd.isna(median):
                median = 0
            df[col] = df[col].fillna(median)

    return df


def predict_moves(df, model_path):
    bundle = joblib.load(model_path)
    features = bundle["features"]
    scaler = bundle["scaler"]
    beta0_samples = bundle["beta0_samples"]
    beta_samples = bundle["beta_samples"]

    df = ensure_model_features(df, features)
    X = scaler.transform(df[features].values)

    logits_samples = beta0_samples[:, None] + beta_samples @ X.T
    prob_samples = sigmoid(logits_samples)

    df["p_engine_mean"] = prob_samples.mean(axis=0)
    df["p_engine_q05"] = np.quantile(prob_samples, 0.05, axis=0)
    df["p_engine_q95"] = np.quantile(prob_samples, 0.95, axis=0)
    return df


# -----------------------------
# Aggregation + calibration
# -----------------------------
def aggregate_group(group, thresholds=(0.5, 0.7), alpha=1.0, beta=1.0):
    probs = group["p_engine_mean"].dropna().values
    n = len(probs)
    if n == 0:
        row = {"n_moves": 0, "pred_engine_rate": np.nan, "k_soft": np.nan, "beta_a_post": np.nan, "beta_b_post": np.nan, "beta_engine_rate_estimate": np.nan}
        for t in thresholds:
            row[f"beta_p_gt_{t}"] = np.nan
        return pd.Series(row)

    k_soft = float(np.sum(probs))
    a_post = alpha + k_soft
    b_post = beta + n - k_soft
    row = {
        "n_moves": n,
        "pred_engine_rate": float(np.mean(probs)),
        "k_soft": k_soft,
        "beta_a_post": a_post,
        "beta_b_post": b_post,
        "beta_engine_rate_estimate": a_post / (a_post + b_post),
    }
    for t in thresholds:
        row[f"beta_p_gt_{t}"] = 1.0 - beta_dist.cdf(t, a_post, b_post)
    return pd.Series(row)


def apply_bayesian_logit_calibration(summary, calibrator_path, thresholds=(0.5, 0.7)):
    if not calibrator_path or not os.path.exists(calibrator_path):
        print("No Bayesian game calibrator found/provided; skipping calibrated Gamma.")
        return summary

    cal = joblib.load(calibrator_path)
    trace = cal["trace"]
    eps = cal.get("eps", 1e-4)
    a_samples = trace.posterior["a"].values.reshape(-1)
    b_samples = trace.posterior["b"].values.reshape(-1)

    raw = summary["pred_engine_rate"].values.astype(float)
    s = logit(raw, eps=eps)
    z = a_samples[:, None] + b_samples[:, None] * s[None, :]
    gamma_samples = sigmoid(z)

    summary["gamma_cal_mean"] = gamma_samples.mean(axis=0)
    summary["gamma_cal_q05"] = np.quantile(gamma_samples, 0.05, axis=0)
    summary["gamma_cal_q95"] = np.quantile(gamma_samples, 0.95, axis=0)

    for t in thresholds:
        summary[f"gamma_cal_p_gt_{t}"] = (gamma_samples > t).mean(axis=0)

    return summary


def aggregate_predictions(df, thresholds=(0.5, 0.7), include_opening=False, calibrator_path=None):
    df_eval = df.copy()
    if not include_opening and "source" in df_eval.columns:
        df_eval = df_eval[df_eval["source"] != "opening"].copy()

    if len(df_eval) == 0:
        raise ValueError("No moves left for aggregation. Try --include-opening or use a longer PGN.")

    summary = (
        df_eval.groupby(["game_id", "color_name"], group_keys=False)
        .apply(lambda g: aggregate_group(g, thresholds=thresholds), include_groups=False)
        .reset_index()
    )
    summary = apply_bayesian_logit_calibration(summary, calibrator_path, thresholds=thresholds)
    return summary


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="PGN -> engine-rate Gamma and P(Gamma > threshold).")
    parser.add_argument("--pgn", required=True, help="Input PGN file")
    parser.add_argument("--model", default="bayesian_logreg_model.joblib", help="Move-level Bayesian logistic regression model")
    parser.add_argument("--calibrator", default="bayesian_game_calibrator.joblib", help="Optional Bayesian logit game-level calibrator")
    parser.add_argument("--out-prefix", default="new_pgn", help="Output prefix")

    parser.add_argument("--stockfish", default=STOCKFISH_PATH_DEFAULT)
    parser.add_argument("--sf-depth", type=int, default=10)
    parser.add_argument("--sf-multipv", type=int, default=5)
    parser.add_argument("--sf-threads", type=int, default=4)
    parser.add_argument("--sf-hash", type=int, default=512)

    parser.add_argument("--lc0", default=LC0_PATH_DEFAULT)
    parser.add_argument("--maia-weights-dir", default=MAIA_WEIGHTS_DIR_DEFAULT)

    parser.add_argument("--white-elo", type=int, default=1500, help="Default White Elo if PGN has no WhiteElo header")
    parser.add_argument("--black-elo", type=int, default=1500, help="Default Black Elo if PGN has no BlackElo header")
    parser.add_argument("--include-opening", action="store_true", help="Include first 10 moves per color in aggregation")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.7], help="Thresholds for P(Gamma > t)")

    args = parser.parse_args()

    print("Parsing PGN and extracting chess features...")
    moves = pgn_file_to_moves(args.pgn, default_white_elo=args.white_elo, default_black_elo=args.black_elo)

    print("Adding Stockfish features...")
    moves = add_stockfish_features(
        moves,
        stockfish_path=args.stockfish,
        depth=args.sf_depth,
        multipv=args.sf_multipv,
        threads=args.sf_threads,
        hash_mb=args.sf_hash,
    )

    print("Adding Maia features...")
    moves = add_maia_features(moves, lc0_path=args.lc0, maia_weights_dir=args.maia_weights_dir)

    print("Predicting move-level engine probabilities...")
    moves = predict_moves(moves, model_path=args.model)

    print("Aggregating per game/color...")
    summary = aggregate_predictions(
        moves,
        thresholds=tuple(args.thresholds),
        include_opening=args.include_opening,
        calibrator_path=args.calibrator,
    )

    moves_out = f"{args.out_prefix}_move_predictions.csv"
    summary_out = f"{args.out_prefix}_game_color_summary.csv"
    json_out = f"{args.out_prefix}_summary.json"

    moves.to_csv(moves_out, index=False)
    summary.to_csv(summary_out, index=False)

    # Compact JSON for quick use.
    records = summary.to_dict(orient="records")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print("\n=== Game/color summary ===")
    print(summary.to_string(index=False))

    print("\nSaved:")
    print(moves_out)
    print(summary_out)
    print(json_out)

    print("\nInterpretation:")
    print("  pred_engine_rate          = raw mean of move-level p_engine")
    print("  beta_engine_rate_estimate = Beta soft-count smoothed rate")
    print("  beta_p_gt_0.7             = P(Gamma > 0.7) from raw Beta soft-count aggregation")
    print("  gamma_cal_mean            = Bayesian-logit calibrated Gamma, if calibrator exists")
    print("  gamma_cal_p_gt_0.7        = P(calibrated Gamma > 0.7) from calibration posterior")


if __name__ == "__main__":
    main()