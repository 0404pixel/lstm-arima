"""
compare_models.py — Head-to-head next-day S&P 500 forecasting:
LSTM vs ARIMA vs Moving Average.

This script reuses the leak-free evaluation framework of lsmttrad_v4.py and
arima_v2.py, but focuses on a single goal: put the next-day Close forecasts of
all three models on ONE chart for easy visual comparison, and report the same
metrics (RMSE / MAE / MAPE / directional accuracy) side by side.

Design notes
------------
• Same dataset, splits, TARGET_MODE, window size, and metrics as the two
  original scripts, so numbers are directly comparable.
• LSTM  — sequence model on the Close series (return target, then reconstructed
  to price via the previous actual Close: one-step-ahead, no compounding).
• ARIMA — univariate model on the target series, one-step-ahead walk-forward
  with an order grid-searched on train/val only (no test leakage).
• Moving Average — rolling mean of the last WINDOW_SIZE actual Close prices.
"""

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.arima.model import ARIMA

sns.set()
tf.random.set_seed(42)
np.random.seed(42)

# ─── DATASET ──────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_CANDIDATE_PATHS = [
    _HERE / 'SP500-2000-2015.csv',
    _HERE.parent / 'dataset' / 'SP500-2000-2015.csv',
]
DATASET_PATH = next((p for p in _CANDIDATE_PATHS if p.exists()), _CANDIDATE_PATHS[0])

# ─── CONFIGURATION (shared by all three models) ───────────────────────────────
# TARGET_MODE — what the models actually predict:
#   'price'      → the Close price directly
#   'return'     → daily simple return r_t = (P_t - P_{t-1}) / P_{t-1}
#   'log_return' → daily log return   r_t = ln(P_t / P_{t-1})
# For 'return'/'log_return' the price is reconstructed from the PREVIOUS actual
# close (P_t ≈ P_{t-1}·(1+r̂)). Because daily returns are tiny, the reconstructed
# price is dominated by P_{t-1} — this is WHY every model visually "tracks"
# yesterday's price. The honest skill lives in the return space (see the
# returns-comparison plot and directional accuracy), not the price plot.
TARGET_MODE: str   = 'log_return'   # 'price' | 'return' | 'log_return'
WINDOW_SIZE: int   = 30         # lookback for LSTM sequences and MA baseline
SPLIT_RATIO: float = 0.80       # fraction of data for train + validation
VAL_RATIO:   float = 0.10       # validation fraction within the train+val block
EPOCHS:      int   = 120
BATCH_SIZE:  int   = 32

# ARIMA order grid — searched on the training slice only (no test leakage)
CANDIDATE_ORDERS_PRICE = [
    (1, 0, 0), (2, 0, 0), (5, 0, 0),
    (0, 1, 0), (1, 1, 0), (2, 1, 0), (5, 1, 0),
    (0, 1, 1), (1, 1, 1), (2, 1, 1), (5, 1, 1),
    (1, 1, 2), (2, 1, 2),
]
CANDIDATE_ORDERS_RETURN = [
    (p, 0, q)
    for p, q in itertools.product(range(6), range(4))
    if p + q > 0
]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))
    mae  = float(np.mean(np.abs(predicted - actual)))
    mape = float(np.mean(np.abs((actual - predicted) / np.maximum(np.abs(actual), 1e-8))) * 100.0)
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}


def directional_accuracy(
    actual:      np.ndarray,
    predicted:   np.ndarray,
    prev_actual: np.ndarray,
) -> float:
    actual_dir = np.sign(actual    - prev_actual)
    pred_dir   = np.sign(predicted - prev_actual)
    non_flat   = actual_dir != 0
    if non_flat.sum() == 0:
        return 0.0
    return float(np.mean(actual_dir[non_flat] == pred_dir[non_flat]) * 100.0)


def build_target_series(close_values: np.ndarray, target_mode: str) -> np.ndarray:
    n = len(close_values)
    if target_mode == 'price':
        return close_values.copy()
    if target_mode == 'return':
        out = np.zeros(n, dtype='float32')
        out[1:] = np.diff(close_values) / close_values[:-1]
        return out
    if target_mode == 'log_return':
        out = np.zeros(n, dtype='float32')
        out[1:] = np.log(close_values[1:] / close_values[:-1])
        return out
    raise ValueError(f'Unknown TARGET_MODE: {target_mode}')


def reconstruct_prices(
    predicted_targets: np.ndarray,
    target_mode:       str,
    prev_prices:       np.ndarray,
) -> np.ndarray:
    if target_mode == 'price':
        return predicted_targets.astype('float32')
    if target_mode == 'return':
        return (prev_prices * (1.0 + predicted_targets)).astype('float32')
    if target_mode == 'log_return':
        return (prev_prices * np.exp(predicted_targets)).astype('float32')
    raise ValueError(f'Unknown TARGET_MODE: {target_mode}')


def make_sequences(
    features: np.ndarray,
    targets:  np.ndarray,
    lookback: int,
) -> tuple:
    x_list, y_list = [], []
    for i in range(lookback, len(features)):
        x_list.append(features[i - lookback : i])
        y_list.append(targets[i])
    return np.array(x_list, dtype='float32'), np.array(y_list, dtype='float32')


def _forecast_value(forecast) -> float:
    if hasattr(forecast, 'iloc'):
        return float(forecast.iloc[0])
    return float(np.asarray(forecast).ravel()[0])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f'Dataset not found: {DATASET_PATH}')

    df = pd.read_csv(DATASET_PATH)
    if 'Date' not in df.columns:
        raise ValueError("Dataset must contain a 'Date' column.")
    if 'Close' not in df.columns:
        raise ValueError("Dataset must contain a 'Close' column.")

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df = df.dropna(subset=['Close']).reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LSTM
# ══════════════════════════════════════════════════════════════════════════════

def build_and_train_lstm(
    x_train:       np.ndarray,
    y_train:       np.ndarray,
    x_val:         np.ndarray,
    y_val:         np.ndarray,
    lookback:      int,
    feature_count: int,
    epochs:        int,
    batch_size:    int,
) -> tuple:
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(lookback, feature_count)),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.Huber(),
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True, verbose=0,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=0,
    )

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )
    return model, history


def lstm_predict_and_reconstruct(
    model:        tf.keras.Model,
    x_data:       np.ndarray,
    target_scaler,
    target_mode:  str,
    prev_prices:  np.ndarray,
) -> np.ndarray:
    raw = model.predict(x_data, verbose=0).reshape(-1, 1)
    unscaled = target_scaler.inverse_transform(raw).reshape(-1)
    return reconstruct_prices(unscaled.astype('float32'), target_mode, prev_prices)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ARIMA
# ══════════════════════════════════════════════════════════════════════════════

def candidate_orders(target_mode: str) -> list:
    if target_mode == 'price':
        return CANDIDATE_ORDERS_PRICE
    return CANDIDATE_ORDERS_RETURN


def fit_arima(series: np.ndarray, order: tuple):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return ARIMA(series, order=order).fit()
    except Exception:
        return None


def walk_forward_targets(
    series: np.ndarray,
    order:  tuple,
    start:  int,
    end:    int,
) -> np.ndarray:
    """One-step-ahead forecasts for indices [start, end) — no future leakage."""
    if start >= end:
        return np.array([], dtype='float32')

    history = series[:start].astype('float64')
    fitted = fit_arima(history, order)
    if fitted is None:
        raise RuntimeError(f'ARIMA{order} failed to fit on history ending at {start}.')

    preds = []
    for t in range(start, end):
        preds.append(_forecast_value(fitted.forecast(steps=1)))
        fitted = fitted.extend([float(series[t])])
    return np.array(preds, dtype='float32')


def select_arima_order(
    series:       np.ndarray,
    train_end:    int,
    val_start:    int,
    val_end:      int,
    close_values: np.ndarray,
    target_mode:  str,
) -> tuple:
    """Grid-search ARIMA(p,d,q); rank by validation RMSE on reconstructed prices."""
    train_series = series[:train_end]
    prev_val   = close_values[val_start - 1 : val_end - 1]
    actual_val = close_values[val_start : val_end]

    best_order, best_rmse = None, float('inf')
    tried = failed = 0

    for order in candidate_orders(target_mode):
        if fit_arima(train_series, order) is None:
            failed += 1
            continue
        tried += 1
        try:
            val_targets = walk_forward_targets(series, order, val_start, val_end)
            val_prices  = reconstruct_prices(val_targets, target_mode, prev_val)
            rmse = float(np.sqrt(np.mean((val_prices - actual_val) ** 2)))
        except Exception:
            failed += 1
            continue
        if rmse < best_rmse:
            best_rmse, best_order = rmse, order

    if best_order is None:
        raise RuntimeError('No ARIMA order converged during grid search.')

    print(f'  Orders tried: {tried} | failed: {failed}')
    print(f'  Selected ARIMA{best_order}  (val RMSE={best_rmse:.4f})')
    return best_order


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MOVING AVERAGE
# ══════════════════════════════════════════════════════════════════════════════

def moving_average_baseline(
    close_values: np.ndarray,
    split_idx:    int,
    lookback:     int,
) -> np.ndarray:
    """Rolling mean of the last `lookback` actual Close prices."""
    return np.array([
        close_values[i - lookback : i].mean()
        for i in range(split_idx, len(close_values))
    ], dtype='float32')


def persistence_baseline(close_values: np.ndarray, split_idx: int) -> np.ndarray:
    """Naive random-walk forecast: tomorrow's price = today's actual close.

    This is the reference every serious model must beat. If LSTM/ARIMA lie on
    top of this line, they add no information beyond "yesterday's price".
    """
    return close_values[split_idx - 1 : len(close_values) - 1].copy()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_model_comparison(
    dates:       pd.Series,
    actual:      np.ndarray,
    preds:       dict,
) -> None:
    """All model forecasts + actual on a single price chart."""
    x = dates if dates is not None else np.arange(len(actual))
    styles = {
        'LSTM':        ('steelblue', '--', 1.6),
        'Persistence': ('purple',    '-',  1.0),
    }
    plt.figure(figsize=(16, 7))
    plt.plot(x, actual, label='Actual', color='black', linewidth=2.2)
    palette = ['crimson', 'green', 'darkorange', 'brown', 'teal']
    ci = 0
    for name, series in preds.items():
        color, ls, lw = styles.get(name, (palette[ci % len(palette)], '-.', 1.6))
        if name not in styles:
            ci += 1
        plt.plot(x, series, label=name, color=color, linestyle=ls, linewidth=lw)

    plt.title(
        'S&P 500 — Next-Day Close Forecast Comparison (Test Period)\n'
        f'LSTM vs ARIMA vs Moving Average vs Persistence   |   '
        f'TARGET_MODE={TARGET_MODE}  WINDOW={WINDOW_SIZE}'
    )
    plt.xlabel('Date' if dates is not None else 'Test Day Index')
    plt.ylabel('Price (USD)')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('model_comparison_predictions.png', dpi=150)
    plt.show()


def plot_returns_comparison(
    dates:        pd.Series,
    actual:       np.ndarray,
    prev_prices:  np.ndarray,
    preds:        dict,
) -> None:
    """Same forecasts, but in RETURN space — this reveals the real skill.

    In price space every model hugs the actual line (because of the P_{t-1}
    anchor). In return space you can see whether a model actually predicts the
    daily change, or just outputs noise around zero.
    """
    x = dates if dates is not None else np.arange(len(actual))
    actual_ret = actual / prev_prices - 1.0

    # Only the learned models are meaningful in return space (MA/Persistence
    # would just clutter the picture). Persistence's predicted return is 0 by
    # construction; MA's is a large lagged artifact.
    learned = [n for n in preds if n.startswith(('LSTM', 'ARIMA'))]
    palette = {'LSTM': 'steelblue'}

    fig, axes = plt.subplots(2, 1, figsize=(16, 9))  # NO shared x — panels differ
    fig.suptitle(
        'Daily RETURN space — do the models predict the change, or just noise?\n'
        f'TARGET_MODE={TARGET_MODE}',
        fontsize=13,
    )

    # Top: actual vs learned-model predicted returns over time
    axes[0].axhline(0.0, color='gray', linewidth=0.8)
    axes[0].plot(x, actual_ret, label='Actual return', color='black', linewidth=1.2, alpha=0.9)
    for name in learned:
        pred_ret = preds[name] / prev_prices - 1.0
        axes[0].plot(x, pred_ret, label=f'{name} pred.',
                     color=palette.get(name, 'crimson'), linewidth=1.0, alpha=0.85)
    ylim = float(np.max(np.abs(actual_ret))) * 1.1
    axes[0].set_ylim(-ylim, ylim)
    axes[0].set_ylabel('Daily return')
    axes[0].set_xlabel('Date' if dates is not None else 'Test Day Index')
    axes[0].set_title(
        'Predicted vs actual daily returns — model predictions stay near 0 '
        'while the market swings widely'
    )
    axes[0].legend(loc='upper right', fontsize=8, ncol=len(learned) + 1)

    # Bottom: scatter of predicted vs actual return for the learned models
    for name in learned:
        pred_ret = preds[name] / prev_prices - 1.0
        axes[1].scatter(actual_ret, pred_ret, s=8, alpha=0.35, label=name,
                        color=palette.get(name, 'crimson'))
    lim = float(np.max(np.abs(actual_ret))) * 1.05
    axes[1].plot([-lim, lim], [-lim, lim], 'r--', linewidth=1.0, label='Perfect prediction')
    axes[1].axhline(0.0, color='gray', linewidth=0.6)
    axes[1].axvline(0.0, color='gray', linewidth=0.6)
    axes[1].set_xlim(-lim, lim)
    axes[1].set_ylim(-lim, lim)
    axes[1].set_xlabel('Actual daily return')
    axes[1].set_ylabel('Predicted daily return')
    axes[1].set_title(
        'If models had skill, points would follow the red diagonal — '
        'instead they form a flat cloud near 0'
    )
    axes[1].legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig('model_comparison_returns.png', dpi=150)
    plt.show()


def plot_crash_zoom(
    dates:        pd.Series,
    actual:       np.ndarray,
    prev_prices:  np.ndarray,
    preds:        dict,
    window:       int = 40,
) -> None:
    """Zoom on the sharpest single-day drop to show the one-day LAG effect.

    Models cannot foresee a crash: on the crash day they miss it (anchored on
    the pre-crash price) and only 'catch up' the next day. This is reaction with
    a lag, not genuine prediction.
    """
    actual_ret = actual / prev_prices - 1.0
    crash_i = int(np.argmin(actual_ret))
    lo = max(0, crash_i - window // 2)
    hi = min(len(actual), crash_i + window // 2)
    x = (dates.iloc[lo:hi] if dates is not None else np.arange(lo, hi))

    plt.figure(figsize=(14, 6))
    plt.plot(x, actual[lo:hi], label='Actual', color='black', linewidth=2.4, marker='o', markersize=3)
    styles = {'LSTM': ('steelblue', '--'), 'Persistence': ('purple', '-')}
    palette = ['crimson', 'green', 'darkorange']
    ci = 0
    for name, series in preds.items():
        color, ls = styles.get(name, (palette[ci % len(palette)], '-.'))
        if name not in styles:
            ci += 1
        plt.plot(x, series[lo:hi], label=name, color=color, linestyle=ls, linewidth=1.5)

    drop_pct = actual_ret[crash_i] * 100.0
    crash_date = dates.iloc[crash_i].date() if dates is not None else f'idx {crash_i}'
    plt.axvline(x.iloc[crash_i - lo] if dates is not None else crash_i,
                color='red', linewidth=1.0, linestyle=':')
    plt.title(
        f'Zoom on sharpest drop ({crash_date}, {drop_pct:.2f}% in one day)\n'
        'Models miss the crash on the day and only react afterwards (one-day lag)'
    )
    plt.xlabel('Date' if dates is not None else 'Test Day Index')
    plt.ylabel('Price (USD)')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('model_comparison_crash_zoom.png', dpi=150)
    plt.show()


def plot_metrics_comparison(metrics_all: dict) -> None:
    """Grouped bar chart comparing RMSE, MAE, MAPE across the three models."""
    models = list(metrics_all.keys())
    x = np.arange(len(models))
    colors = ['steelblue', 'crimson', 'green']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'LSTM vs ARIMA vs Moving Average | TARGET={TARGET_MODE}', fontsize=13)

    for idx, metric in enumerate(['RMSE', 'MAE', 'MAPE']):
        values = [metrics_all[m][metric] for m in models]
        axes[idx].bar(x, values, color=colors[:len(models)], alpha=0.85, edgecolor='white')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(models, rotation=15, ha='right')
        axes[idx].set_title(metric)
        axes[idx].set_ylabel(metric)

    plt.tight_layout()
    plt.savefig('model_comparison_metrics.png', dpi=150)
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print(f'\n{"═"*70}')
    print(f'  compare_models  |  LSTM vs ARIMA vs Moving Average')
    print(f'  TARGET_MODE={TARGET_MODE}  WINDOW={WINDOW_SIZE}')
    print(f'{"═"*70}\n')

    # ── Load data & shared splits ────────────────────────────────────────────
    df = load_data()
    N = len(df)
    close_values  = df['Close'].values.astype('float32')
    target_series = build_target_series(close_values, TARGET_MODE)
    split_idx     = int(N * SPLIT_RATIO)

    val_days  = max(1, int(split_idx * VAL_RATIO))
    train_end = split_idx - val_days
    val_start = train_end
    val_end   = split_idx

    print(f'Dataset : {N} trading days ({DATASET_PATH.name})')
    print(f'Train   : days 0–{train_end - 1} ({train_end} days)')
    print(f'Val     : days {val_start}–{val_end - 1} ({val_days} days)')
    print(f'Test    : days {split_idx}–{N - 1} ({N - split_idx} days)')

    assert split_idx > WINDOW_SIZE, 'split_idx must be larger than WINDOW_SIZE.'
    assert split_idx < N, 'Test set is empty.'

    # Common test-period arrays
    dates_test       = df['Date'].iloc[split_idx:].reset_index(drop=True)
    prev_prices_test = close_values[split_idx - 1 : N - 1]
    actual_close     = close_values[split_idx:]

    # ── LSTM ─────────────────────────────────────────────────────────────────
    print('\n' + '─'*70)
    print(' [1/3] LSTM')
    print('─'*70)

    feature_data = close_values.reshape(-1, 1)
    feature_scaler = MinMaxScaler()
    feature_scaler.fit(feature_data[:split_idx])
    features_scaled = feature_scaler.transform(feature_data).astype('float32')

    if TARGET_MODE == 'price':
        target_scaler = MinMaxScaler()
    else:
        target_scaler = StandardScaler()
    target_scaler.fit(target_series[:split_idx].reshape(-1, 1))
    targets_scaled = (
        target_scaler.transform(target_series.reshape(-1, 1)).reshape(-1).astype('float32')
    )

    x_tv, y_tv = make_sequences(
        features_scaled[:split_idx], targets_scaled[:split_idx], WINDOW_SIZE,
    )
    val_size   = max(1, int(len(x_tv) * VAL_RATIO))
    train_size = len(x_tv) - val_size
    x_train, y_train = x_tv[:train_size], y_tv[:train_size]
    x_val,   y_val   = x_tv[train_size:], y_tv[train_size:]

    test_ctx_f = features_scaled[split_idx - WINDOW_SIZE:]
    test_ctx_t = targets_scaled[split_idx - WINDOW_SIZE:]
    x_test, _ = make_sequences(test_ctx_f, test_ctx_t, WINDOW_SIZE)

    assert len(x_test) == len(actual_close) == len(prev_prices_test), \
        'Test array length mismatch — check indexing.'

    print('Training LSTM...')
    lstm_model, _ = build_and_train_lstm(
        x_train, y_train, x_val, y_val,
        lookback=WINDOW_SIZE, feature_count=1,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
    )
    lstm_preds = lstm_predict_and_reconstruct(
        lstm_model, x_test, target_scaler, TARGET_MODE, prev_prices_test,
    )

    # ── ARIMA ────────────────────────────────────────────────────────────────
    print('\n' + '─'*70)
    print(' [2/3] ARIMA')
    print('─'*70)
    print('Selecting ARIMA order on train/val...')
    best_order = select_arima_order(
        target_series, train_end, val_start, val_end, close_values, TARGET_MODE,
    )
    print('Walk-forward forecasting on test set...')
    arima_targets = walk_forward_targets(target_series, best_order, split_idx, N)
    arima_preds = reconstruct_prices(arima_targets, TARGET_MODE, prev_prices_test)

    # ── Moving Average ───────────────────────────────────────────────────────
    print('\n' + '─'*70)
    print(' [3/3] Moving Average')
    print('─'*70)
    ma_preds = moving_average_baseline(close_values, split_idx, WINDOW_SIZE)
    print(f'Rolling mean over last {WINDOW_SIZE} actual closes computed.')

    # ── Reference: naive persistence (random walk) ───────────────────────────
    persistence_preds = persistence_baseline(close_values, split_idx)

    # ── Metrics ──────────────────────────────────────────────────────────────
    lstm_label  = 'LSTM'
    arima_label = f'ARIMA{best_order}'
    ma_label    = f'MA-{WINDOW_SIZE}'
    pers_label  = 'Persistence'

    # Order matters for plots/tables: learned models, then baselines.
    preds_all = {
        lstm_label:  lstm_preds,
        arima_label: arima_preds,
        ma_label:    ma_preds,
        pers_label:  persistence_preds,
    }
    metrics = {name: compute_metrics(actual_close, p) for name, p in preds_all.items()}
    dir_acc = {
        name: directional_accuracy(actual_close, p, prev_prices_test)
        for name, p in preds_all.items()
    }

    print(f'\n{"═"*70}')
    print(' OUT-OF-SAMPLE COMPARISON (test set)')
    print(f'{"═"*70}')
    print(f'  {"Model":<16} {"RMSE":>9} {"MAE":>9} {"MAPE%":>9} {"DirAcc%":>10}')
    print(f'  {"─"*56}')
    for name in preds_all:
        m = metrics[name]
        print(
            f'  {name:<16} {m["RMSE"]:>9.4f} {m["MAE"]:>9.4f} '
            f'{m["MAPE"]:>9.4f} {dir_acc[name]:>10.1f}'
        )

    best_by_rmse = min(metrics, key=lambda k: metrics[k]['RMSE'])
    print(f'\n  → Best by RMSE: {best_by_rmse} ({metrics[best_by_rmse]["RMSE"]:.4f})')

    # How close are the learned models to simply repeating yesterday's price?
    pers_rmse = metrics[pers_label]['RMSE']
    print(f'\n  Do LSTM / ARIMA beat the naive "tomorrow = today" random walk?')
    for name in (lstm_label, arima_label):
        gain = (pers_rmse - metrics[name]['RMSE']) / pers_rmse * 100.0
        verdict = 'beats it' if gain > 1 else ('≈ same as' if gain > -1 else 'worse than')
        print(f'    {name:<16} RMSE gain vs persistence: {gain:+6.2f}%  → {verdict} random walk')
    print('    (A gain near 0% means the model adds no info beyond yesterday\'s price —')
    print('     the "repetition" your professor noticed. This is the expected EMH result.)')

    # ── Save results & plots ─────────────────────────────────────────────────
    results_df = pd.DataFrame([
        {'Model': name, **metrics[name], 'DirAcc%': dir_acc[name]}
        for name in preds_all
    ])
    results_df.to_csv('model_comparison_results.csv', index=False)
    print('\n  Results saved → model_comparison_results.csv')

    pd.DataFrame({
        'Date':      dates_test,
        'Actual':    actual_close,
        'PrevClose': prev_prices_test,
        lstm_label:  lstm_preds,
        arima_label: arima_preds,
        ma_label:    ma_preds,
        pers_label:  persistence_preds,
    }).to_csv('model_comparison_predictions.csv', index=False)
    print('  Predictions saved → model_comparison_predictions.csv')

    plot_model_comparison(dates_test, actual_close, preds_all)
    plot_metrics_comparison({k: metrics[k] for k in (lstm_label, arima_label, ma_label)})
    plot_returns_comparison(dates_test, actual_close, prev_prices_test, preds_all)
    plot_crash_zoom(dates_test, actual_close, prev_prices_test, preds_all)
    print(
        '\n  Plots saved → model_comparison_predictions.png, '
        'model_comparison_metrics.png,'
        '\n                model_comparison_returns.png, '
        'model_comparison_crash_zoom.png'
    )


if __name__ == '__main__':
    main()
