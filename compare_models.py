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
TARGET_MODE: str   = 'return'   # 'price' | 'return' | 'log_return'
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


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_model_comparison(
    dates:       pd.Series,
    actual:      np.ndarray,
    lstm_preds:  np.ndarray,
    arima_preds: np.ndarray,
    ma_preds:    np.ndarray,
    arima_order: tuple,
) -> None:
    """All three model forecasts + actual on a single chart."""
    x = dates if dates is not None else np.arange(len(actual))

    plt.figure(figsize=(16, 7))
    plt.plot(x, actual,      label='Actual',            color='black',      linewidth=2.2)
    plt.plot(x, lstm_preds,  label='LSTM',              color='steelblue',  linewidth=1.6, linestyle='--')
    plt.plot(x, arima_preds, label=f'ARIMA{arima_order}', color='crimson',  linewidth=1.6, linestyle='-.')
    plt.plot(x, ma_preds,    label=f'Moving Average ({WINDOW_SIZE})', color='green', linewidth=1.6, linestyle=':')

    plt.title(
        'S&P 500 — Next-Day Close Forecast Comparison (Test Period)\n'
        f'LSTM vs ARIMA vs Moving Average   |   TARGET_MODE={TARGET_MODE}  WINDOW={WINDOW_SIZE}'
    )
    plt.xlabel('Date' if dates is not None else 'Test Day Index')
    plt.ylabel('Price (USD)')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('model_comparison_predictions.png', dpi=150)
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

    # ── Metrics ──────────────────────────────────────────────────────────────
    lstm_label  = 'LSTM'
    arima_label = f'ARIMA{best_order}'
    ma_label    = f'MA-{WINDOW_SIZE}'

    metrics = {
        lstm_label:  compute_metrics(actual_close, lstm_preds),
        arima_label: compute_metrics(actual_close, arima_preds),
        ma_label:    compute_metrics(actual_close, ma_preds),
    }
    dir_acc = {
        lstm_label:  directional_accuracy(actual_close, lstm_preds,  prev_prices_test),
        arima_label: directional_accuracy(actual_close, arima_preds, prev_prices_test),
        ma_label:    directional_accuracy(actual_close, ma_preds,    prev_prices_test),
    }

    print(f'\n{"═"*70}')
    print(' OUT-OF-SAMPLE COMPARISON (test set)')
    print(f'{"═"*70}')
    print(f'  {"Model":<16} {"RMSE":>9} {"MAE":>9} {"MAPE%":>9} {"DirAcc%":>10}')
    print(f'  {"─"*56}')
    for name in (lstm_label, arima_label, ma_label):
        m = metrics[name]
        print(
            f'  {name:<16} {m["RMSE"]:>9.4f} {m["MAE"]:>9.4f} '
            f'{m["MAPE"]:>9.4f} {dir_acc[name]:>10.1f}'
        )

    best_by_rmse = min(metrics, key=lambda k: metrics[k]['RMSE'])
    print(f'\n  → Best by RMSE: {best_by_rmse} ({metrics[best_by_rmse]["RMSE"]:.4f})')

    # ── Save results & plots ─────────────────────────────────────────────────
    results_df = pd.DataFrame([
        {'Model': name, **metrics[name], 'DirAcc%': dir_acc[name]}
        for name in (lstm_label, arima_label, ma_label)
    ])
    results_df.to_csv('model_comparison_results.csv', index=False)
    print('\n  Results saved → model_comparison_results.csv')

    pd.DataFrame({
        'Date':    dates_test,
        'Actual':  actual_close,
        'LSTM':    lstm_preds,
        arima_label: arima_preds,
        ma_label:  ma_preds,
    }).to_csv('model_comparison_predictions.csv', index=False)
    print('  Predictions saved → model_comparison_predictions.csv')

    plot_model_comparison(
        dates_test, actual_close, lstm_preds, arima_preds, ma_preds, best_order,
    )
    plot_metrics_comparison(metrics)
    print('\n  Plots saved → model_comparison_predictions.png, model_comparison_metrics.png')


if __name__ == '__main__':
    main()
