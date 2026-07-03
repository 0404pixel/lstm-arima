import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from statsmodels.tsa.arima.model import ARIMA

sns.set()
np.random.seed(42)

# DATASET 
DATASET_PATH = Path(__file__).resolve().parent.parent / 'dataset' / 'SP500-2000-2015.csv'

# CONFIGURATION
TARGET_MODE: str = 'return'   # 'price' | 'return' | 'log_return'
SPLIT_RATIO: float = 0.80
VAL_RATIO:   float = 0.10
WINDOW_SIZE: int   = 30       # lookback for Ridge / MA baselines

# ARIMA order grid
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



# UTILITY FUNCTIONS


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


def _forecast_value(forecast) -> float:
    if hasattr(forecast, 'iloc'):
        return float(forecast.iloc[0])
    return float(np.asarray(forecast).ravel()[0])


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



# DATA LOADING


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



#  ARIMA
def candidate_orders(target_mode: str) -> list:
    if target_mode == 'price':
        return CANDIDATE_ORDERS_PRICE
    return CANDIDATE_ORDERS_RETURN


def fit_arima(series: np.ndarray, order: tuple):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = ARIMA(series, order=order)
            return model.fit()
    except Exception:
        return None


def walk_forward_targets(
    series: np.ndarray,
    order:  tuple,
    start:  int,
    end:    int,
) -> np.ndarray:
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
    series:      np.ndarray,
    train_end:   int,
    val_start:   int,
    val_end:     int,
    close_values: np.ndarray,
    target_mode: str,
) -> tuple:
    train_series = series[:train_end]
    prev_val = close_values[val_start - 1 : val_end - 1]
    actual_val = close_values[val_start : val_end]

    best_order = None
    best_rmse = float('inf')
    tried = 0
    failed = 0

    for order in candidate_orders(target_mode):
        if fit_arima(train_series, order) is None:
            failed += 1
            continue
        tried += 1
        try:
            val_targets = walk_forward_targets(series, order, val_start, val_end)
            val_prices = reconstruct_prices(val_targets, target_mode, prev_val)
            rmse = float(np.sqrt(np.mean((val_prices - actual_val) ** 2)))
        except Exception:
            failed += 1
            continue

        if rmse < best_rmse:
            best_rmse = rmse
            best_order = order

    if best_order is None:
        raise RuntimeError('No ARIMA order converged during grid search.')

    print(f'  Orders tried: {tried} | failed: {failed}')
    print(f'  Selected ARIMA{best_order}  (val RMSE={best_rmse:.4f})')
    return best_order



#  BASELINES 

def persistence_baseline(close_values: np.ndarray, split_idx: int) -> np.ndarray:
    return close_values[split_idx - 1 : len(close_values) - 1].copy()


def ridge_regression_baseline(
    close_values:   np.ndarray,
    split_idx:      int,
    lookback:       int,
) -> np.ndarray:
    feature_scaler = MinMaxScaler()
    feature_scaler.fit(close_values[:split_idx].reshape(-1, 1))
    close_scaled = feature_scaler.transform(
        close_values.reshape(-1, 1)
    ).reshape(-1)

    x_ridge, y_ridge = [], []
    for i in range(lookback, len(close_scaled)):
        x_ridge.append(close_scaled[i - lookback : i])
        y_ridge.append(close_scaled[i])
    x_ridge = np.array(x_ridge, dtype='float32')
    y_ridge = np.array(y_ridge, dtype='float32')

    train_end = split_idx - lookback
    test_start = split_idx - lookback

    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(x_ridge[:train_end], y_ridge[:train_end])
    preds_scaled = ridge_model.predict(x_ridge[test_start:]).astype('float32')
    return feature_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(-1)


def moving_average_baseline(
    close_values: np.ndarray,
    split_idx:    int,
    lookback:     int,
) -> np.ndarray:
    return np.array([
        close_values[i - lookback : i].mean()
        for i in range(split_idx, len(close_values))
    ], dtype='float32')


# PLOTTING

def plot_predictions(
    actual:            np.ndarray,
    arima_preds:       np.ndarray,
    persistence_preds: np.ndarray,
    ridge_preds:       np.ndarray,
    ma_preds:          np.ndarray,
) -> None:
    idx = np.arange(len(actual))
    plt.figure(figsize=(16, 6))
    plt.plot(idx, actual,            label='Actual',      color='black',      linewidth=2.0)
    plt.plot(idx, arima_preds,       label='ARIMA',       color='steelblue',  linewidth=1.6, linestyle='--')
    plt.plot(idx, persistence_preds, label='Persistence', color='crimson',    linewidth=1.2, linestyle=':')
    plt.plot(idx, ridge_preds,       label='Ridge',       color='darkorange', linewidth=1.2, linestyle='-.')
    plt.plot(idx, ma_preds,          label=f'MA-{WINDOW_SIZE}', color='green', linewidth=1.2, linestyle=':')
    plt.title(
        f'S&P 500 — Next-Day Close Forecast (Test Period)\n'
        f'ARIMA  TARGET_MODE={TARGET_MODE}'
    )
    plt.xlabel('Test Day Index')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('arima_v3_predictions.png', dpi=150)
    plt.show()


def plot_diagnostics(actual: np.ndarray, predicted: np.ndarray, model_name: str) -> None:
    residuals = actual - predicted
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{model_name} Diagnostics | TARGET={TARGET_MODE}', fontsize=13)

    axes[0, 0].plot(residuals, color='steelblue', linewidth=0.8)
    axes[0, 0].axhline(0.0, color='red', linewidth=1.2, linestyle='--')
    axes[0, 0].set_title('Residuals Over Time  (Actual − Predicted)')
    axes[0, 0].set_xlabel('Test Day Index')
    axes[0, 0].set_ylabel('Residual (USD)')

    axes[0, 1].hist(residuals, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
    axes[0, 1].axvline(0.0, color='red', linewidth=1.2, linestyle='--')
    axes[0, 1].set_title('Prediction Error Distribution')
    axes[0, 1].set_xlabel('Error (USD)')
    axes[0, 1].set_ylabel('Frequency')

    axes[1, 0].scatter(actual, predicted, alpha=0.35, s=10, color='steelblue')
    lo = min(actual.min(), predicted.min())
    hi = max(actual.max(), predicted.max())
    axes[1, 0].plot([lo, hi], [lo, hi], 'r--', linewidth=1.2, label='Perfect prediction')
    axes[1, 0].set_title('Actual vs Predicted')
    axes[1, 0].set_xlabel('Actual Price (USD)')
    axes[1, 0].set_ylabel('Predicted Price (USD)')
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(np.cumsum(np.abs(residuals)), color='darkorange', linewidth=1.2)
    axes[1, 1].set_title('Cumulative Absolute Error')
    axes[1, 1].set_xlabel('Test Day Index')
    axes[1, 1].set_ylabel('Cumulative |Error| (USD)')

    plt.tight_layout()
    fname = f'arima_v3_diagnostics_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(fname, dpi=150)
    plt.show()


def plot_metrics_comparison(metrics_all: dict) -> None:
    models = list(metrics_all.keys())
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Model Comparison | TARGET={TARGET_MODE}', fontsize=13)

    for idx, (metric, color) in enumerate(zip(
        ['RMSE', 'MAE', 'MAPE'],
        ['#1f77b4', '#ff7f0e', '#2ca02c'],
    )):
        values = [metrics_all[m][metric] for m in models]
        axes[idx].bar(x, values, color=color, alpha=0.85, edgecolor='white')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(models, rotation=15, ha='right')
        axes[idx].set_title(metric)
        axes[idx].set_ylabel(metric)

    plt.tight_layout()
    plt.savefig('arima_v3_metrics_comparison.png', dpi=150)
    plt.show()


#  MAIN

def main() -> None:
    print(f'\n{"═"*70}')
    print(f'  arimatrad_v3  |  TARGET_MODE={TARGET_MODE}')
    print(f'{"═"*70}\n')

    df = load_data()
    N = len(df)
    close_values = df['Close'].values.astype('float32')
    target_series = build_target_series(close_values, TARGET_MODE)
    split_idx = int(N * SPLIT_RATIO)

    val_days = max(1, int(split_idx * VAL_RATIO))
    train_end = split_idx - val_days
    val_start = train_end
    val_end = split_idx

    print(f'Dataset : {N} trading days')
    print(f'Train   : days 0–{train_end - 1} ({train_end} days)')
    print(f'Val     : days {val_start}–{val_end - 1} ({val_days} days)')
    print(f'Test    : days {split_idx}–{N - 1} ({N - split_idx} days)')

    assert train_end > 30, 'Training slice too small for ARIMA.'
    assert split_idx < N, 'Test set is empty.'

    # Order selection on train → val (no test leakage) 
    print('\nSelecting ARIMA order on train/val...')
    best_order = select_arima_order(
        target_series, train_end, val_start, val_end,
        close_values, TARGET_MODE,
    )

    # Test predictions: walk-forward from split_idx 
    print('\nWalk-forward forecasting on test set...')
    arima_test_targets = walk_forward_targets(
        target_series, best_order, split_idx, N,
    )
    prev_prices_test = close_values[split_idx - 1 : N - 1]
    actual_close = close_values[split_idx :]
    arima_test_preds = reconstruct_prices(
        arima_test_targets, TARGET_MODE, prev_prices_test,
    )

    # In-sample (train) evaluation 
    train_eval_start = max(1, WINDOW_SIZE)
    arima_train_targets = walk_forward_targets(
        target_series, best_order, train_eval_start, train_end,
    )
    prev_prices_train = close_values[train_eval_start - 1 : train_end - 1]
    actual_close_train = close_values[train_eval_start : train_end]
    arima_train_preds = reconstruct_prices(
        arima_train_targets, TARGET_MODE, prev_prices_train,
    )

    assert len(arima_test_preds) == len(actual_close) == len(prev_prices_test)

    print('\nFirst 10 test predictions (actual | prev day | predicted):')
    for i in range(min(10, len(actual_close))):
        print(
            f'  {df["Date"].iloc[split_idx + i].date()}  '
            f'actual={actual_close[i]:.2f}  prev={prev_prices_test[i]:.2f}  '
            f'pred={arima_test_preds[i]:.2f}  '
            f'|pred-prev|={abs(arima_test_preds[i] - prev_prices_test[i]):.2f}'
        )

    pd.DataFrame({
        'Date': df['Date'].iloc[split_idx:].reset_index(drop=True),
        'Actual': actual_close,
        'PrevClose': prev_prices_test,
        'Predicted': arima_test_preds,
        'PredMinusPrev': arima_test_preds - prev_prices_test,
        'ActualMinusPrev': actual_close - prev_prices_test,
    }).to_csv('arima_v3_predictions.csv', index=False)
    print('  All test predictions saved → arima_v3_predictions.csv')

    # Baselines 
    persistence_preds = persistence_baseline(close_values, split_idx)
    ridge_preds = ridge_regression_baseline(close_values, split_idx, WINDOW_SIZE)
    ma_preds = moving_average_baseline(close_values, split_idx, WINDOW_SIZE)

    # Metrics
    metrics_arima_test  = compute_metrics(actual_close, arima_test_preds)
    metrics_arima_train = compute_metrics(actual_close_train, arima_train_preds)
    metrics_persistence = compute_metrics(actual_close, persistence_preds)
    metrics_ridge       = compute_metrics(actual_close, ridge_preds)
    metrics_ma          = compute_metrics(actual_close, ma_preds)

    da_arima        = directional_accuracy(actual_close, arima_test_preds, prev_prices_test)
    da_persistence  = directional_accuracy(actual_close, persistence_preds, prev_prices_test)
    da_ridge        = directional_accuracy(actual_close, ridge_preds, prev_prices_test)
    da_ma           = directional_accuracy(actual_close, ma_preds, prev_prices_test)

    print(f'\n{"─"*70}')
    print(' IN-SAMPLE performance (train set)')
    print(f'{"─"*70}')
    print(
        f'  ARIMA{best_order}  RMSE={metrics_arima_train["RMSE"]:.4f} | '
        f'MAE={metrics_arima_train["MAE"]:.4f} | '
        f'MAPE={metrics_arima_train["MAPE"]:.4f}%'
    )

    print(f'\n{"─"*70}')
    print(' OUT-OF-SAMPLE performance (test set)')
    print(f'{"─"*70}')
    print(f'  {"Model":<16} {"RMSE":>9} {"MAE":>9} {"MAPE%":>9} {"DirAcc%":>10}')
    print(f'  {"─"*56}')

    rows = {
        f'ARIMA{best_order}': (metrics_arima_test, da_arima),
        'Persistence': (metrics_persistence, da_persistence),
        'Ridge': (metrics_ridge, da_ridge),
        f'MA-{WINDOW_SIZE}': (metrics_ma, da_ma),
    }
    for name, (m, da) in rows.items():
        print(
            f'  {name:<16} {m["RMSE"]:>9.4f} {m["MAE"]:>9.4f} '
            f'{m["MAPE"]:>9.4f} {da:>10.1f}'
        )

    # Interpretation
    print(f'\n{"─"*70}')
    print(' INTERPRETATION')
    print(f'{"─"*70}')

    rmse_vs_persistence = (
        (metrics_persistence['RMSE'] - metrics_arima_test['RMSE'])
        / metrics_persistence['RMSE'] * 100.0
    )
    rmse_vs_ridge = (
        (metrics_ridge['RMSE'] - metrics_arima_test['RMSE'])
        / metrics_ridge['RMSE'] * 100.0
    )
    overfitting_ratio = (
        metrics_arima_test['RMSE'] / metrics_arima_train['RMSE']
        if metrics_arima_train['RMSE'] > 1e-8 else float('inf')
    )

    if rmse_vs_persistence > 5:
        print(f'  ✓ ARIMA beats persistence by {rmse_vs_persistence:.1f}% RMSE.')
    elif rmse_vs_persistence > 0:
        print(f'  △ ARIMA marginally outperforms persistence (+{rmse_vs_persistence:.1f}% RMSE).')
    else:
        print(
            f'  ✗ ARIMA does NOT outperform persistence ({rmse_vs_persistence:.1f}% RMSE).\n'
            f'    Predictions are driven by autocorrelation, not genuine signal.'
        )

    if da_arima > 55:
        print(f'  ✓ ARIMA directional accuracy {da_arima:.1f}% — above 55% threshold.')
    elif da_arima > 50:
        print(f'  △ ARIMA directional accuracy {da_arima:.1f}% — marginal signal.')
    else:
        print(f'  ✗ ARIMA directional accuracy {da_arima:.1f}% — at or below random.')

    if rmse_vs_ridge > 5:
        print(f'  ✓ ARIMA beats Ridge by {rmse_vs_ridge:.1f}% RMSE.')
    elif rmse_vs_ridge >= 0:
        print(f'  △ ARIMA similar to Ridge ({rmse_vs_ridge:.1f}% RMSE gap).')
    else:
        print(f'  ✗ Ridge outperforms ARIMA ({-rmse_vs_ridge:.1f}% better RMSE).')

    if overfitting_ratio > 3.0:
        print(f'  ⚠ Possible overfitting: test/train RMSE ratio = {overfitting_ratio:.1f}×.')
    else:
        print(f'  ✓ Train/Test RMSE ratio: {overfitting_ratio:.2f}.')

    results_df = pd.DataFrame([
        {'Model': name, **m, 'DirAcc%': da}
        for name, (m, da) in rows.items()
    ])
    results_df.to_csv('arima_v3_results.csv', index=False)
    print(f'\n  Results saved → arima_v3_results.csv')

    plot_predictions(actual_close, arima_test_preds, persistence_preds, ridge_preds, ma_preds)
    plot_diagnostics(actual_close, arima_test_preds, 'ARIMA')
    plot_metrics_comparison({
        f'ARIMA{best_order}': metrics_arima_test,
        'Persistence': metrics_persistence,
        'Ridge': metrics_ridge,
        f'MA-{WINDOW_SIZE}': metrics_ma,
    })


if __name__ == '__main__':
    main()
