import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import Ridge

sns.set()
tf.random.set_seed(42)
np.random.seed(42)

# DATASET 
DATASET_PATH = Path(__file__).resolve().parent.parent / 'dataset' / 'SP500-2000-2015.csv'

# CONFIGURATION 
FEATURE_MODE: str = 'close_only'

# TARGET_MODE
TARGET_MODE: str = 'return'

WINDOW_SIZE:  int   = 30    # lookback days per input sequence
SPLIT_RATIO:  float = 0.80  # fraction of data for train + validation
VAL_RATIO:    float = 0.10  # validation fraction (within train+val block)
EPOCHS:       int   = 120
BATCH_SIZE:   int   = 32



#  UTILITY FUNCTIONS

def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute RMSE, MAE, and MAPE for a set of predictions."""
    rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))
    mae  = float(np.mean(np.abs(predicted - actual)))
    mape = float(np.mean(np.abs((actual - predicted) / np.maximum(np.abs(actual), 1e-8))) * 100.0)
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}


def directional_accuracy(
    actual:       np.ndarray,
    predicted:    np.ndarray,
    prev_actual:  np.ndarray,
) -> float:
    
    actual_dir = np.sign(actual    - prev_actual)
    pred_dir   = np.sign(predicted - prev_actual)
    non_flat   = actual_dir != 0
    if non_flat.sum() == 0:
        return 0.0
    return float(np.mean(actual_dir[non_flat] == pred_dir[non_flat]) * 100.0)


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

#  DATA LOADING & PREPROCESSING

def load_data() -> tuple:

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f'Dataset not found: {DATASET_PATH}')

    df = pd.read_csv(DATASET_PATH)

    if 'Date' not in df.columns:
        raise ValueError("Dataset must contain a 'Date' column.")
    if 'Close' not in df.columns:
        raise ValueError("Dataset must contain a 'Close' column.")

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)  # ensures time order

    if FEATURE_MODE == 'all':
        candidates = ['Open', 'High', 'Low', 'Close', 'Volume']
    else:
        candidates = ['Close']

    feature_cols = [c for c in candidates if c in df.columns]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    return df, feature_cols


# LSTM MODEL

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
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=0,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-5,
        verbose=0,
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,    
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )
    return model, history


#  BASELINES

def persistence_baseline(close_values: np.ndarray, split_idx: int) -> np.ndarray:
    # For t in [split_idx, …, N-1]:  pred[t] = close[t-1]
    # = close_values[split_idx-1 : N-1]
    return close_values[split_idx - 1 : len(close_values) - 1].copy()


def ridge_regression_baseline(
    features_scaled:  np.ndarray,
    close_values:     np.ndarray,
    split_idx:        int,
    lookback:         int,
    close_idx:        int,
    feature_count:    int,
    feature_scaler:   MinMaxScaler,
) -> np.ndarray:
    close_scaled = features_scaled[:, close_idx]

    x_ridge, y_ridge = [], []
    for i in range(lookback, len(close_scaled)):
        x_ridge.append(close_scaled[i - lookback : i])
        y_ridge.append(close_scaled[i])
    x_ridge = np.array(x_ridge, dtype='float32')
    y_ridge = np.array(y_ridge, dtype='float32')

    train_end  = split_idx - lookback   # last training sequence index (exclusive)
    test_start = split_idx - lookback   # first test sequence index

    # LEAKAGE CHECK: fit only on sequences that predict training-period days
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(x_ridge[:train_end], y_ridge[:train_end])

    preds_scaled = ridge_model.predict(x_ridge[test_start:]).astype('float32')

    # Inverse-transform via the feature scaler (Close column)
    dummy = np.zeros((len(preds_scaled), feature_count), dtype='float32')
    dummy[:, close_idx] = preds_scaled
    return feature_scaler.inverse_transform(dummy)[:, close_idx]


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


#  PREDICTION & PRICE RECONSTRUCTION

def predict_and_reconstruct(
    model:         tf.keras.Model,
    x_data:        np.ndarray,
    target_scaler,
    target_mode:   str,
    prev_prices:   np.ndarray,
) -> np.ndarray:
    raw = model.predict(x_data, verbose=0).reshape(-1, 1)
    unscaled = target_scaler.inverse_transform(raw).reshape(-1)

    if target_mode == 'price':
        # Raw Close price was predicted directly
        return unscaled.astype('float32')
    elif target_mode == 'return':
        # price[t] = price[t-1] * (1 + return[t])
        return (prev_prices * (1.0 + unscaled)).astype('float32')
    elif target_mode == 'log_return':
        # price[t] = price[t-1] * exp(log_return[t])
        return (prev_prices * np.exp(unscaled)).astype('float32')
    else:
        raise ValueError(f'Unknown TARGET_MODE: {target_mode}')


# PLOTTING

def plot_predictions(
    actual:            np.ndarray,
    lstm_preds:        np.ndarray,
    persistence_preds: np.ndarray,
    ridge_preds:       np.ndarray,
    ma_preds:          np.ndarray,
) -> None:
    idx = np.arange(len(actual))

    plt.figure(figsize=(16, 6))
    plt.plot(idx, actual,            label='Actual',      color='black',      linewidth=2.0)
    plt.plot(idx, lstm_preds,        label='LSTM',        color='steelblue',  linewidth=1.6, linestyle='--')
    plt.plot(idx, persistence_preds, label='Persistence', color='crimson',    linewidth=1.2, linestyle=':')
    plt.plot(idx, ridge_preds,       label='Ridge',       color='darkorange', linewidth=1.2, linestyle='-.')
    plt.plot(idx, ma_preds,          label=f'MA-{WINDOW_SIZE}', color='green', linewidth=1.2, linestyle=':')

    plt.title(
        f'S&P 500 — Next-Day Close Forecast (Test Period)\n'
        f'FEATURE_MODE={FEATURE_MODE}  TARGET_MODE={TARGET_MODE}  WINDOW={WINDOW_SIZE}'
    )
    plt.xlabel('Test Day Index')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('v3_predictions.png', dpi=150)
    plt.show()


def plot_diagnostics(
    actual:     np.ndarray,
    predicted:  np.ndarray,
    model_name: str,
) -> None:
    residuals = actual - predicted

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{model_name} Diagnostics | FEATURE={FEATURE_MODE} TARGET={TARGET_MODE}', fontsize=13)

    # 1. Residuals over time
    ax = axes[0, 0]
    ax.plot(residuals, color='steelblue', linewidth=0.8)
    ax.axhline(0.0, color='red', linewidth=1.2, linestyle='--')
    ax.set_title('Residuals Over Time  (Actual − Predicted)')
    ax.set_xlabel('Test Day Index')
    ax.set_ylabel('Residual (USD)')

    # 2. Error distribution
    ax = axes[0, 1]
    ax.hist(residuals, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(0.0, color='red', linewidth=1.2, linestyle='--')
    ax.set_title('Prediction Error Distribution')
    ax.set_xlabel('Error (USD)')
    ax.set_ylabel('Frequency')

    # 3. Actual vs Predicted scatter
    ax = axes[1, 0]
    ax.scatter(actual, predicted, alpha=0.35, s=10, color='steelblue')
    lo = min(actual.min(), predicted.min())
    hi = max(actual.max(), predicted.max())
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.2, label='Perfect prediction')
    ax.set_title('Actual vs Predicted')
    ax.set_xlabel('Actual Price (USD)')
    ax.set_ylabel('Predicted Price (USD)')
    ax.legend(fontsize=8)

    # 4. Cumulative absolute error over time
    ax = axes[1, 1]
    ax.plot(np.cumsum(np.abs(residuals)), color='darkorange', linewidth=1.2)
    ax.set_title('Cumulative Absolute Error')
    ax.set_xlabel('Test Day Index')
    ax.set_ylabel('Cumulative |Error| (USD)')

    plt.tight_layout()
    fname = f'v3_diagnostics_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(fname, dpi=150)
    plt.show()


def plot_loss_curve(history: tf.keras.callbacks.History) -> None:
    """Training vs validation Huber loss curves."""
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'],     label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('LSTM Training & Validation Loss (Huber)')
    plt.xlabel('Epoch')
    plt.ylabel('Huber Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('v3_loss_curve.png', dpi=150)
    plt.show()


def plot_metrics_comparison(metrics_all: dict) -> None:
    """Grouped bar chart comparing RMSE, MAE, MAPE across all models."""
    models   = list(metrics_all.keys())
    x        = np.arange(len(models))
    width    = 0.25
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f'Model Comparison | FEATURE={FEATURE_MODE}  TARGET={TARGET_MODE}',
        fontsize=13,
    )

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
    plt.savefig('v3_metrics_comparison.png', dpi=150)
    plt.show()



# MAIN

def main() -> None:
    print(f'\n{"═"*70}')
    print(f'  lstmtrad_v3  |  FEATURE_MODE={FEATURE_MODE}  TARGET_MODE={TARGET_MODE}  WINDOW={WINDOW_SIZE}')
    print(f'{"═"*70}\n')

    #  Load data 
    df, feature_cols = load_data()
    N             = len(df)
    close_values  = df['Close'].values.astype('float32')
    feature_data  = df[feature_cols].values.astype('float32')
    close_idx     = feature_cols.index('Close')
    feature_count = len(feature_cols)
    split_idx     = int(N * SPLIT_RATIO)

    print(f'Dataset : {N} trading days')
    print(f'Features: {feature_cols}')
    print(f'Train+Val ends at day {split_idx} | Test size: {N - split_idx} days')

    #  Build raw target array based on TARGET_MODE 
    if TARGET_MODE == 'price':
        raw_targets = close_values.copy()
    elif TARGET_MODE == 'return':
        raw_targets = np.zeros(N, dtype='float32')
        raw_targets[1:] = np.diff(close_values) / close_values[:-1]
    elif TARGET_MODE == 'log_return':
        raw_targets = np.zeros(N, dtype='float32')
        raw_targets[1:] = np.log(close_values[1:] / close_values[:-1])
    else:
        raise ValueError(f'Unknown TARGET_MODE: {TARGET_MODE}')

    # Fit scalers on TRAINING DATA ONLY (no leakage)
    feature_scaler = MinMaxScaler()
    feature_scaler.fit(feature_data[:split_idx])          # TRAIN ONLY
    features_scaled = feature_scaler.transform(feature_data).astype('float32')

    if TARGET_MODE == 'price':
        target_scaler = MinMaxScaler()
    else:
        target_scaler = StandardScaler()
    target_scaler.fit(raw_targets[:split_idx].reshape(-1, 1))  # TRAIN ONLY
    targets_scaled = (
        target_scaler
        .transform(raw_targets.reshape(-1, 1))
        .reshape(-1)
        .astype('float32')
    )

    #  LEAKAGE ASSERTIONS 
    assert hasattr(feature_scaler, 'data_min_'), \
        'Feature scaler was not fitted — aborting.'
    assert split_idx > WINDOW_SIZE, \
        'split_idx must be larger than WINDOW_SIZE.'
    # Training sequences: predicting days [WINDOW_SIZE, …, split_idx-1]
    # None of these days are in the test set [split_idx, …, N-1].
    assert WINDOW_SIZE + (split_idx - WINDOW_SIZE) - 1 < split_idx, \
        'Training sequences would reach into test period — aborting.'
    print('✓ Leakage assertions passed.')

    #  Build sequences 
    x_tv, y_tv = make_sequences(
        features_scaled[:split_idx],
        targets_scaled[:split_idx],
        WINDOW_SIZE,
    )

    # Validation follows training chronologically (no shuffle).
    val_size   = max(1, int(len(x_tv) * VAL_RATIO))
    train_size = len(x_tv) - val_size

    x_train, y_train = x_tv[:train_size], y_tv[:train_size]
    x_val,   y_val   = x_tv[train_size:], y_tv[train_size:]

    test_ctx_f = features_scaled[split_idx - WINDOW_SIZE :]
    test_ctx_t = targets_scaled [split_idx - WINDOW_SIZE :]
    x_test, _ = make_sequences(test_ctx_f, test_ctx_t, WINDOW_SIZE)

    print(f'Train sequences : {train_size}')
    print(f'Val   sequences : {val_size}')
    print(f'Test  sequences : {len(x_test)}')

    #  Previous actual prices for reconstruction and direction accuracy
    prev_prices_test  = close_values[split_idx - 1 : N - 1]       
    actual_close      = close_values[split_idx :]                  

    prev_prices_train = close_values[WINDOW_SIZE - 1 : WINDOW_SIZE - 1 + train_size]
    actual_close_train = close_values[WINDOW_SIZE    : WINDOW_SIZE     + train_size]

    assert len(prev_prices_test) == len(actual_close) == len(x_test), \
        'Test array length mismatch — check indexing.'

    # Train LSTM 
    print('\nTraining LSTM...')
    lstm_model, history = build_and_train_lstm(
        x_train, y_train, x_val, y_val,
        lookback=WINDOW_SIZE,
        feature_count=feature_count,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    best_epoch = int(np.argmin(history.history['val_loss']) + 1)
    print(f'Best epoch: {best_epoch}')

    # LSTM predictions (test and in-sample)
    lstm_test_preds  = predict_and_reconstruct(
        lstm_model, x_test,  target_scaler, TARGET_MODE, prev_prices_test
    )
    lstm_train_preds = predict_and_reconstruct(
        lstm_model, x_train, target_scaler, TARGET_MODE, prev_prices_train
    )

    #  Baselines
    persistence_preds = persistence_baseline(close_values, split_idx)
    ridge_preds       = ridge_regression_baseline(
        features_scaled, close_values, split_idx, WINDOW_SIZE,
        close_idx, feature_count, feature_scaler,
    )
    ma_preds = moving_average_baseline(close_values, split_idx, WINDOW_SIZE)

    # Compute all metrics 
    metrics_lstm_test  = compute_metrics(actual_close,       lstm_test_preds)
    metrics_lstm_train = compute_metrics(actual_close_train, lstm_train_preds)
    metrics_persistence = compute_metrics(actual_close, persistence_preds)
    metrics_ridge       = compute_metrics(actual_close, ridge_preds)
    metrics_ma          = compute_metrics(actual_close, ma_preds)

    da_lstm        = directional_accuracy(actual_close, lstm_test_preds,  prev_prices_test)
    da_persistence = directional_accuracy(actual_close, persistence_preds, prev_prices_test)
    da_ridge       = directional_accuracy(actual_close, ridge_preds,       prev_prices_test)
    da_ma          = directional_accuracy(actual_close, ma_preds,          prev_prices_test)

    # Print results 
    print(f'\n{"─"*70}')
    print(' IN-SAMPLE performance (train set)')
    print(f'{"─"*70}')
    print(
        f'  LSTM  RMSE={metrics_lstm_train["RMSE"]:.4f} | '
        f'MAE={metrics_lstm_train["MAE"]:.4f} | '
        f'MAPE={metrics_lstm_train["MAPE"]:.4f}%'
    )

    print(f'\n{"─"*70}')
    print(' OUT-OF-SAMPLE performance (test set)')
    print(f'{"─"*70}')
    print(f'  {"Model":<16} {"RMSE":>9} {"MAE":>9} {"MAPE%":>9} {"DirAcc%":>10}')
    print(f'  {"─"*56}')

    rows = {
        'LSTM':        (metrics_lstm_test,   da_lstm),
        'Persistence': (metrics_persistence, da_persistence),
        'Ridge':       (metrics_ridge,       da_ridge),
        f'MA-{WINDOW_SIZE}': (metrics_ma,   da_ma),
    }
    for name, (m, da) in rows.items():
        print(
            f'  {name:<16} {m["RMSE"]:>9.4f} {m["MAE"]:>9.4f} '
            f'{m["MAPE"]:>9.4f} {da:>10.1f}'
        )

    #  Interpretation 
    print(f'\n{"─"*70}')
    print(' INTERPRETATION')
    print(f'{"─"*70}')

    rmse_vs_persistence = (
        (metrics_persistence['RMSE'] - metrics_lstm_test['RMSE'])
        / metrics_persistence['RMSE'] * 100.0
    )
    rmse_vs_ridge = (
        (metrics_ridge['RMSE'] - metrics_lstm_test['RMSE'])
        / metrics_ridge['RMSE'] * 100.0
    )
    overfitting_ratio = (
        metrics_lstm_test['RMSE'] / metrics_lstm_train['RMSE']
        if metrics_lstm_train['RMSE'] > 1e-8 else float('inf')
    )

    # RMSE vs persistence
    if rmse_vs_persistence > 5:
        print(f'  ✓ LSTM beats persistence by {rmse_vs_persistence:.1f}% RMSE — genuine forecasting signal present.')
    elif rmse_vs_persistence > 0:
        print(f'  △ LSTM marginally outperforms persistence (+{rmse_vs_persistence:.1f}% RMSE). Weak signal.')
    else:
        print(
            f'  ✗ LSTM does NOT outperform persistence ({rmse_vs_persistence:.1f}% RMSE).\n'
            f'    The model is exploiting autocorrelation, not genuine market patterns.'
        )

    # Directional accuracy
    if da_lstm > 55:
        print(f'  ✓ LSTM directional accuracy {da_lstm:.1f}% — reliable directional signal above 55% threshold.')
    elif da_lstm > 50:
        print(f'  △ LSTM directional accuracy {da_lstm:.1f}% — marginal signal just above 50% random baseline.')
    else:
        print(
            f'  ✗ LSTM directional accuracy {da_lstm:.1f}% — at or below random (50%).\n'
            f'    Model cannot reliably predict price direction.'
        )

    # vs Ridge (is LSTM capacity adding value?)
    if rmse_vs_ridge > 5:
        print(f'  ✓ LSTM beats Ridge by {rmse_vs_ridge:.1f}% RMSE — non-linear patterns contribute.')
    elif rmse_vs_ridge >= 0:
        print(f'  △ LSTM similar to Ridge ({rmse_vs_ridge:.1f}% RMSE gap). Non-linear capacity not clearly helping.')
    else:
        print(f'  ✗ Ridge outperforms LSTM ({-rmse_vs_ridge:.1f}% better RMSE). Overfitting likely.')

    # Overfitting indicator
    if overfitting_ratio > 3.0:
        print(
            f'  ⚠ Possible overfitting: test RMSE is {overfitting_ratio:.1f}× train RMSE.\n'
            f'    Consider more dropout, less capacity, or shorter training.'
        )
    else:
        print(f'  ✓ Train/Test RMSE ratio: {overfitting_ratio:.2f} (within acceptable range).')

    # Mode-specific warning
    if TARGET_MODE == 'price' and FEATURE_MODE == 'all':
        print(
            '\n  ⚠ WARNING: raw price target + all OHLCV features.\n'
            '    Predictions may look accurate purely due to persistence autocorrelation.\n'
            '    Try TARGET_MODE="return" and FEATURE_MODE="close_only" for a harder test.'
        )

    # Overall assessment
    print(f'\n  {"─"*60}')
    if rmse_vs_persistence > 5 and da_lstm > 52:
        print('  OVERALL: LSTM appears to provide genuine next-day forecasting signal.')
    elif rmse_vs_persistence > 0 or da_lstm > 50:
        print('  OVERALL: LSTM shows marginal improvement over baselines. Results are weak but not zero.')
    else:
        print(
            '  OVERALL: LSTM behaves as a price smoother/follower rather than a genuine\n'
            '  forecaster. It closely tracks prices but adds no predictive value beyond\n'
            '  the persistence baseline. This is consistent with the Efficient Market\n'
            '  Hypothesis for liquid equity indices.'
        )

    #  Save results CSV 
    results_df = pd.DataFrame([
        {'Model': name, **m, 'DirAcc%': da}
        for name, (m, da) in rows.items()
    ])
    results_df.to_csv('v3_results.csv', index=False)
    print(f'\n  Results saved → v3_results.csv')

    # Plots
    plot_predictions(actual_close, lstm_test_preds, persistence_preds, ridge_preds, ma_preds)
    plot_diagnostics(actual_close, lstm_test_preds, 'LSTM')
    plot_loss_curve(history)
    plot_metrics_comparison({
        'LSTM':        metrics_lstm_test,
        'Persistence': metrics_persistence,
        'Ridge':       metrics_ridge,
        f'MA-{WINDOW_SIZE}': metrics_ma,
    })


if __name__ == '__main__':
    main()
