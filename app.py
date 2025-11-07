# app.py
import os
import io
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

# Try to use a valid matplotlib style
try:
    plt.style.use('seaborn-v0_8')
except Exception:
    plt.style.use('default')

st.set_page_config(layout="wide", page_title="Stock Direction Predictor — (ensemble)")

# ---------------------------
# Helpers: Feature engineering
# ---------------------------
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features for the model. Input df must contain columns:
    'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'
    Returns dataframe with additional feature columns and drops NA rows.
    """
    out = df.copy()
    # ensure numeric
    for c in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors='coerce')

    adj = out['Adj Close']

    # Moving averages
    out['SMA_20'] = adj.rolling(20).mean()
    out['EMA_20'] = adj.ewm(span=20, adjust=False).mean()

    # MACD
    ema12 = adj.ewm(span=12, adjust=False).mean()
    ema26 = adj.ewm(span=26, adjust=False).mean()
    out['MACD'] = ema12 - ema26
    out['MACD_signal'] = out['MACD'].ewm(span=9, adjust=False).mean()

    # RSI (try to use ta if installed, else fallback simple)
    try:
        import ta
        out['RSI_14'] = ta.momentum.rsi(adj, window=14)
    except Exception:
        delta = adj.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        out['RSI_14'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    rolling20 = adj.rolling(20)
    out['BB_high'] = rolling20.mean() + 2 * rolling20.std()
    out['BB_low'] = rolling20.mean() - 2 * rolling20.std()
    out['BB_width'] = out['BB_high'] - out['BB_low']

    # Lagged returns and rolling stats
    for lag in range(1, 6):
        out[f'return_lag_{lag}'] = adj.pct_change(lag)

    out['mean_ret_5'] = adj.pct_change().rolling(5).mean()
    out['mean_ret_10'] = adj.pct_change().rolling(10).mean()
    out['vol_10'] = adj.pct_change().rolling(10).std()
    out['vol_20'] = adj.pct_change().rolling(20).std()

    out['close_open_ratio'] = out['Close'] / out['Open'] - 1
    out['high_low_ratio'] = out['High'] / out['Low'] - 1
    out['vol_change'] = out['Volume'].pct_change()

    # ADX & OBV best-effort (use ta if available)
    try:
        import ta
        out['ADX_14'] = ta.trend.adx(out['High'], out['Low'], adj, window=14)
        out['OBV'] = ta.volume.on_balance_volume(adj, out['Volume'])
    except Exception:
        out['ADX_14'] = out['Close'].rolling(14).apply(lambda x: 0)  # dummy
        out['OBV'] = (adj.diff() * out['Volume']).cumsum()

    out['dayofweek'] = out.index.dayofweek

    # final cleanup
    out = out.dropna()
    return out

# ---------------------------
# Helpers: model loading / predict
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_models_and_artifacts():
    """
    Loads models and artifacts (if present in working directory):
    - best_rf.pkl
    - best_xgb.pkl
    - best_lgbm_model.pkl
    - scaler.pkl (optional)
    - feature_selector.pkl (optional)
    Returns dict(models), scaler_or_none, selector_or_none
    """
    models = {}
    # Model filenames expected in repo root
    candidates = {
        'RandomForest': 'best_rf.pkl',
        'XGBoost': 'best_xgb.pkl',
        'LightGBM': 'best_lgbm_model.pkl'
    }
    for name, fname in candidates.items():
        if os.path.exists(fname):
            try:
                models[name] = joblib.load(fname)
            except Exception as e:
                st.warning(f"Could not load {fname}: {e}")

    # scaler
    scaler = None
    if os.path.exists('scaler.pkl'):
        try:
            scaler = joblib.load('scaler.pkl')
        except Exception as e:
            st.warning(f"Could not load scaler.pkl: {e}")
            scaler = None

    # selector
    selector = None
    if os.path.exists('feature_selector.pkl'):
        try:
            selector = joblib.load('feature_selector.pkl')
        except Exception as e:
            st.warning(f"Could not load feature_selector.pkl: {e}")
            selector = None

    return models, scaler, selector

def get_proba_from_model(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, 'decision_function'):
        df = model.decision_function(X)
        # normalize to 0-1
        df = np.asarray(df)
        df_min, df_max = df.min(), df.max()
        if df_max - df_min == 0:
            return np.zeros_like(df)
        return (df - df_min) / (df_max - df_min)
    else:
        # fallback: use predicted labels as 0/1 probabilities
        return model.predict(X).astype(float)

# ---------------------------
# Helpers: data fetch & normalize
# ---------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_and_prepare(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches OHLCV with yfinance, normalizes columns (handles MultiIndex),
    ensures 'Adj Close' exists (fallback to 'Close' if necessary), and returns dataframe.
    """
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()  # caller handles empty

    # collapse multiindex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.droplevel(1)
        except Exception:
            df.columns = df.columns.get_level_values(0)

    # normalize common alt names
    if 'Adj_Close' in df.columns and 'Adj Close' not in df.columns:
        df.rename(columns={'Adj_Close': 'Adj Close'}, inplace=True)
    if 'AdjClose' in df.columns and 'Adj Close' not in df.columns:
        df.rename(columns={'AdjClose': 'Adj Close'}, inplace=True)

    # ensure Adj Close exists
    if 'Adj Close' not in df.columns:
        if 'Close' in df.columns:
            df['Adj Close'] = df['Close'].copy()

    # required columns check
    required = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        # return original df but empty so caller can display message
        return pd.DataFrame()

    # keep required
    df = df[required].copy()
    # drop rows with NaN
    df = df.dropna()
    return df

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Stock Direction Predictor — (ensemble)")

with st.sidebar:
    st.header("Controls")
    ticker = st.text_input("Ticker (NSE format)", value="HDFCBANK.NS")
    end_date = st.date_input("End date", value=datetime.today().date())
    start_date = st.date_input("Start date", value=end_date - timedelta(days=365*2))
    smooth_days = st.selectbox("Smoothing window for quick evaluation (days)", options=[1,3,5], index=1)
    threshold_input = st.slider("Ensemble threshold (probability)", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
    run_button = st.button("Fetch & Predict")

# Load models & artifacts
models, scaler, selector = load_models_and_artifacts()

# Show loaded models
with st.sidebar.expander("Loaded artifacts"):
    if models:
        st.write("Models found:")
        for k in models.keys():
            st.write("-", k)
    else:
        st.warning("No model files (best_rf.pkl, best_xgb.pkl, best_lgbm_model.pkl) were found in the app folder.")
    st.write("Scaler:", "found" if scaler is not None else "not found")
    st.write("Selector:", "found" if selector is not None else "not found")

# Main: run when user clicks
if run_button:
    st.info(f"Fetching {ticker} from {start_date} to {end_date} ...")
    df = fetch_and_prepare(ticker, str(start_date), str(end_date))
    if df.empty:
        st.error("No data returned or required columns missing. Ensure ticker is NSE format like 'HDFCBANK.NS' and the date range is valid.")
        st.stop()

    st.success(f"Fetched {len(df)} rows. Computing features ...")
    feats = compute_features(df)
    if feats.empty:
        st.error("No rows after feature engineering (not enough history). Try an earlier start date or a different ticker.")
        st.stop()

    # Determine feature list (prefer saved selector, else default)
    default_features = [
        'SMA_20','EMA_20','RSI_14','MACD','MACD_signal','BB_width',
        'return_lag_1','return_lag_2','return_lag_3','return_lag_4','return_lag_5',
        'mean_ret_5','mean_ret_10','vol_10','vol_20','close_open_ratio','high_low_ratio',
        'vol_change','ADX_14','OBV','dayofweek','Volume'
    ]

    if selector is not None and hasattr(selector, 'get_support'):
        # We need the original feature names used when selector was fitted.
        # Many selector objects don't store column names; if they do, use them. Otherwise, fallback to defaults.
        try:
            # if selector was saved together with feature names, it might be a dict; try to be robust
            if isinstance(selector, dict) and 'feature_names' in selector:
                selected_features = selector['feature_names']
            else:
                # fallback: assume selector was fit on same default feature ordering
                mask = selector.get_support()
                # if size mismatch between mask and available features, fallback to defaults
                avail = [c for c in default_features if c in feats.columns]
                if len(mask) == len(avail):
                    selected_features = [f for f, m in zip(avail, mask) if m]
                else:
                    selected_features = [c for c in default_features if c in feats.columns]
        except Exception:
            selected_features = [c for c in default_features if c in feats.columns]
    else:
        selected_features = [c for c in default_features if c in feats.columns]

    if len(selected_features) == 0:
        st.error("No valid features available for prediction. Available columns: " + ", ".join(list(feats.columns[:50])))
        st.stop()

    st.write("Using features:", selected_features)

    # ----------------------
    # Robust scaling routine
    # ----------------------
    X = feats[selected_features].copy()

    # 1) ensure numeric dtype
    X = X.apply(pd.to_numeric, errors='coerce')

    # 2) replace inf/-inf with NaN
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 3) quick gap-filling: forward then backward fill (for small missing runs)
    X = X.fillna(method='ffill').fillna(method='bfill')

    # 4) if still NaNs remain, fill with column medians (robust)
    if X.isna().any().any():
        medians = X.median()
        X = X.fillna(medians)
        st.warning("Some features had missing values after computation — filled with column medians.")

    # 5) final check for non-finite values (should be none)
    if not np.isfinite(X.to_numpy()).all():
        # as last resort, replace any remaining non-finite with 0 (should be extremely rare)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        st.warning("Replaced remaining non-finite feature values with 0 as last resort.")

    # 6) scaling: prefer loaded scaler if compatible; otherwise fit local scaler
    try:
        if scaler is not None:
            # check if scaler was fitted on same number of features
            try:
                expected_n_features = scaler.mean_.shape[0]
            except Exception:
                expected_n_features = None

            if expected_n_features is None or expected_n_features != X.shape[1]:
                st.warning("Saved scaler is incompatible with current feature set — fitting a new scaler locally.")
                local_scaler = StandardScaler()
                X_scaled = pd.DataFrame(local_scaler.fit_transform(X), index=X.index, columns=X.columns)
            else:
                # use the saved scaler (it may still error if column order differs)
                X_scaled = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
        else:
            local_scaler = StandardScaler()
            X_scaled = pd.DataFrame(local_scaler.fit_transform(X), index=X.index, columns=X.columns)
    except ValueError as e:
        # Catch persistent ValueErrors and attempt one final cleanup
        st.warning(f"Scaling failed with ValueError: {str(e)}. Attempting final cleanup and local scaling.")
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median()).fillna(0)
        local_scaler = StandardScaler()
        X_scaled = pd.DataFrame(local_scaler.fit_transform(X), index=X.index, columns=X.columns)
    except Exception as e:
        st.error("Unexpected error during scaling: " + str(e))
        st.stop()

    # If no models loaded, stop with message
    if not models:
        st.error("No models available to make predictions. Add model files (best_rf.pkl, best_xgb.pkl, best_lgbm_model.pkl) to the app folder and re-run.")
        st.stop()

    # Predict probabilities per model
    proba_df = pd.DataFrame(index=X_scaled.index)
    for name, model in models.items():
        try:
            proba_df[name] = get_proba_from_model(model, X_scaled)
        except Exception as e:
            st.warning(f"Failed to predict with {name}: {e}")

    if proba_df.shape[1] == 0:
        st.error("No model produced probabilities. Check model types and ensure they support predict_proba or decision_function.")
        st.stop()

    # Ensemble: average available probabilities
    proba_df['ensemble_proba'] = proba_df.mean(axis=1)

    # Optionally tune threshold on a small validation slice of the earliest portion
    # For simplicity, we allow user to override with slider; otherwise we use slider value
    threshold = float(threshold_input)

    proba_df['signal'] = (proba_df['ensemble_proba'] >= threshold).astype(int)

    # Show recent predictions
    st.subheader("Latest Predictions (ensemble probability & signal)")
    st.dataframe(proba_df[['ensemble_proba', 'signal']].tail(20).sort_index(ascending=False))

    # Plot price with predicted signals
    st.subheader("Price & Predicted Signals")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(feats.index, feats['Adj Close'], label='Adj Close')
    up_dates = proba_df[proba_df['signal'] == 1].index
    down_dates = proba_df[proba_df['signal'] == 0].index
    if len(up_dates) > 0:
        ax.scatter(up_dates, feats.loc[up_dates, 'Adj Close'], marker='^', label='Predicted Up', s=60)
    if len(down_dates) > 0:
        ax.scatter(down_dates, feats.loc[down_dates, 'Adj Close'], marker='v', label='Predicted Down', s=20)
    ax.legend()
    st.pyplot(fig)

    # Quick evaluation against smoothed target (for user's information only)
    st.subheader(f"Quick evaluation vs {smooth_days}-day smoothed future (informational)")
    future_mean = feats['Adj Close'].shift(-1).rolling(window=smooth_days).mean()
    tmp_target = (future_mean > feats['Adj Close']).astype(int).dropna()
    aligned_idx = proba_df.index.intersection(tmp_target.index)
    if len(aligned_idx) > 0:
        y_true = tmp_target.loc[aligned_idx]
        y_pred = proba_df.loc[aligned_idx, 'signal']
        st.write("Accuracy:", float(accuracy_score(y_true, y_pred)))
        st.write("Precision:", float(precision_score(y_true, y_pred, zero_division=0)))
        st.write("Recall:", float(recall_score(y_true, y_pred, zero_division=0)))
        st.write("F1:", float(f1_score(y_true, y_pred, zero_division=0)))

        # confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        st.pyplot(fig2)
    else:
        st.info("Not enough overlap between predictions and a smoothed future target to compute quick evaluation.")

    # ROC curve for ensemble
    st.subheader("ROC curve (ensemble)")
    try:
        fpr, tpr, _ = roc_curve(tmp_target.loc[aligned_idx], proba_df.loc[aligned_idx, 'ensemble_proba'])
        roc_auc = auc(fpr, tpr)
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        ax3.plot(fpr, tpr, label=f'Ensemble ROC (AUC={roc_auc:.3f})')
        ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax3.set_xlabel('False Positive Rate'); ax3.set_ylabel('True Positive Rate')
        ax3.legend()
        st.pyplot(fig3)
    except Exception:
        st.info("ROC unavailable (need True labels alignment).")

    # Show per-model short metrics (optional)
    st.subheader("Per-model sample probabilities (first 5 rows)")
    st.write(proba_df.drop(columns=['ensemble_proba', 'signal']).head())

    # Download CSV of ensemble probabilities and signals
    csv_bytes = proba_df[['ensemble_proba', 'signal']].to_csv().encode('utf-8')
    st.download_button("Download ensemble probabilities CSV", data=csv_bytes,
                       file_name=f"{ticker}_ensemble_probs.csv", mime="text/csv")

    # Save ensemble probs to disk (optional)
    try:
        out_df = proba_df[['ensemble_proba', 'signal']].copy()
        out_df.to_csv('ensemble_probs.csv')
    except Exception:
        pass

    st.success("Prediction complete.")
else:
    st.info("Enter parameters in the sidebar and click 'Fetch & Predict' to run the ensemble predictor.")
    st.write("Tip: Use NSE-style tickers like `HDFCBANK.NS`, `RELIANCE.NS`, `TCS.NS`.")
