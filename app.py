# app.py
import streamlit as st
st.set_page_config(layout="wide", page_title="Stock Direction Predictor")

import os
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, auc, classification_report)

# try to use a valid style
try:
    plt.style.use('seaborn-v0_8')
except Exception:
    plt.style.use('default')

# ---- helper functions: feature engineering (must match notebook) ----
def compute_features(df):
    # expects df with columns: Open, High, Low, Close, Adj Close, Volume; index = dates
    out = df.copy()
    adj = out['Adj Close']
    out['SMA_20'] = adj.rolling(20).mean()
    out['EMA_20'] = adj.ewm(span=20, adjust=False).mean()

    ema12 = adj.ewm(span=12, adjust=False).mean()
    ema26 = adj.ewm(span=26, adjust=False).mean()
    out['MACD'] = ema12 - ema26
    out['MACD_signal'] = out['MACD'].ewm(span=9, adjust=False).mean()
    # RSI via simple implementation fallback if ta not present in runtime
    try:
        import ta
        out['RSI_14'] = ta.momentum.rsi(adj, window=14)
    except Exception:
        # simple RSI implementation
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

    # lagged returns and rolling stats
    for lag in range(1,6):
        out[f'return_lag_{lag}'] = adj.pct_change(lag)
    out['mean_ret_5'] = adj.pct_change().rolling(5).mean()
    out['mean_ret_10'] = adj.pct_change().rolling(10).mean()
    out['vol_10'] = adj.pct_change().rolling(10).std()
    out['vol_20'] = adj.pct_change().rolling(20).std()
    out['close_open_ratio'] = out['Close'] / out['Open'] - 1
    out['high_low_ratio'] = out['High'] / out['Low'] - 1
    out['vol_change'] = out['Volume'].pct_change()
    # ADX & OBV require ta; guard if not available
    try:
        import ta
        out['ADX_14'] = ta.trend.adx(out['High'], out['Low'], adj, window=14)
        out['OBV'] = ta.volume.on_balance_volume(adj, out['Volume'])
    except Exception:
        out['ADX_14'] = out['Close'].rolling(14).apply(lambda x: 0)  # dummy small
        out['OBV'] = (adj.diff() * out['Volume']).cumsum()
    out['dayofweek'] = out.index.dayofweek
    out = out.dropna()
    return out

# ---- model loader ----
@st.cache_resource
def load_models():
    models = {}
    for name, filename in [('RandomForest', 'best_rf.pkl'),
                           ('XGBoost', 'best_xgb.pkl'),
                           ('LightGBM', 'best_lgbm_model.pkl')]:
        if os.path.exists(filename):
            try:
                models[name] = joblib.load(filename)
            except Exception as e:
                st.warning(f"Could not load {filename}: {e}")
    # scaler & selector
    scaler = None
    selector = None
    if os.path.exists('scaler.pkl'):
        try:
            scaler = joblib.load('scaler.pkl')
        except:
            scaler = None
    if os.path.exists('feature_selector.pkl'):
        try:
            selector = joblib.load('feature_selector.pkl')
        except:
            selector = None
    return models, scaler, selector

models, scaler, selector = load_models()

# ---- UI ----
st.title("Stock Direction Predictor — (ensemble)")

with st.sidebar:
    st.header("Control Panel")
    ticker = st.text_input("Ticker (NSE format, e.g. HDFCBANK.NS)", value="HDFCBANK.NS")
    end_date = st.date_input("End date", datetime.today().date())
    start_date = st.date_input("Start date", end_date - timedelta(days=365*2))
    smooth_target_days = st.selectbox("Smoothing window for target (used only for explanation)", [1,3,5], index=1)
    run_button = st.button("Fetch & Predict")

# show loaded models
st.sidebar.markdown("### Loaded models")
if models:
    for k in models.keys():
        st.sidebar.write(f"- {k}")
else:
    st.sidebar.warning("No model files found (best_rf.pkl, best_xgb.pkl, best_lgbm_model.pkl).")

if run_button:
    with st.spinner("Fetching data and computing features..."):
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty:
            st.error("No data fetched — check ticker and date range.")
        else:
            # ensure single-level columns
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    df.columns = df.columns.droplevel(1)
                except:
                    df.columns = df.columns.get_level_values(0)
            df = df[['Open','High','Low','Close','Adj Close','Volume']].dropna()
            feats = compute_features(df)
            st.success(f"Fetched {len(df)} rows; after feature engineering {len(feats)} rows remain.")

            # choose features (from selector if present else infer)
            if selector is not None and hasattr(selector, 'get_support'):
                # if selector was saved as scikit selector, use it to get features
                try:
                    all_feat_names = feats.columns.tolist()
                    mask = selector.get_support()
                    selected_features = [f for f, m in zip(all_feat_names, mask) if m]
                except Exception:
                    selected_features = feats.columns.tolist()
            else:
                # default feature set used in notebook
                default_features = ['SMA_20','EMA_20','RSI_14','MACD','MACD_signal','BB_width',
                                    'return_lag_1','return_lag_2','return_lag_3','return_lag_4','return_lag_5',
                                    'mean_ret_5','mean_ret_10','vol_10','vol_20','close_open_ratio','high_low_ratio',
                                    'vol_change','ADX_14','OBV','dayofweek','Volume']
                selected_features = [f for f in default_features if f in feats.columns]

            st.write("Using features:", selected_features)

            # Align X to features and scale if scaler present
            X = feats[selected_features].copy()
            if scaler is not None:
                try:
                    X_scaled = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
                except Exception:
                    # if scaler not fitted on same columns, re-fit locally (not ideal)
                    local_scaler = StandardScaler()
                    X_scaled = pd.DataFrame(local_scaler.fit_transform(X), index=X.index, columns=X.columns)
            else:
                local_scaler = StandardScaler()
                X_scaled = pd.DataFrame(local_scaler.fit_transform(X), index=X.index, columns=X.columns)

            # predictions from available models
            def get_proba(model, X_in):
                if hasattr(model, 'predict_proba'):
                    return model.predict_proba(X_in)[:,1]
                elif hasattr(model, 'decision_function'):
                    df_ = model.decision_function(X_in)
                    return (df_ - df_.min())/(df_.max() - df_.min())
                else:
                    return model.predict(X_in).astype(float)

            proba_df = pd.DataFrame(index=X_scaled.index)
            for name, model in models.items():
                try:
                    proba_df[name] = get_proba(model, X_scaled)
                except Exception as e:
                    st.warning(f"Prediction failed for {name}: {e}")

            if proba_df.shape[1] == 0:
                st.error("No model probabilities available. Place model .pkl files in the app folder or train models.")
            else:
                # ensemble average
                proba_df['ensemble_proba'] = proba_df.mean(axis=1)
                # threshold: default 0.5, you can tune by validation offline and hardcode here
                threshold = 0.5
                proba_df['signal'] = (proba_df['ensemble_proba'] >= threshold).astype(int)

                # show recent predictions
                st.subheader("Recent predictions (top rows)")
                st.dataframe(proba_df[['ensemble_proba','signal']].tail(20).sort_index(ascending=False))

                # Plot price with predicted signals
                st.subheader("Price & Predicted Signals")
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(feats.index, feats['Adj Close'], label='Adj Close')
                up_dates = proba_df[proba_df['signal'] == 1].index
                down_dates = proba_df[proba_df['signal'] == 0].index
                ax.scatter(up_dates, feats.loc[up_dates, 'Adj Close'], marker='^', label='Predicted Up', s=40)
                ax.scatter(down_dates, feats.loc[down_dates, 'Adj Close'], marker='v', label='Predicted Down', s=20)
                ax.legend()
                st.pyplot(fig)

                # If you have true next-day returns, compute toy metrics for the overlap period
                # we'll compute a smoothed "future_mean" like in the notebook to make a temporary target for evaluation
                future_mean = feats['Adj Close'].shift(-1).rolling(window=smooth_target_days).mean()
                tmp_target = (future_mean > feats['Adj Close']).astype(int).dropna()
                # align indexes
                aligned_idx = proba_df.index.intersection(tmp_target.index)
                if len(aligned_idx) > 0:
                    y_true = tmp_target.loc[aligned_idx]
                    y_pred = proba_df.loc[aligned_idx, 'signal']
                    st.subheader("Quick evaluation (smoothed target)")
                    st.write("Accuracy:", float(accuracy_score(y_true, y_pred)))
                    st.write("Precision:", float(precision_score(y_true, y_pred, zero_division=0)))
                    st.write("Recall:", float(recall_score(y_true, y_pred, zero_division=0)))
                    st.write("F1:", float(f1_score(y_true, y_pred, zero_division=0)))
                    # confusion matrix
                    cm = confusion_matrix(y_true, y_pred)
                    fig2, ax2 = plt.subplots(figsize=(4,3))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
                    ax2.set_xlabel('Predicted')
                    ax2.set_ylabel('Actual')
                    st.pyplot(fig2)

                # allow user to download ensemble probabilities CSV
                csv = proba_df[['ensemble_proba','signal']].to_csv().encode('utf-8')
                st.download_button("Download ensemble probabilities CSV", data=csv, file_name=f"{ticker}_ensemble_probs.csv", mime="text/csv")

                st.success("Done.")
