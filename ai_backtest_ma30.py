import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
import datetime
import random
import ta
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ===========================
# ⚙️ 策略設定 (回歸高性能版)
# ===========================
INITIAL_CASH = 10000
START_DATE = "2024-01-01"
END_DATE = datetime.datetime.now().strftime("%Y-%m-%d")

TICKERS = [
    'NVDA', 'TSLA', 'AMZN', 'MSFT', 'GOOGL', 'META', 'AAPL', 
    'AMD', 'INTC', 'QCOM', 'AVGO', 'MU',                       
    'JPM', 'V', 'DIS', 'NFLX', 'COST', 'PEP', 'KO', 'JNJ'    
]

RETRAIN_EVERY_N_DAYS = 20  # 縮短週期，確保 AI 靈敏度
EPOCHS_PER_TRAIN = 10      # 增加訓練深度
LOOK_BACK = 60
BUY_PROB_THRESHOLD = 0.55

# ===========================

def build_heavy_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(100, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

def run_backtest_final():
    print(f"🚀 啟動高效能回測系統 | 初始金額: ${INITIAL_CASH}")
    
    full_data = {}
    download_start = (datetime.datetime.strptime(START_DATE, "%Y-%m-%d") - datetime.timedelta(days=1000)).strftime("%Y-%m-%d")
    
    for t in TICKERS:
        df = yf.download(t, start=download_start, end=END_DATE, progress=False)
        if df.empty: continue
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        df['EMA20'] = ta.trend.ema_indicator(df['Close'], window=20)
        df['EMA60'] = ta.trend.ema_indicator(df['Close'], window=60)
        df['MA30'] = ta.trend.sma_indicator(df['Close'], window=30)
        df['MA30_Slope'] = df['MA30'].diff()
        df.ffill(inplace=True); df.bfill(inplace=True)
        full_data[t] = df

    portfolio = {"cash": INITIAL_CASH, "holdings": []}
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq='B')
    model_cache = {}
    scaler = MinMaxScaler()

    for idx, current_date in enumerate(dates):
        date_str = current_date.strftime("%Y-%m-%d")
        market_prices = {t: full_data[t].loc[current_date]['Close'] for t in TICKERS if t in full_data and current_date in full_data[t].index}
        if not market_prices: continue

        # --- 計算當前實時總資產 ---
        current_equity = portfolio["cash"] + sum(h["Shares"] * market_prices.get(h["Ticker"], h["Entry"]) for h in portfolio["holdings"])

        # --- 賣出檢查 ---
        for h in portfolio["holdings"][:]:
            t = h["Ticker"]
            if t in market_prices:
                curr_p = market_prices[t]
                h["Highest"] = max(h["Highest"], curr_p)
                drop = (curr_p - h["Highest"]) / h["Highest"]
                ema20 = full_data[t].loc[current_date]['EMA20']
                
                reason = None
                if curr_p <= h["Stop"]: reason = "🛑 ATR止損"
                elif curr_p > ema20 and drop <= -0.05: reason = "📉 強勢回調"
                elif curr_p <= ema20 and drop <= -0.025: reason = "🏃 弱勢反彈"
                elif (current_date - h["BuyDate"]).days >= 20: reason = "⏰ 到期賣出"

                if reason:
                    portfolio["cash"] += (h["Shares"] * curr_p) - 2.0
                    portfolio["holdings"].remove(h)
                    temp_equity = portfolio["cash"] + sum(x["Shares"] * market_prices.get(x["Ticker"], x["Entry"]) for x in portfolio["holdings"])
                    print(f"[{date_str}] 賣出 {t} | 原因: {reason} | 價格: {curr_p:.2f} | 帳戶總額: ${temp_equity:.2f}")

        # --- 買入檢查 ---
        if len(portfolio["holdings"]) < 3:
            candidates = []
            for t in TICKERS:
                if t in market_prices and t not in [x['Ticker'] for x in portfolio['holdings']]:
                    row = full_data[t].loc[current_date]
                    if row['MA30_Slope'] > 0 and market_prices[t] > row['MA30'] * 1.01:
                        score = market_prices[t] / row['EMA60']
                        candidates.append((t, score))
            
            candidates.sort(key=lambda x: x[1], reverse=True)

            for t, _ in candidates[:3]:
                df_past = full_data[t][full_data[t].index < current_date]
                if len(df_past) < LOOK_BACK + 20: continue
                
                if t not in model_cache or (current_date - model_cache[t]['date']).days >= RETRAIN_EVERY_N_DAYS:
                    feat_cols = ['Close', 'Volume', 'RSI', 'ATR', 'MA30']
                    scaled_data = scaler.fit_transform(df_past[feat_cols].values)
                    x_train, y_train = [], []
                    for i in range(LOOK_BACK, len(scaled_data)-10):
                        x_train.append(scaled_data[i-LOOK_BACK:i])
                        roi = (df_past['Close'].iloc[i+1:i+11].max() - df_past['Close'].iloc[i]) / df_past['Close'].iloc[i]
                        y_train.append(1 if roi > 0.03 else 0)
                    
                    if not x_train: continue
                    model = build_heavy_model((LOOK_BACK, 5))
                    model.fit(np.array(x_train), np.array(y_train), epochs=EPOCHS_PER_TRAIN, verbose=0)
                    model_cache[t] = {'model': model, 'date': current_date, 'scaler': scaler}

                m_info = model_cache[t]
                feat_now = df_past[['Close', 'Volume', 'RSI', 'ATR', 'MA30']].values[-LOOK_BACK:]
                scaled_now = m_info['scaler'].transform(feat_now).reshape(1, LOOK_BACK, 5)
                prob = float(m_info['model'].predict(scaled_now, verbose=0)[0][0])

                if prob > BUY_PROB_THRESHOLD:
                    # 🚀 動態倉位
                    alloc = 0.4 if prob > 0.7 else (0.3 if prob > 0.6 else 0.2)
                    buy_val = min(current_equity * alloc, portfolio["cash"])
                    if buy_val > market_prices[t] + 10:
                        shares = (buy_val - 2.0) / market_prices[t]
                        portfolio["cash"] -= (shares * market_prices[t] + 2.0)
                        portfolio["holdings"].append({
                            "Ticker": t, "Shares": shares, "Entry": market_prices[t], "Highest": market_prices[t],
                            "BuyDate": current_date, "Stop": market_prices[t] - (full_data[t].loc[current_date]['ATR'] * 2.5)
                        })
                        print(f"[{date_str}] 🚀 買入 {t} | AI信心: {prob*100:.1f}% | 倉位: {alloc*100}% | 帳戶總額: ${current_equity:.2f}")
                        break

    print(f"\n🏁 最終總報酬率: {((current_equity - INITIAL_CASH) / INITIAL_CASH) * 100:.2f}%")

if __name__ == "__main__":
    run_backtest_final()