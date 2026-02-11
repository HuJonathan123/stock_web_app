import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input
import tensorflow as tf
import os
import datetime
import json
import random

# ===========================
# ⚙️ 策略設定 (階梯式動態止盈版)
# ===========================
INITIAL_CASH = 1000
START_DATE = "2025-01-01"
END_DATE = datetime.datetime.now().strftime("%Y-%m-%d")

# 1. 入選標準
MIN_ROI_THRESHOLD = 8.0   

# 2. 基礎風控
STOP_LOSS_PCT = 0.04      
TIME_STOP_DAYS = 20       

# 3. 階梯式動態止盈
TRAILING_ACTIVATION = 0.05 
TRAILING_DROP_PCT = 0.04   

# 4. 暴利鎖定
SUPER_PROFIT_PCT = 0.20    
SUPER_DROP_PCT = 0.02      

# 其他參數
LOOK_BACK = 60      
PREDICT_DAYS = 10   
RETRAIN_EVERY_N_DAYS = 20

TICKERS = [
    'NVDA', 'TSLA', 'AMZN', 'MSFT', 'GOOGL', 'META', 'AAPL', 
    'AMD', 'INTC', 'QCOM', 'AVGO', 'MU',                     
    'JPM', 'V', 'DIS', 'NFLX', 'COST', 'PEP', 'KO', 'JNJ'    
]

DATA_DIR = "data"
MODEL_DIR = "saved_models"
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)

# 🔒 固定種子
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# ===========================
# 1. AI 模型
# ===========================
model_cache = {} 

def prepare_data(df, look_back):
    if len(df) < look_back + 10: return None, None, None, None
    data = df.filter(['Close']).values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    x_train, y_train = [], []
    start_idx = max(look_back, len(scaled_data) - 300) 
    
    for i in range(start_idx, len(scaled_data)):
        x_train.append(scaled_data[i-look_back:i, 0])
        y_train.append(scaled_data[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    if len(x_train) == 0: return None, None, None, None
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    return x_train, y_train, scaler, scaled_data

def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_future_roi(ticker, current_date, full_data):
    try:
        if ticker not in full_data: return -999
        df = full_data[ticker]
        mask = df.index < current_date
        past_df = df.loc[mask]
        
        if len(past_df) < LOOK_BACK + 20: return -999

        x_train, y_train, scaler, scaled_data = prepare_data(past_df, LOOK_BACK)
        if x_train is None: return -999

        global model_cache
        model_info = model_cache.get(ticker)
        model = None
        needs_training = False

        if model_info is None:
            model = build_model((x_train.shape[1], 1))
            needs_training = True
        else:
            model = model_info['model']
            last_train = model_info['last_train_date']
            days_diff = (current_date - last_train).days
            if days_diff >= RETRAIN_EVERY_N_DAYS:
                needs_training = True
        
        if needs_training:
            model.fit(x_train, y_train, batch_size=16, epochs=3, verbose=0)
            model_cache[ticker] = {'model': model, 'last_train_date': current_date}
        
        last_sequence = scaled_data[-LOOK_BACK:]
        curr_input = last_sequence.reshape(1, LOOK_BACK, 1)
        preds = []
        for _ in range(PREDICT_DAYS):
            pred = model.predict(curr_input, verbose=0)
            preds.append(pred[0, 0])
            curr_input = np.append(curr_input[:, 1:, :], [[pred[0]]], axis=1)
            
        real_preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1))
        curr_price = past_df['Close'].iloc[-1]
        max_future = np.max(real_preds)
        
        if curr_price == 0: return -999
        return (max_future - curr_price) / curr_price * 100
    except:
        return -999

def save_system_state(run_id):
    # 保存到 saved_models/latest (方便網頁讀取)
    save_path = os.path.join(MODEL_DIR, "latest")
    if not os.path.exists(save_path): os.makedirs(save_path)
    
    print(f"\n💾 正在保存模型至: {save_path} ...")
    
    count = 0
    for ticker, info in model_cache.items():
        model = info['model']
        model_file = os.path.join(save_path, f"{ticker}.keras")
        model.save(model_file)
        count += 1
        
    config = {
        "LOOK_BACK": LOOK_BACK,
        "PREDICT_DAYS": PREDICT_DAYS,
        "MIN_ROI_THRESHOLD": MIN_ROI_THRESHOLD,
        "TICKERS": TICKERS
    }
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
        
    print(f"✅ 成功保存 {count} 個智能模型！")

def run_backtest():
    print(f"🚀 啟動回測...")
    full_data = {}
    download_start = (datetime.datetime.strptime(START_DATE, "%Y-%m-%d") - datetime.timedelta(days=400)).strftime("%Y-%m-%d")
    
    print("📥 下載數據中...")
    for t in TICKERS:
        try:
            df = yf.download(t, start=download_start, end=END_DATE, progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            if not df.empty: full_data[t] = df
        except: pass

    portfolio = {"cash": INITIAL_CASH, "holdings": None} 
    trade_log = []
    balance_history = []
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq='B')
    next_trade_date = dates[0]
    
    total_steps = len(dates)

    for idx, current_date in enumerate(dates):
        date_str = current_date.strftime("%Y-%m-%d")
        if idx % 10 == 0: print(f"📅 {date_str} ({idx}/{total_steps})", end='\r')

        market_prices = {}
        for t in TICKERS:
            if t in full_data and current_date in full_data[t].index:
                market_prices[t] = full_data[t].loc[current_date]['Close']
        
        if not market_prices: continue
        
        # 賣出檢查
        if portfolio["holdings"]:
            h = portfolio["holdings"]
            ticker = h["Ticker"]
            if ticker in market_prices:
                curr_price = market_prices[ticker]
                if curr_price > h["Highest"]: h["Highest"] = curr_price
                
                entry_price = h["Entry"]
                highest_price = h["Highest"]
                pnl_pct = (curr_price - entry_price) / entry_price
                max_pnl_pct = (highest_price - entry_price) / entry_price
                drop_from_peak = (curr_price - highest_price) / highest_price
                held_days = (current_date - h["BuyDate"]).days
                
                sell_reason = None
                if pnl_pct <= -STOP_LOSS_PCT: sell_reason = f"🛑 止損 ({pnl_pct*100:.1f}%)"
                elif max_pnl_pct >= SUPER_PROFIT_PCT and drop_from_peak <= -SUPER_DROP_PCT: sell_reason = f"🏆 暴利鎖定 ({drop_from_peak*100:.1f}%)"
                elif max_pnl_pct >= TRAILING_ACTIVATION and drop_from_peak <= -TRAILING_DROP_PCT: sell_reason = f"📉 波段止盈 ({drop_from_peak*100:.1f}%)"
                elif held_days >= TIME_STOP_DAYS: 
                    if pnl_pct > 0: sell_reason = f"⏰ 到期獲利 (+{pnl_pct*100:.1f}%)"
                    else: sell_reason = f"⏰ 到期平倉 ({pnl_pct*100:.1f}%)"
                
                if sell_reason:
                    revenue = h["Shares"] * curr_price
                    portfolio["cash"] = revenue
                    portfolio["holdings"] = None
                    trade_log.append({"Date": date_str, "Action": "SELL", "Ticker": ticker, "Price": curr_price, "Reason": sell_reason, "Balance": revenue})
                    print(f"\n[{date_str}] 賣出 {ticker}: {sell_reason} | 餘額: {revenue:.0f}")
                    next_trade_date = current_date

        # 買入檢查
        if portfolio["holdings"] is None and current_date >= next_trade_date:
            best_ticker, best_roi = None, -999
            for t in TICKERS:
                if t in market_prices:
                    roi = predict_future_roi(t, current_date, full_data)
                    if roi > best_roi: best_roi, best_ticker = roi, t
            
            if best_roi > MIN_ROI_THRESHOLD and best_ticker:
                buy_price = market_prices[best_ticker]
                shares = portfolio["cash"] / buy_price
                portfolio["holdings"] = {"Ticker": best_ticker, "Shares": shares, "Entry": buy_price, "Highest": buy_price, "BuyDate": current_date}
                portfolio["cash"] = 0
                trade_log.append({"Date": date_str, "Action": "BUY", "Ticker": best_ticker, "Price": buy_price, "Reason": f"AI信心 {best_roi:.1f}%", "Balance": 0})
                print(f"\n[{date_str}] 🚀 買入 {best_ticker} (預測 +{best_roi:.1f}%)")
            else:
                next_trade_date = current_date + datetime.timedelta(days=2) 

        equity = portfolio["cash"]
        if portfolio["holdings"]:
            if portfolio["holdings"]["Ticker"] in market_prices:
                equity = portfolio["holdings"]["Shares"] * market_prices[portfolio["holdings"]["Ticker"]]
        balance_history.append({"Date": date_str, "Equity": equity})

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    final_equity = balance_history[-1]['Equity']
    print(f"\n🏁 最終資產: ${final_equity:.2f} | 總報酬: {(final_equity - INITIAL_CASH) / INITIAL_CASH * 100:.1f}%")
    
    # 存檔供網頁使用
    pd.DataFrame(trade_log).to_csv(os.path.join(DATA_DIR, "ai_backtest_log.csv"), index=False)
    pd.DataFrame(balance_history).to_csv(os.path.join(DATA_DIR, "ai_backtest_balance.csv"), index=False)
    save_system_state(run_id) # 同時保存到 latest

if __name__ == "__main__":
    run_backtest()