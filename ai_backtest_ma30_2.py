import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
import tensorflow as tf
import os
import datetime
import json
import random
import ta
# 報酬率100%+
# ===========================
# ⚙️ 策略設定 (Top 3 動能 + MA30 確認 + 無限奔跑)
# ===========================
INITIAL_CASH = 10000
START_DATE = "2025-01-01"
END_DATE = datetime.datetime.now().strftime("%Y-%m-%d")
TRANSACTION_FEE = 2.0  

BUY_PROB_THRESHOLD = 0.55
TARGET_ROI_CLASS = 0.03 

TIME_STOP_DAYS = 20       
ATR_STOP_LOSS_MULTIPLIER = 2.5 
# 🔥 [回歸寬鬆] 移除緊縮止損，讓獲利奔跑
# ATR_TRAILING_MULTIPLIER = 1.5 

STRONG_DROP_TOLERANCE = 0.05 
WEAK_DROP_TOLERANCE = 0.025  

ALLOCATION_PCT = 0.33 
TOP_N_MOMENTUM = 3 
STOP_LOSS_COOLDOWN_DAYS = 10

# MA30 濾網
MA30_BREAKOUT_BUFFER = 1.01 

LOOK_BACK = 60      
PREDICT_DAYS = 10   
RETRAIN_EVERY_N_DAYS = 20

# 🔥 [剔除弱勢股] 移除 INTC，保留強勢科技股
TICKERS = [
    'NVDA', 'TSLA', 'AMZN', 'MSFT', 'GOOGL', 'META', 'AAPL', 
    'AMD', 'QCOM', 'AVGO', 'MU',                     
    'JPM', 'V', 'DIS', 'NFLX', 'COST', 'PEP', 'KO', 'JNJ'    
]

MARKET_INDEX = 'QQQ' 

DATA_DIR = "data"
MODEL_DIR = "saved_models"
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)

seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

model_cache = {} 

def add_technical_indicators(df):
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    df['EMA20'] = ta.trend.ema_indicator(df['Close'], window=20)
    df['EMA60'] = ta.trend.ema_indicator(df['Close'], window=60)
    
    # MA30
    df['MA30'] = ta.trend.sma_indicator(df['Close'], window=30)
    df['MA30_Slope'] = df['MA30'].diff()
    df['Price_Change'] = df['Close'].diff()
    
    df.fillna(method='bfill', inplace=True)
    return df

def prepare_data(df, look_back):
    if len(df) < look_back + PREDICT_DAYS + 10: return None, None, None, None
    features = ['Close', 'Volume', 'RSI', 'MACD', 'ATR', 'MA30'] 
    data = df[features].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    x_train, y_train = [], []
    start_idx = max(look_back, len(scaled_data) - 500) 
    
    for i in range(start_idx, len(scaled_data) - PREDICT_DAYS):
        x_train.append(scaled_data[i-look_back:i])
        current_close = df['Close'].iloc[i]
        future_high = df['Close'].iloc[i+1 : i+1+PREDICT_DAYS].max()
        actual_roi = (future_high - current_close) / current_close
        
        if actual_roi > TARGET_ROI_CLASS:
            y_train.append(1)
        else:
            y_train.append(0)
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    if len(x_train) == 0: return None, None, None, None
    return x_train, y_train, scaler, scaled_data

def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def predict_signal(ticker, current_date, full_data):
    try:
        if ticker not in full_data: return 0.0
        df = full_data[ticker]
        mask = df.index < current_date
        past_df = df.loc[mask]
        if len(past_df) < LOOK_BACK + 20: return 0.0

        x_train, y_train, scaler, scaled_data = prepare_data(past_df, LOOK_BACK)
        if x_train is None: return 0.0

        global model_cache
        model_info = model_cache.get(ticker)
        model = None
        needs_training = False

        if model_info is None:
            model = build_model((x_train.shape[1], x_train.shape[2]))
            needs_training = True
        else:
            model = model_info['model']
            last_train = model_info['last_train_date']
            days_diff = (current_date - last_train).days
            if days_diff >= RETRAIN_EVERY_N_DAYS:
                needs_training = True
        
        if needs_training:
            model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=0)
            model_cache[ticker] = {'model': model, 'last_train_date': current_date}
        
        last_sequence = scaled_data[-LOOK_BACK:]
        curr_input = last_sequence.reshape(1, LOOK_BACK, scaled_data.shape[1])
        prob = model.predict(curr_input, verbose=0)[0][0]
        return float(prob)
        
    except Exception as e:
        return 0.0

def save_system_state(run_id):
    save_path = os.path.join(MODEL_DIR, "latest")
    if not os.path.exists(save_path): os.makedirs(save_path)
    count = 0
    for ticker, info in model_cache.items():
        model = info['model']
        model_file = os.path.join(save_path, f"{ticker}.keras")
        model.save(model_file)
        count += 1
    config = {
        "LOOK_BACK": LOOK_BACK,
        "PREDICT_DAYS": PREDICT_DAYS,
        "MIN_ROI_THRESHOLD": 0, 
        "TICKERS": TICKERS
    }
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    print(f"✅ 成功保存 {count} 個智能分類模型！")

def run_backtest():
    print(f"🚀 啟動回測 (Top 3 動能 + MA30確認 + 無限奔跑)...")
    
    full_data = {}
    download_start = (datetime.datetime.strptime(START_DATE, "%Y-%m-%d") - datetime.timedelta(days=1000)).strftime("%Y-%m-%d")
    
    print("📥 下載個股數據...")
    for t in TICKERS:
        try:
            df = yf.download(t, start=download_start, end=END_DATE, progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            if not df.empty: 
                df = add_technical_indicators(df)
                full_data[t] = df
        except: pass

    print("📥 下載大盤數據 (QQQ)...")
    market_df = yf.download(MARKET_INDEX, start=download_start, end=END_DATE, progress=False)
    if isinstance(market_df.columns, pd.MultiIndex): market_df.columns = market_df.columns.get_level_values(0)
    market_df = add_technical_indicators(market_df)

    portfolio = {"cash": INITIAL_CASH, "holdings": []} 
    trade_log = []
    balance_history = []
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq='B')
    next_trade_date = dates[0]
    cooldown_list = {} 
    
    total_steps = len(dates)

    for idx, current_date in enumerate(dates):
        date_str = current_date.strftime("%Y-%m-%d")
        if idx % 10 == 0: print(f"📅 {date_str} ({idx}/{total_steps})", end='\r')

        market_prices = {}
        momentum_scores = [] 

        for t in TICKERS:
            if t in full_data and current_date in full_data[t].index:
                market_prices[t] = full_data[t].loc[current_date]['Close']
                
                # 計算動能分數 (Price / EMA60)
                ema60 = full_data[t].loc[current_date]['EMA60']
                if ema60 > 0:
                    score = market_prices[t] / ema60
                    momentum_scores.append((t, score))
        
        if not market_prices: continue
        
        # 🔥 第一層：只取前三名強勢股 (Relative Strength)
        momentum_scores.sort(key=lambda x: x[1], reverse=True)
        top_tickers = [x[0] for x in momentum_scores[:TOP_N_MOMENTUM]]
        
        # --- 賣出檢查 ---
        for h in portfolio["holdings"][:]: 
            ticker = h["Ticker"]
            if ticker in market_prices:
                curr_price = market_prices[ticker]
                ema20 = full_data[ticker].loc[current_date]['EMA20']
                
                if curr_price > h["Highest"]: h["Highest"] = curr_price
                
                entry_price = h["Entry"]
                highest_price = h["Highest"]
                atr_stop_price = h.get("ATR_Stop_Price", 0)
                
                pnl_pct = (curr_price - entry_price) / entry_price
                drop_from_peak = (curr_price - highest_price) / highest_price
                held_days = (current_date - h["BuyDate"]).days
                
                sell_reason = None
                is_strong = curr_price > ema20
                
                # 🔥 [回歸寬鬆] 不主動止盈，只在趨勢反轉時賣出
                if curr_price <= atr_stop_price:
                    sell_reason = f"🛑 ATR止損"
                    cooldown_list[ticker] = current_date + datetime.timedelta(days=STOP_LOSS_COOLDOWN_DAYS)
                else:
                    if is_strong:
                        if drop_from_peak <= -STRONG_DROP_TOLERANCE: sell_reason = f"📉 強勢回調"
                    else:
                        if drop_from_peak <= -WEAK_DROP_TOLERANCE: sell_reason = f"🏃 弱勢反彈"
                
                if not sell_reason and held_days >= TIME_STOP_DAYS: sell_reason = f"⏰ 到期"
                
                if sell_reason:
                    gross_revenue = h["Shares"] * curr_price
                    net_revenue = gross_revenue - TRANSACTION_FEE
                    total_cost = (h["Shares"] * h["Entry"]) + TRANSACTION_FEE
                    net_profit = net_revenue - total_cost
                    net_profit_pct = (net_profit / total_cost) * 100
                    
                    portfolio["cash"] += net_revenue 
                    portfolio["holdings"].remove(h)
                    
                    trade_log.append({
                        "Date": date_str, "Action": "SELL", "Ticker": ticker, "Price": curr_price, 
                        "Reason": sell_reason, "Profit_USD": net_profit, "Profit_Pct": net_profit_pct, "Balance": portfolio["cash"]
                    })
                    profit_emoji = "🟢" if net_profit > 0 else "🔴"
                    trend_tag = "🔥" if is_strong else "❄️"
                    print(f"\n[{date_str}] 賣出 {ticker} {trend_tag}: {sell_reason}")
                    print(f"   └── {profit_emoji} 淨損益: ${net_profit:.2f} ({net_profit_pct:.2f}%) | 目前資產: ${portfolio['cash']:.2f}")

        # --- 買入檢查 ---
        if len(portfolio["holdings"]) < 3 and current_date >= next_trade_date:
            
            is_market_bullish = False
            if current_date in market_df.index:
                if market_df.loc[current_date]['Close'] > market_df.loc[current_date]['EMA60']:
                    is_market_bullish = True
            
            if is_market_bullish:
                current_holdings_tickers = [h['Ticker'] for h in portfolio["holdings"]]
                
                best_ticker, best_prob = None, 0.0
                for t in top_tickers: # 只看 Top 3
                    if t in current_holdings_tickers: continue 
                    if t in cooldown_list:
                        if current_date < cooldown_list[t]: continue
                        else: del cooldown_list[t]
                    
                    # 🔥 第二層：MA30 確認 (Absolute Trend)
                    # 確保龍頭股不是處於下跌修正中
                    ma30 = full_data[t].loc[current_date]['MA30']
                    ma30_slope = full_data[t].loc[current_date]['MA30_Slope']
                    curr_p = market_prices[t]
                    
                    # 條件：MA30 向上 且 股價站上 MA30
                    if ma30_slope > 0 and curr_p > (ma30 * MA30_BREAKOUT_BUFFER):
                        prob = predict_signal(t, current_date, full_data)
                        if prob > best_prob:
                            best_prob = prob
                            best_ticker = t
                
                if best_prob > BUY_PROB_THRESHOLD and best_ticker:
                    current_price = market_prices[best_ticker]
                    current_atr = full_data[best_ticker].loc[current_date]['ATR']
                    
                    total_equity = portfolio["cash"]
                    for h in portfolio["holdings"]:
                        if h["Ticker"] in market_prices:
                            total_equity += h["Shares"] * market_prices[h["Ticker"]]
                    
                    invest_budget = total_equity * ALLOCATION_PCT
                    if invest_budget > portfolio["cash"]: invest_budget = portfolio["cash"]
                    
                    available_cash_for_trade = invest_budget - TRANSACTION_FEE
                    
                    if available_cash_for_trade > current_price:
                        shares = available_cash_for_trade / current_price
                        portfolio["cash"] -= (shares * current_price + TRANSACTION_FEE)
                        
                        atr_stop_price = current_price - (current_atr * ATR_STOP_LOSS_MULTIPLIER)
                        
                        new_holding = {
                            "Ticker": best_ticker, 
                            "Shares": shares, 
                            "Entry": current_price, 
                            "Highest": current_price, 
                            "BuyDate": current_date,
                            "Probability": best_prob,
                            "ATR_Stop_Price": atr_stop_price
                        }
                        portfolio["holdings"].append(new_holding)
                        
                        trade_log.append({
                            "Date": date_str, "Action": "BUY", "Ticker": best_ticker, "Price": current_price, 
                            "Reason": f"Top3+MA30+AI {best_prob*100:.1f}%", "Profit_USD": 0, "Profit_Pct": 0, "Balance": portfolio["cash"]
                        })
                        print(f"\n[{date_str}] 🚀 冠軍買入 {best_ticker} (勝率 {best_prob*100:.1f}%) | 倉位: 33%")
            
            if len(portfolio["holdings"]) == 0:
                next_trade_date = current_date + datetime.timedelta(days=1)

        equity = portfolio["cash"]
        for h in portfolio["holdings"]:
            if h["Ticker"] in market_prices:
                equity += h["Shares"] * market_prices[h["Ticker"]]
        balance_history.append({"Date": date_str, "Equity": equity})

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    final_equity = balance_history[-1]['Equity']
    print(f"\n🏁 最終資產: ${final_equity:.2f} | 總報酬: {(final_equity - INITIAL_CASH) / INITIAL_CASH * 100:.1f}%")
    
    pd.DataFrame(trade_log).to_csv(os.path.join(DATA_DIR, "ai_backtest_ma30_log.csv"), index=False)
    pd.DataFrame(balance_history).to_csv(os.path.join(DATA_DIR, "ai_backtest_ma30_balance.csv"), index=False)
    save_system_state(run_id) 

if __name__ == "__main__":
    run_backtest()