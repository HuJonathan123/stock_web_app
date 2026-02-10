import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import os
import datetime

# ===========================
# ⚙️ 策略設定 (你可以調整這裡)
# ===========================
INITIAL_CASH = 1000
START_DATE = "2025-01-01"
END_DATE = "2026-02-10"

# AI 參數
LOOK_BACK = 60      # 回看 60 天
PREDICT_DAYS = 10   # 預測未來 10 天
MIN_ROI_THRESHOLD = 3.0 # 預測漲幅 > 3% 才買 (信心門檻)

# 止盈止損 (Vulture 風格)
TAKE_PROFIT_PCT = 0.15  # 賺 15% 止盈
STOP_LOSS_PCT = 0.08    # 虧 8% 止損
TIME_STOP_DAYS = 12     # 持有超過 12 天沒動靜就賣

# 股票池 (選波動大的才有肉吃)
TICKERS = ['NVDA', 'TSLA', 'AMZN', 'MSFT', 'GOOGL', 'AMD', 'META']

# 資料夾
DATA_DIR = "data"
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

# ===========================
# 1. 核心 AI 模型 (輕量化版)
# ===========================
def prepare_data(df, look_back):
    data = df.filter(['Close']).values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    x_train, y_train = [], []
    for i in range(look_back, len(scaled_data)):
        x_train.append(scaled_data[i-look_back:i, 0])
        y_train.append(scaled_data[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    return x_train, y_train, scaler, scaled_data

def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    # 稍微簡化模型以加快回測速度
    model.add(LSTM(50, return_sequences=False)) 
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_future_roi(ticker, current_date_str, look_back=60):
    # 獲取 "當時" 之前的數據
    # 為了訓練，我們需要往前抓足夠長的歷史 (比如 1 年)
    end_dt = datetime.datetime.strptime(current_date_str, "%Y-%m-%d")
    start_dt = end_dt - datetime.timedelta(days=400) # 抓前 400 天
    
    try:
        df = yf.download(ticker, start=start_dt.strftime("%Y-%m-%d"), end=current_date_str, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        if len(df) < 100: return -999 # 數據不足
        
        # 訓練
        x_train, y_train, scaler, scaled_data = prepare_data(df, look_back)
        model = build_model((x_train.shape[1], 1))
        # epochs=3 加快速度 (犧牲一點點準確度換取時間)
        model.fit(x_train, y_train, batch_size=16, epochs=3, verbose=0)
        
        # 預測
        last_sequence = scaled_data[-look_back:]
        curr_input = last_sequence.reshape(1, look_back, 1)
        
        preds = []
        for _ in range(PREDICT_DAYS):
            pred = model.predict(curr_input, verbose=0)
            preds.append(pred[0, 0])
            curr_input = np.append(curr_input[:, 1:, :], [[pred[0]]], axis=1)
            
        real_preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1))
        
        curr_price = df['Close'].iloc[-1]
        max_future = np.max(real_preds)
        roi = (max_future - curr_price) / curr_price * 100
        
        return roi
    except Exception as e:
        print(f"⚠️ {ticker} 預測失敗: {e}")
        return -999

# ===========================
# 2. 回測主引擎 (Walk-Forward)
# ===========================
def run_backtest():
    print(f"🚀 啟動 AI 回測 ({START_DATE} ~ {END_DATE})")
    print("⏳ 這會比較久，因為 AI 需要不斷重新訓練學習...")
    
    # 1. 先下載所有數據 (作為驗證答案)
    print("📥 下載驗證數據中...")
    full_data = {}
    for t in TICKERS:
        df = yf.download(t, start="2024-01-01", end=END_DATE, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        full_data[t] = df

    # 初始化帳戶
    portfolio = {"cash": INITIAL_CASH, "holdings": None} # None 或 {"Ticker": "NVDA", "Shares": 10, "Entry": 100, "Date": "..."}
    trade_log = []
    balance_history = []
    
    # 產生回測日期 (每個交易日)
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq='B') # Business days
    
    # 下次可以執行 AI 預測的時間 (冷卻時間)
    next_ai_check = dates[0]
    
    for current_date in dates:
        date_str = current_date.strftime("%Y-%m-%d")
        
        # 取得當日股價 (模擬市場)
        daily_prices = {}
        for t in TICKERS:
            if t in full_data and current_date in full_data[t].index:
                daily_prices[t] = full_data[t].loc[current_date]['Close']
        
        # 如果今天沒開盤 (例如假日)，跳過
        if not daily_prices: continue
        
        # --- A. 持倉管理 (賣出檢查) ---
        if portfolio["holdings"]:
            h = portfolio["holdings"]
            ticker = h["Ticker"]
            
            if ticker in daily_prices:
                curr_price = daily_prices[ticker]
                entry_price = h["Entry"]
                pnl_pct = (curr_price - entry_price) / entry_price
                held_days = (current_date - h["BuyDate"]).days
                
                sell_reason = None
                if pnl_pct >= TAKE_PROFIT_PCT: sell_reason = f"💰 止盈 (+{pnl_pct*100:.1f}%)"
                elif pnl_pct <= -STOP_LOSS_PCT: sell_reason = f"🛑 止損 ({pnl_pct*100:.1f}%)"
                elif held_days >= TIME_STOP_DAYS: sell_reason = f"⏰ 時間到 ({held_days}天)"
                
                if sell_reason:
                    amount = h["Shares"] * curr_price
                    portfolio["cash"] = amount
                    portfolio["holdings"] = None
                    trade_log.append({
                        "Date": date_str, "Action": "SELL", "Ticker": ticker,
                        "Price": curr_price, "Reason": sell_reason, "Balance": amount
                    })
                    print(f"[{date_str}] 賣出 {ticker}: {sell_reason} | 餘額: {amount:.0f}")
                    # 賣出後，馬上可以尋找下一個機會
                    next_ai_check = current_date 

        # --- B. 空手時，執行 AI 買入檢查 ---
        # 只有在 "空手" 且 "到達檢查日" 時才跑 AI (節省運算資源)
        if portfolio["holdings"] is None and current_date >= next_ai_check:
            print(f"[{date_str}] 🤖 AI 正在掃描市場尋找機會...")
            
            best_ticker = None
            best_roi = -999
            
            for t in TICKERS:
                # 這裡就是 "回到過去" 訓練模型
                roi = predict_future_roi(t, date_str)
                if roi > best_roi:
                    best_roi = roi
                    best_ticker = t
            
            print(f"   👉 最佳標的: {best_ticker} (預測漲幅: {best_roi:.1f}%)")
            
            if best_roi > MIN_ROI_THRESHOLD and best_ticker in daily_prices:
                # 買入!
                price = daily_prices[best_ticker]
                shares = portfolio["cash"] / price
                portfolio["holdings"] = {
                    "Ticker": best_ticker, "Shares": shares, 
                    "Entry": price, "BuyDate": current_date
                }
                portfolio["cash"] = 0 # All-in
                
                trade_log.append({
                    "Date": date_str, "Action": "BUY", "Ticker": best_ticker,
                    "Price": price, "Reason": f"AI預測漲幅 {best_roi:.1f}%", 
                    "Balance": 0
                })
                print(f"[{date_str}] 🚀 買入 {best_ticker} @ {price:.2f}")
            else:
                # 沒好貨，休息 5 天再看
                next_ai_check = current_date + datetime.timedelta(days=5)
                print(f"   💤 沒達到信心門檻 (> {MIN_ROI_THRESHOLD}%)，觀望 5 天。")

        # --- C. 紀錄資產 ---
        equity = portfolio["cash"]
        if portfolio["holdings"]:
            h = portfolio["holdings"]
            if h["Ticker"] in daily_prices:
                equity = h["Shares"] * daily_prices[h["Ticker"]]
        
        balance_history.append({"Date": date_str, "Equity": equity})

    # 結算
    final_equity = balance_history[-1]['Equity']
    roi = (final_equity - INITIAL_CASH) / INITIAL_CASH * 100
    
    print("="*30)
    print(f"🏁 回測結束！")
    print(f"最終資產: ${final_equity:.2f}")
    print(f"總報酬率: {roi:.2f}%")
    print("="*30)

    # 存檔供網頁顯示
    pd.DataFrame(trade_log).to_csv(os.path.join(DATA_DIR, "ai_backtest_log.csv"), index=False)
    pd.DataFrame(balance_history).to_csv(os.path.join(DATA_DIR, "ai_backtest_balance.csv"), index=False)

if __name__ == "__main__":
    run_backtest()