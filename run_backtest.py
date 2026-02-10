import pandas as pd
import json
import os
import yfinance as yf
from strategy import vulture_strategy_check

# 設定
DATA_DIR = "data"
LOG_FILE = os.path.join(DATA_DIR, "trade_log.csv")
BALANCE_FILE = os.path.join(DATA_DIR, "balance_history.csv") # 新增：每日資產紀錄
PORTFOLIO_FILE = os.path.join(DATA_DIR, "portfolio.json")

# 擴大股票池 (加入波動大的股票以增加交易頻率)
TICKERS = ['MSFT', 'GOOGL', 'AMZN', 'COST', 'PEP', 'KO', 'JPM', 'UNH', 'TSLA', 'NVDA', 'AMD']
INITIAL_CASH = 1000
COMMISSION = 2

# 初始化資料夾
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

# 初始化狀態
portfolio = {"cash": INITIAL_CASH, "holdings": None, "last_update": ""}
trade_logs = []
balance_history = []

# 下載數據 (從 2025-10-01 開始，確保 2026-01-01 時有足夠數據算指標)
print("正在下載歷史數據...")
data_cache = {}
for t in TICKERS:
    try:
        df = yf.download(t, start="2025-10-01", end="2026-02-11", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        data_cache[t] = df
    except:
        pass

# 生成回測日期範圍 (2026-01-01 到今天)
dates = pd.date_range(start="2026-01-01", end="2026-02-10")

print("開始每日回測 (Mark-to-Market)...")

for date in dates:
    date_str = date.strftime("%Y-%m-%d")
    
    # --- 1. 策略執行 (買賣判斷) ---
    current_holdings = portfolio['holdings']
    cash = portfolio['cash']
    action_taken = False
    
    # 如果持有股票，檢查賣出
    if current_holdings:
        t = current_holdings['Ticker']
        if date in data_cache[t].index:
            price = data_cache[t].loc[date]['Close']
            entry_price = current_holdings['Entry']
            
            # 計算當前報酬率
            pnl = (price - entry_price) / entry_price
            
            # 賣出邏輯 (簡化版：賺20%或賠15%或是RSI過熱)
            # 這裡簡單模擬：若持有超過 20 天且沒大跌也賣出換現金 (增加流動性)
            sell_reason = None
            if pnl > 0.20: sell_reason = f"💰 獲利達標 (+{pnl*100:.1f}%)"
            elif pnl < -0.15: sell_reason = f"💀 止損 (-15%)"
            
            if sell_reason:
                gross = current_holdings['Shares'] * price
                cash = gross - COMMISSION
                portfolio['holdings'] = None
                portfolio['cash'] = cash
                
                trade_logs.append({
                    "Date": date_str, "Action": "SELL", "Ticker": t, 
                    "Price": round(price, 2), "Reason": sell_reason, "Balance": round(cash, 2)
                })
                action_taken = True

    # 如果空手，檢查買入
    if not portfolio['holdings'] and not action_taken:
        candidates = []
        for t in TICKERS:
            if t in data_cache and date in data_cache[t].index:
                # 取得當日數據
                idx = data_cache[t].index.get_loc(date)
                if idx > 20:
                    subset = data_cache[t].iloc[:idx+1]
                    close = subset['Close'].iloc[-1]
                    
                    # 計算 RSI
                    delta = subset['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    rsi = 100 - (100 / (1 + gain/loss))
                    curr_rsi = rsi.iloc[-1]
                    
                    # 計算布林下軌
                    ma20 = subset['Close'].rolling(20).mean()
                    std = subset['Close'].rolling(20).std()
                    lower = ma20 - (2*std)
                    curr_lower = lower.iloc[-1]
                    
                    # 買入條件：放寬一點點以便測試 (RSI < 35)
                    if curr_rsi < 35 and close < curr_lower:
                        candidates.append((t, close, curr_rsi))
        
        # 執行買入
        if candidates:
            candidates.sort(key=lambda x: x[2]) # 選 RSI 最低的
            best_t, best_p, best_r = candidates[0]
            
            if cash > 20:
                invest_amt = cash - COMMISSION # 先扣手續費
                shares = invest_amt / best_p
                portfolio['holdings'] = {"Ticker": best_t, "Shares": shares, "Entry": best_p}
                portfolio['cash'] = 0 # 全倉買入
                
                trade_logs.append({
                    "Date": date_str, "Action": "BUY", "Ticker": best_t, 
                    "Price": round(best_p, 2), "Reason": f"撿屍體 (RSI: {best_r:.1f})", 
                    "Balance": round(invest_amt, 2) # 這裡紀錄的是扣費後的淨值
                })

    # --- 2. 每日資產結算 (Mark-to-Market) ---
    # 這是計算「浮動盈虧」的關鍵
    total_equity = portfolio['cash']
    
    if portfolio['holdings']:
        t = portfolio['holdings']['Ticker']
        shares = portfolio['holdings']['Shares']
        if date in data_cache[t].index:
            current_price = data_cache[t].loc[date]['Close']
            market_value = shares * current_price
            total_equity += market_value
        else:
            # 如果假日沒數據，沿用上次的價值 (或是 entry price)
            total_equity += shares * portfolio['holdings']['Entry']
            
    balance_history.append({"Date": date_str, "Equity": round(total_equity, 2)})
    portfolio['last_update'] = date_str

# 存檔
pd.DataFrame(trade_logs).to_csv(LOG_FILE, index=False)
pd.DataFrame(balance_history).to_csv(BALANCE_FILE, index=False) # 存這個給圖表用

with open(PORTFOLIO_FILE, 'w') as f:
    json.dump(portfolio, f)

print(f"回測完成！最終資產: ${balance_history[-1]['Equity']}")