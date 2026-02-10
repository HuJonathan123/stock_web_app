import pandas as pd
import json
import os
import yfinance as yf
from strategy import vulture_strategy_check

# ===========================
# 1. 設定
# ===========================
DATA_DIR = "data"
LOG_FILE = os.path.join(DATA_DIR, "trade_log.csv")
BALANCE_FILE = os.path.join(DATA_DIR, "balance_history.csv")
PORTFOLIO_FILE = os.path.join(DATA_DIR, "portfolio.json")

# 股票池
TICKERS = ['MSFT', 'GOOGL', 'AMZN', 'COST', 'PEP', 'KO', 'JPM', 'UNH', 'TSLA', 'NVDA', 'AMD']
INITIAL_CASH = 1000
COMMISSION = 2

# 初始化
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

portfolio = {"cash": INITIAL_CASH, "holdings": None, "last_update": ""}
trade_logs = []
balance_history = []
latest_prices = {}

# ===========================
# 2. 下載數據
# ===========================
DOWNLOAD_START = "2024-06-01" 
BACKTEST_START = "2025-01-01"  
TODAY = "2026-02-11"

print(f"正在下載數據 (Start: {BACKTEST_START})...")
data_cache = {}

for t in TICKERS:
    try:
        df = yf.download(t, start=DOWNLOAD_START, end=TODAY, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        data_cache[t] = df
    except: pass

dates = pd.date_range(start=BACKTEST_START, end="2026-02-10")

print("開始活躍回測 (加入時間止損)...")

# ===========================
# 3. 回測主迴圈
# ===========================
for date in dates:
    date_str = date.strftime("%Y-%m-%d")
    
    # 更新最新價格 (前值填充)
    for t in TICKERS:
        if t in data_cache and date in data_cache[t].index:
            latest_prices[t] = data_cache[t].loc[date]['Close']
            
    # --- 賣出檢查 ---
    current_holdings = portfolio['holdings']
    cash = portfolio['cash']
    action_taken = False
    
    if current_holdings:
        t = current_holdings['Ticker']
        # 即使當天沒開盤，如果有 latest_price 也可以估算，但交易必須在開盤日
        if t in data_cache and date in data_cache[t].index:
            price = data_cache[t].loc[date]['Close']
            entry_price = current_holdings['Entry']
            
            # 計算持有天數 (🔥 新增邏輯)
            buy_date = pd.to_datetime(current_holdings['BuyDate'])
            curr_date = pd.to_datetime(date_str)
            days_held = (curr_date - buy_date).days
            
            pnl = (price - entry_price) / entry_price
            
            sell_reason = None
            
            # 1. 獲利達標 (降低標準到 10%，比較容易觸發)
            if pnl > 0.10: sell_reason = f"💰 獲利達標 (+{pnl*100:.1f}%)"
            # 2. 止損
            elif pnl < -0.10: sell_reason = f"💀 止損 (-10%)"
            # 3. 🔥 時間止損：持有超過 14 天且沒虧太多，就賣掉換股
            elif days_held > 14 and pnl > -0.05:
                sell_reason = f"💤 持有過久 ({days_held}天)"
            
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

    # --- 買入檢查 ---
    if not portfolio['holdings'] and not action_taken:
        candidates = []
        for t in TICKERS:
            if t in data_cache and date in data_cache[t].index:
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
                    
                    # 計算布林
                    ma20 = subset['Close'].rolling(20).mean()
                    std = subset['Close'].rolling(20).std()
                    lower = ma20 - (2*std)
                    curr_lower = lower.iloc[-1]
                    
                    # 買入條件 (RSI < 35)
                    if curr_rsi < 35 and close < curr_lower:
                        candidates.append((t, close, curr_rsi))
        
        if candidates:
            candidates.sort(key=lambda x: x[2])
            best_t, best_p, best_r = candidates[0]
            
            if cash > 20:
                invest_amt = cash - COMMISSION
                shares = invest_amt / best_p
                # 🔥 記錄 BuyDate 以便計算持有天數
                portfolio['holdings'] = {"Ticker": best_t, "Shares": shares, "Entry": best_p, "BuyDate": date_str}
                portfolio['cash'] = 0
                
                trade_logs.append({
                    "Date": date_str, "Action": "BUY", "Ticker": best_t, 
                    "Price": round(best_p, 2), "Reason": f"撿屍體 (RSI: {best_r:.1f})", 
                    "Balance": round(invest_amt, 2)
                })

    # --- 資產結算 ---
    total_equity = portfolio['cash']
    if portfolio['holdings']:
        t = portfolio['holdings']['Ticker']
        shares = portfolio['holdings']['Shares']
        if t in latest_prices:
            total_equity += shares * latest_prices[t]
        else:
            total_equity += shares * portfolio['holdings']['Entry']
            
    balance_history.append({"Date": date_str, "Equity": round(total_equity, 2)})
    portfolio['last_update'] = date_str

# 存檔
pd.DataFrame(trade_logs).to_csv(LOG_FILE, index=False)
pd.DataFrame(balance_history).to_csv(BALANCE_FILE, index=False)
with open(PORTFOLIO_FILE, 'w') as f:
    json.dump(portfolio, f)

print(f"回測完成！最終資產: ${balance_history[-1]['Equity']}")