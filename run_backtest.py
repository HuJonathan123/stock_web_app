import pandas as pd
import json
import os
import yfinance as yf
from strategy import vulture_strategy_check

# ===========================
# 1. 設定與初始化
# ===========================
DATA_DIR = "data"
LOG_FILE = os.path.join(DATA_DIR, "trade_log.csv")
BALANCE_FILE = os.path.join(DATA_DIR, "balance_history.csv")
PORTFOLIO_FILE = os.path.join(DATA_DIR, "portfolio.json")

# 股票池
TICKERS = ['MSFT', 'GOOGL', 'AMZN', 'COST', 'PEP', 'KO', 'JPM', 'UNH', 'TSLA', 'NVDA', 'AMD']
INITIAL_CASH = 1000
COMMISSION = 2

# 確保資料夾存在
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

# 初始化變數
portfolio = {"cash": INITIAL_CASH, "holdings": None, "last_update": ""}
trade_logs = []
balance_history = []
latest_prices = {} # 🔥 新增：用來記憶每隻股票的最新價格

# ===========================
# 2. 下載數據
# ===========================
# 我們需要比回測開始日更早的數據來計算 RSI (Buffer 期)
DOWNLOAD_START = "2025-06-01" 
BACKTEST_START = "2025-10-01"
TODAY = "2026-02-11"

print(f"正在下載數據 (Buffer: {DOWNLOAD_START} -> Start: {BACKTEST_START})...")
data_cache = {}

for t in TICKERS:
    try:
        # 下載數據
        df = yf.download(t, start=DOWNLOAD_START, end=TODAY, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        data_cache[t] = df
    except Exception as e:
        print(f"Error downloading {t}: {e}")

# 生成回測日期範圍 (日曆日，包含週末)
dates = pd.date_range(start=BACKTEST_START, end="2026-02-10")

print("開始每日回測 (含週末市值維持)...")

# ===========================
# 3. 開始回測循環
# ===========================
for date in dates:
    date_str = date.strftime("%Y-%m-%d")
    
    # 🔥 步驟 A: 更新當日所有股票的最新價格 (如果有開盤)
    for t in TICKERS:
        if t in data_cache and date in data_cache[t].index:
            latest_prices[t] = data_cache[t].loc[date]['Close']
            
    # 策略執行
    current_holdings = portfolio['holdings']
    cash = portfolio['cash']
    action_taken = False
    
    # --- 賣出檢查 ---
    if current_holdings:
        t = current_holdings['Ticker']
        # 只有當天有開盤才能賣
        if t in data_cache and date in data_cache[t].index:
            price = data_cache[t].loc[date]['Close']
            entry_price = current_holdings['Entry']
            pnl = (price - entry_price) / entry_price
            
            sell_reason = None
            if pnl > 0.20: sell_reason = f"💰 獲利達標 (+{pnl*100:.1f}%)"
            elif pnl < -0.15: sell_reason = f"💀 止損 (-15%)"
            
            # RSI 過高賣出 (這需要重新計算當日 RSI，這裡簡化邏輯)
            
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
            # 只有當天有開盤才能買
            if t in data_cache and date in data_cache[t].index:
                # 確保有足夠歷史數據算指標
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
                    
                    # 買入條件
                    if curr_rsi < 35 and close < curr_lower:
                        candidates.append((t, close, curr_rsi))
        
        if candidates:
            candidates.sort(key=lambda x: x[2]) # 選 RSI 最低的
            best_t, best_p, best_r = candidates[0]
            
            if cash > 20:
                invest_amt = cash - COMMISSION
                shares = invest_amt / best_p
                portfolio['holdings'] = {"Ticker": best_t, "Shares": shares, "Entry": best_p}
                portfolio['cash'] = 0
                
                trade_logs.append({
                    "Date": date_str, "Action": "BUY", "Ticker": best_t, 
                    "Price": round(best_p, 2), "Reason": f"撿屍體 (RSI: {best_r:.1f})", 
                    "Balance": round(invest_amt, 2)
                })

    # --- 每日結算 (Mark-to-Market) ---
    total_equity = portfolio['cash']
    
    if portfolio['holdings']:
        t = portfolio['holdings']['Ticker']
        shares = portfolio['holdings']['Shares']
        
        # 🔥 修正點：優先使用當日價格，如果是週末，使用 latest_prices (最近一次收盤價)
        if t in latest_prices:
            current_price = latest_prices[t]
            market_value = shares * current_price
            total_equity += market_value
        else:
            # 萬一連最近價格都沒有 (極少見)，用買入價
            total_equity += shares * portfolio['holdings']['Entry']
            
    balance_history.append({"Date": date_str, "Equity": round(total_equity, 2)})
    portfolio['last_update'] = date_str

# ===========================
# 4. 存檔
# ===========================
pd.DataFrame(trade_logs).to_csv(LOG_FILE, index=False)
pd.DataFrame(balance_history).to_csv(BALANCE_FILE, index=False)

with open(PORTFOLIO_FILE, 'w') as f:
    json.dump(portfolio, f)

print(f"回測完成！最終資產: ${balance_history[-1]['Equity']}")