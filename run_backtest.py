import pandas as pd
import json
import os
import yfinance as yf
from strategy import vulture_strategy_check

# 設定
DATA_DIR = "data"
LOG_FILE = os.path.join(DATA_DIR, "trade_log.csv")
PORTFOLIO_FILE = os.path.join(DATA_DIR, "portfolio.json")
TICKERS = ['MSFT', 'GOOGL', 'AMZN', 'COST', 'PEP', 'KO', 'JPM', 'UNH']
INITIAL_CASH = 1000
COMMISSION = 2

# 初始化
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
pd.DataFrame(columns=["Date", "Action", "Ticker", "Price", "Reason", "Balance"]).to_csv(LOG_FILE, index=False)
portfolio = {"cash": INITIAL_CASH, "holdings": None, "last_update": ""}

# 下載所有數據 (從 2025-12-01 開始，為了計算指標)
print("正在下載歷史數據...")
data_cache = {}
for t in TICKERS:
    df = yf.download(t, start="2025-12-01", end="2026-02-11", progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    data_cache[t] = df

# 生成回測日期範圍 (2026-01-01 到今天)
dates = pd.date_range(start="2026-01-01", end="2026-02-10")

print("開始回測...")
for date in dates:
    date_str = date.strftime("%Y-%m-%d")
    
    # 策略邏輯
    current_holdings = portfolio['holdings']
    cash = portfolio['cash']
    action_taken = False

    # 1. 檢查賣出
    if current_holdings:
        t = current_holdings['Ticker']
        if date in data_cache[t].index:
            # 為了使用 strategy.py 的邏輯，我們需要切片出當天之前的數據
            # 這裡為了簡化，直接重寫部分邏輯以適應回測，或模擬 strategy 需要的輸入
            # 這裡我們手動計算當日指標
            
            # 取得當日價格
            price = data_cache[t].loc[date]['Close']
            entry_price = current_holdings['Entry']
            pnl = (price - entry_price) / entry_price
            
            # 取得 RSI (需要往前推算)
            idx = data_cache[t].index.get_loc(date)
            if idx > 15:
                # 簡單計算賣出條件
                sell_reason = None
                if pnl > 0.20: sell_reason = f"💰 獲利達標 (+{pnl*100:.1f}%)"
                elif pnl < -0.15: sell_reason = f"💀 止損 (-15%)"
                
                # 如果要嚴謹的 RSI，這裡可以計算，簡化起見先用 PnL 觸發
                
                if sell_reason:
                    gross = current_holdings['Shares'] * price
                    cash = gross - COMMISSION
                    portfolio['holdings'] = None
                    portfolio['cash'] = cash
                    
                    new_row = {"Date": date_str, "Action": "SELL", "Ticker": t, "Price": round(price, 2), "Reason": sell_reason, "Balance": round(cash, 2)}
                    pd.DataFrame([new_row]).to_csv(LOG_FILE, mode='a', header=False, index=False)
                    action_taken = True

    # 2. 檢查買入 (只有空手時)
    if not current_holdings and not action_taken:
        candidates = []
        for t in TICKERS:
            if date in data_cache[t].index:
                # 計算當日指標
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
                    
                    if curr_rsi < 30 and close < curr_lower:
                        candidates.append((t, close, curr_rsi))
        
        # 選 RSI 最低的
        if candidates:
            candidates.sort(key=lambda x: x[2]) # sort by RSI
            best_t, best_p, best_r = candidates[0]
            
            if cash > 20:
                invest_amt = cash - COMMISSION
                shares = invest_amt / best_p
                portfolio['holdings'] = {"Ticker": best_t, "Shares": shares, "Entry": best_p}
                portfolio['cash'] = 0
                
                new_row = {"Date": date_str, "Action": "BUY", "Ticker": best_t, "Price": round(best_p, 2), "Reason": f"撿屍體 (RSI: {best_r:.1f})", "Balance": round(cash, 2)}
                pd.DataFrame([new_row]).to_csv(LOG_FILE, mode='a', header=False, index=False)

    portfolio['last_update'] = date_str

# 存檔
with open(PORTFOLIO_FILE, 'w') as f:
    json.dump(portfolio, f)

print(f"回測完成！最終資產: {portfolio['cash'] if not portfolio['holdings'] else '持倉中'}")