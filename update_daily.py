import pandas as pd
import json
import os
import datetime
from strategy import vulture_strategy_check

# 設定
DATA_DIR = "data"
LOG_FILE = os.path.join(DATA_DIR, "trade_log.csv")
PORTFOLIO_FILE = os.path.join(DATA_DIR, "portfolio.json")
TICKERS = ['MSFT', 'GOOGL', 'AMZN', 'COST', 'PEP', 'KO', 'JPM', 'UNH']
INITIAL_CASH = 1000
COMMISSION = 2

# 初始化數據文件 (如果是第一次執行)
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

if not os.path.exists(PORTFOLIO_FILE):
    init_state = {"cash": INITIAL_CASH, "holdings": None, "last_update": ""}
    with open(PORTFOLIO_FILE, 'w') as f: json.dump(init_state, f)

if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["Date", "Action", "Ticker", "Price", "Reason", "Balance"]).to_csv(LOG_FILE, index=False)

# 讀取狀態
with open(PORTFOLIO_FILE, 'r') as f:
    portfolio = json.load(f)

today = datetime.date.today().strftime("%Y-%m-%d")
print(f"Running update for {today}...")

# 執行策略
action_taken = False
current_holdings = portfolio['holdings']
cash = portfolio['cash']

# 如果持有股票，檢查賣出
if current_holdings:
    ticker = current_holdings['Ticker']
    action, reason = vulture_strategy_check(ticker, cash, current_holdings)
    
    if action == "SELL":
        # 模擬取得今日收盤價 (實際應用可以用 yfinance 最新價)
        import yfinance as yf
        price = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
        
        # 結算
        gross = current_holdings['Shares'] * price
        cash = gross - COMMISSION
        portfolio['holdings'] = None # 變回空手
        portfolio['cash'] = cash
        
        # 寫入 Log
        new_row = {"Date": today, "Action": "SELL", "Ticker": ticker, "Price": round(price, 2), "Reason": reason, "Balance": round(cash, 2)}
        pd.DataFrame([new_row]).to_csv(LOG_FILE, mode='a', header=False, index=False)
        action_taken = True

# 如果空手，檢查買入 (掃描所有候選股)
if not current_holdings:
    best_opportunity = None
    min_rsi = 100
    
    for t in TICKERS:
        action, reason = vulture_strategy_check(t, cash, None)
        if action == "BUY":
            # 簡單解析 RSI 數值來比較誰更低
            try:
                rsi_val = float(reason.split("RSI: ")[1].replace(")", ""))
                if rsi_val < min_rsi:
                    min_rsi = rsi_val
                    # 取得即時價格
                    import yfinance as yf
                    price = yf.Ticker(t).history(period="1d")['Close'].iloc[-1]
                    best_opportunity = (t, price, reason)
            except: pass
            
    if best_opportunity:
        ticker, price, reason = best_opportunity
        invest_amt = cash - COMMISSION
        if invest_amt > 0:
            shares = invest_amount / price
            portfolio['holdings'] = {"Ticker": ticker, "Shares": shares, "Entry": price}
            portfolio['cash'] = 0 # 全倉
            
            # 寫入 Log
            new_row = {"Date": today, "Action": "BUY", "Ticker": ticker, "Price": round(price, 2), "Reason": reason, "Balance": round(cash, 2)}
            pd.DataFrame([new_row]).to_csv(LOG_FILE, mode='a', header=False, index=False)
            action_taken = True

# 更新最後執行時間
portfolio['last_update'] = today

# 存檔
with open(PORTFOLIO_FILE, 'w') as f:
    json.dump(portfolio, f)

print("Update complete.")