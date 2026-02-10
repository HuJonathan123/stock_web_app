import pandas as pd
import json
import os
import yfinance as yf

# ===========================
# 1. 共用設定與數據下載
# ===========================
DATA_DIR = "data"
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

TICKERS = ['MSFT', 'GOOGL', 'AMZN', 'COST', 'PEP', 'KO', 'JPM', 'UNH', 'TSLA', 'NVDA', 'AMD', 'META', 'NFLX']
DOWNLOAD_START = "2024-06-01"
BACKTEST_START = "2025-01-01"
TODAY = "2026-02-11"

print("正在下載數據 (共用)...")
data_cache = {}
for t in TICKERS:
    try:
        df = yf.download(t, start=DOWNLOAD_START, end=TODAY, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        data_cache[t] = df
    except: pass

dates = pd.date_range(start=BACKTEST_START, end="2026-02-10")

# ===========================
# 函數：執行回測引擎
# ===========================
def run_simulation(strategy_name, max_positions, initial_cash=1000):
    print(f"--- 正在執行策略：{strategy_name} ---")
    
    # 檔案路徑
    LOG_FILE = os.path.join(DATA_DIR, f"{strategy_name}_log.csv")
    BALANCE_FILE = os.path.join(DATA_DIR, f"{strategy_name}_balance.csv")
    PORTFOLIO_FILE = os.path.join(DATA_DIR, f"{strategy_name}_portfolio.json")
    
    # 初始化
    portfolio = {"cash": initial_cash, "holdings": [], "last_update": ""}
    trade_logs = []
    balance_history = []
    latest_prices = {}
    commission = 2

    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        
        # 更新最新價格
        for t in TICKERS:
            if t in data_cache and date in data_cache[t].index:
                latest_prices[t] = data_cache[t].loc[date]['Close']
        
        # --- 賣出檢查 ---
        # 倒序遍歷以便刪除
        for i in range(len(portfolio['holdings']) - 1, -1, -1):
            holding = portfolio['holdings'][i]
            t = holding['Ticker']
            
            if t in data_cache and date in data_cache[t].index:
                price = data_cache[t].loc[date]['Close']
                entry = holding['Entry']
                shares = holding['Shares']
                buy_date = pd.to_datetime(holding['BuyDate'])
                curr_date = pd.to_datetime(date_str)
                days = (curr_date - buy_date).days
                pnl = (price - entry) / entry
                
                # 更新最高價 (用於動態止盈)
                highest = holding.get('Highest', entry)
                if price > highest: holding['Highest'] = price
                
                sell_reason = None
                
                # === 策略分支邏輯 ===
                if strategy_name == "vulture": # 禿鷹 (All-in, 死板)
                    if pnl > 0.20: sell_reason = f"💰 獲利達標 (+{pnl*100:.1f}%)"
                    elif pnl < -0.15: sell_reason = f"💀 止損 (-15%)"
                    elif days > 15 and pnl > -0.05: sell_reason = f"💤 資金卡死 ({days}天)"
                
                elif strategy_name == "octopus": # 章魚 (分散, 靈活)
                    high_pnl = (highest - entry) / entry
                    drop = (highest - price) / highest
                    
                    if high_pnl > 0.10 and drop > 0.05: sell_reason = f"📉 回調鎖利 (最高+{high_pnl*100:.1f}%)"
                    elif pnl > 0.25: sell_reason = f"🚀 暴賺離場 (+{pnl*100:.1f}%)"
                    elif pnl < -0.08: sell_reason = f"💀 嚴格止損 (-8%)"
                    elif pnl < 0 and days > 7: sell_reason = f"🗑️ 弱勢清理 ({days}天)"
                    elif days > 20: sell_reason = f"💤 資金輪動 ({days}天)"

                # 執行賣出
                if sell_reason:
                    amount = shares * price
                    portfolio['cash'] += (amount - commission)
                    trade_logs.append({
                        "Date": date_str, "Action": "SELL", "Ticker": t,
                        "Price": round(price, 2), "Reason": sell_reason,
                        "Balance": round(portfolio['cash'], 2)
                    })
                    portfolio['holdings'].pop(i)

        # --- 買入檢查 ---
        if len(portfolio['holdings']) < max_positions:
            candidates = []
            for t in TICKERS:
                if any(h['Ticker'] == t for h in portfolio['holdings']): continue
                
                if t in data_cache and date in data_cache[t].index:
                    idx = data_cache[t].index.get_loc(date)
                    if idx > 20:
                        subset = data_cache[t].iloc[:idx+1]
                        close = subset['Close'].iloc[-1]
                        
                        # RSI 計算
                        delta = subset['Close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                        rsi = 100 - (100 / (1 + gain/loss))
                        curr_rsi = rsi.iloc[-1]
                        
                        # 布林通道
                        ma20 = subset['Close'].rolling(20).mean()
                        std = subset['Close'].rolling(20).std()
                        lower = ma20 - (2*std)
                        curr_lower = lower.iloc[-1]
                        
                        if curr_rsi < 35 and close < curr_lower:
                            candidates.append((t, close, curr_rsi))
            
            if candidates:
                candidates.sort(key=lambda x: x[2])
                
                # 計算本次下注金額
                slots = max_positions - len(portfolio['holdings'])
                per_trade = portfolio['cash'] / slots
                
                # 最多買幾隻
                buy_count = min(len(candidates), slots)
                
                for i in range(buy_count):
                    t, p, r = candidates[i]
                    if per_trade > 100:
                        shares = (per_trade - commission) / p
                        portfolio['holdings'].append({
                            "Ticker": t, "Shares": shares, "Entry": p,
                            "BuyDate": date_str, "Highest": p
                        })
                        portfolio['cash'] -= per_trade
                        trade_logs.append({
                            "Date": date_str, "Action": "BUY", "Ticker": t,
                            "Price": round(p, 2), "Reason": f"RSI: {r:.1f}",
                            "Balance": round(portfolio['cash'], 2)
                        })

        # --- 資產結算 ---
        equity = portfolio['cash']
        for h in portfolio['holdings']:
            t = h['Ticker']
            if t in latest_prices: equity += h['Shares'] * latest_prices[t]
            else: equity += h['Shares'] * h['Entry']
        
        balance_history.append({"Date": date_str, "Equity": round(equity, 2)})
        portfolio['last_update'] = date_str

    # 存檔
    pd.DataFrame(trade_logs).to_csv(LOG_FILE, index=False)
    pd.DataFrame(balance_history).to_csv(BALANCE_FILE, index=False)
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(portfolio, f)
    
    final_eq = balance_history[-1]['Equity']
    print(f"策略 {strategy_name} 完成。最終資產: ${final_eq}")

# ===========================
# 主程式：執行兩個策略
# ===========================
# 1. 禿鷹策略 (原本的): 1 份資金 (All-in)
run_simulation("vulture", max_positions=1)

# 2. 章魚策略 (新的): 3 份資金 (分散)
run_simulation("octopus", max_positions=3)
