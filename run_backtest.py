import pandas as pd
import json
import os
import yfinance as yf

# ===========================
# 1. 設定與數據下載
# ===========================
DATA_DIR = "data"
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

TICKERS = ['MSFT', 'GOOGL', 'AMZN', 'COST', 'PEP', 'KO', 'JPM', 'UNH', 'TSLA', 'NVDA', 'AMD', 'META', 'NFLX']
DOWNLOAD_START = "2024-06-01"
BACKTEST_START = "2025-01-01"
TODAY = "2026-02-11"

print("正在下載數據...")
data_cache = {}
for t in TICKERS:
    try:
        df = yf.download(t, start=DOWNLOAD_START, end=TODAY, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        data_cache[t] = df
    except: pass

dates = pd.date_range(start=BACKTEST_START, end="2026-02-10")

# ===========================
# 核心回測函數
# ===========================
def run_simulation(strategy_name, max_positions=1):
    print(f"--- 執行策略：{strategy_name} ---")
    
    LOG_FILE = os.path.join(DATA_DIR, f"{strategy_name}_log.csv")
    BALANCE_FILE = os.path.join(DATA_DIR, f"{strategy_name}_balance.csv")
    PORTFOLIO_FILE = os.path.join(DATA_DIR, f"{strategy_name}_portfolio.json")
    
    portfolio = {"cash": 1000, "holdings": [], "last_update": ""}
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
                
                # 更新最高價 (用於 Super Vulture 的移動止盈)
                highest = holding.get('Highest', entry)
                if price > highest: holding['Highest'] = price
                
                sell_reason = None
                
                # === 策略大對決 ===
                
                # 策略 1: 原始禿鷹 (Vulture) - 你現在 40% 的那個版本
                if strategy_name == "vulture":
                    if pnl > 0.20: sell_reason = f"💰 獲利達標 (+{pnl*100:.1f}%)"
                    elif pnl < -0.15: sell_reason = f"💀 止損 (-15%)"
                    # 時間止損 (這是獲利的關鍵)
                    elif days > 15 and pnl > -0.05: sell_reason = f"💤 資金卡死換股 ({days}天)"
                
                # 策略 2: 超級禿鷹 (Super Vulture) - 嘗試挑戰更高報酬
                elif strategy_name == "super_vulture":
                    # 計算從最高點回落的幅度
                    drop_from_high = (highest - price) / highest
                    
                    # 條件 A: 移動止盈 (讓利潤奔跑)
                    # 只有當獲利超過 5% 後，如果回吐 5% 才賣
                    if pnl > 0.05 and drop_from_high > 0.05:
                        sell_reason = f"📉 高點回落鎖利 (最高+{((highest-entry)/entry)*100:.1f}%)"
                    
                    # 條件 B: 嚴格止損 (比原本更嚴，虧 10% 就砍)
                    elif pnl < -0.10:
                        sell_reason = f"🛡️ 嚴格止損 (-10%)"
                    
                    # 條件 C: 時間止損 (保持資金流動性，這很重要)
                    elif days > 15 and pnl > -0.05:
                        sell_reason = f"💤 資金卡死換股 ({days}天)"

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
                        
                        # RSI & Bollinger Bands
                        delta = subset['Close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                        rsi = 100 - (100 / (1 + gain/loss))
                        curr_rsi = rsi.iloc[-1]
                        
                        ma20 = subset['Close'].rolling(20).mean()
                        std = subset['Close'].rolling(20).std()
                        lower = ma20 - (2*std)
                        
                        # 買入信號 (兩者共用)
                        if curr_rsi < 35 and close < lower.iloc[-1]:
                            candidates.append((t, close, curr_rsi))
            
            if candidates:
                candidates.sort(key=lambda x: x[2]) # 選 RSI 最低的
                
                slots = max_positions - len(portfolio['holdings'])
                per_trade = portfolio['cash'] / slots
                
                for i in range(min(len(candidates), slots)):
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

    pd.DataFrame(trade_logs).to_csv(LOG_FILE, index=False)
    pd.DataFrame(balance_history).to_csv(BALANCE_FILE, index=False)
    with open(PORTFOLIO_FILE, 'w') as f: json.dump(portfolio, f)
    
    print(f"策略 {strategy_name} 完成。最終資產: ${balance_history[-1]['Equity']}")

# ===========================
# 執行回測
# ===========================
# 1. 原始禿鷹 (你現在 40% 的版本)
run_simulation("vulture", max_positions=1)

# 2. 超級禿鷹 (新的挑戰者)
run_simulation("super_vulture", max_positions=1)
