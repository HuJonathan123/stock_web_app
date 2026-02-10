import pandas as pd
import json
import os
import yfinance as yf

# ===========================
# 1. 全局設定
# ===========================
DATA_DIR = "data"
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

TICKERS = ['MSFT', 'GOOGL', 'AMZN', 'COST', 'PEP', 'KO', 'JPM', 'UNH', 'TSLA', 'NVDA', 'AMD', 'META', 'NFLX']
DOWNLOAD_START = "2021-06-01"
TODAY = "2026-02-11"

print(f"📥 正在下載長歷史數據 ({DOWNLOAD_START} ~ {TODAY})...")
data_cache = {}
for t in TICKERS:
    try:
        df = yf.download(t, start=DOWNLOAD_START, end=TODAY, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        data_cache[t] = df
    except: pass

# 定義測試區間 (平行宇宙)
TEST_PERIODS = {
    "2022_bear": ("2022-01-01", "2022-12-31"),      # 熊市
    "2023_recovery": ("2023-01-01", "2023-12-31"),  # 復甦
    "2024_bull": ("2024-01-01", "2024-12-31"),      # 牛市
    "2025_now": ("2025-01-01", "2026-02-10")        # 現況
}

# ===========================
# 2. 回測核心
# ===========================
def run_simulation(strategy_type, start_date, end_date, file_prefix):
    print(f"🚀 執行：{file_prefix} ({start_date} ~ {end_date})")
    
    LOG_FILE = os.path.join(DATA_DIR, f"{file_prefix}_log.csv")
    BALANCE_FILE = os.path.join(DATA_DIR, f"{file_prefix}_balance.csv")
    portfolio = {"cash": 1000, "holdings": []}
    trade_logs = []
    balance_history = []
    latest_prices = {}
    commission = 2
    
    dates = pd.date_range(start=start_date, end=end_date)

    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        
        # 更新價格
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
                
                # 更新最高價
                highest = holding.get('Highest', entry)
                if price > highest: holding['Highest'] = price
                
                sell_reason = None
                
                # === 策略分支 ===
                
                # 1. 經典禿鷹 (Classic) - 20% 獲利 / 15% 止損
                if strategy_type == "classic":
                    if pnl > 0.20: sell_reason = f"💰 獲利達標 (+{pnl*100:.1f}%)"
                    elif pnl < -0.15: sell_reason = f"💀 止損 (-15%)"
                    elif days > 15 and pnl > -0.05: sell_reason = f"💤 資金卡死 ({days}天)"
                
                # 2. 超級禿鷹 (Super) - 動態止盈 / 10% 止損
                elif strategy_type == "super":
                    drop_from_high = (highest - price) / highest
                    if pnl > 0.05 and drop_from_high > 0.05:
                        sell_reason = f"📉 高點回落鎖利 (最高+{((highest-entry)/entry)*100:.1f}%)"
                    elif pnl < -0.10:
                        sell_reason = f"🛡️ 嚴格止損 (-10%)"
                    elif days > 15 and pnl > -0.05:
                        sell_reason = f"💤 資金卡死 ({days}天)"

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
        if len(portfolio['holdings']) < 1:
            candidates = []
            for t in TICKERS:
                if any(h['Ticker'] == t for h in portfolio['holdings']): continue
                
                if t in data_cache and date in data_cache[t].index:
                    idx = data_cache[t].index.get_loc(date)
                    if idx > 20:
                        subset = data_cache[t].iloc[:idx+1]
                        close = subset['Close'].iloc[-1]
                        
                        delta = subset['Close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                        rsi = 100 - (100 / (1 + gain/loss))
                        
                        ma20 = subset['Close'].rolling(20).mean()
                        std = subset['Close'].rolling(20).std()
                        lower = ma20 - (2*std)
                        
                        if rsi.iloc[-1] < 35 and close < lower.iloc[-1]:
                            candidates.append((t, close, rsi.iloc[-1]))
            
            if candidates:
                candidates.sort(key=lambda x: x[2])
                best_t, best_p, best_r = candidates[0]
                
                if portfolio['cash'] > 100:
                    shares = (portfolio['cash'] - commission) / best_p
                    portfolio['holdings'].append({
                        "Ticker": best_t, "Shares": shares, "Entry": best_p,
                        "BuyDate": date_str, "Highest": best_p
                    })
                    portfolio['cash'] = 0
                    trade_logs.append({
                        "Date": date_str, "Action": "BUY", "Ticker": best_t,
                        "Price": round(best_p, 2), "Reason": f"RSI: {best_r:.1f}",
                        "Balance": round(0, 2)
                    })

        # --- 資產結算 ---
        equity = portfolio['cash']
        for h in portfolio['holdings']:
            t = h['Ticker']
            if t in latest_prices: equity += h['Shares'] * latest_prices[t]
            else: equity += h['Shares'] * h['Entry']
        
        balance_history.append({"Date": date_str, "Equity": round(equity, 2)})

    pd.DataFrame(trade_logs).to_csv(LOG_FILE, index=False)
    pd.DataFrame(balance_history).to_csv(BALANCE_FILE, index=False)

# ===========================
# 3. 執行所有組合
# ===========================
for name, (start, end) in TEST_PERIODS.items():
    # 跑 Tab 1 的策略 (Classic)
    run_simulation("classic", start, end, f"vulture_{name}")
    
    # 跑 Tab 2 的策略 (Super)
    run_simulation("super", start, end, f"super_vulture_{name}")

print("✅ 所有回測完成！")
