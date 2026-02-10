import pandas as pd
import json
import os
import yfinance as yf

# ===========================
# 1. 全局設定與資料夾初始化
# ===========================
DATA_DIR = "data"
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

# 股票池：科技成長 + 防禦型 (確保樣本多樣性)
TICKERS = ['MSFT', 'GOOGL', 'AMZN', 'COST', 'PEP', 'KO', 'JPM', 'UNH', 'TSLA', 'NVDA', 'AMD', 'META', 'NFLX']

# 下載足夠長的歷史數據 (涵蓋 2022 熊市前)
DOWNLOAD_START = "2021-06-01"
TODAY = "2026-02-11"

print(f"📥 正在下載長歷史數據 ({DOWNLOAD_START} ~ {TODAY})...")
data_cache = {}
for t in TICKERS:
    try:
        df = yf.download(t, start=DOWNLOAD_START, end=TODAY, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        data_cache[t] = df
    except Exception as e:
        print(f"❌ 下載失敗 {t}: {e}")

# ===========================
# 2. 通用回測函數
# ===========================
def run_simulation(strategy_type, start_date, end_date, file_prefix):
    """
    strategy_type: 'classic' (Tab 1) 或 'super' (Tab 2)
    file_prefix: 輸出的檔案名稱前綴
    """
    print(f"🚀 正在執行：{file_prefix} ({start_date} ~ {end_date})")
    
    # 檔案路徑
    LOG_FILE = os.path.join(DATA_DIR, f"{file_prefix}_log.csv")
    BALANCE_FILE = os.path.join(DATA_DIR, f"{file_prefix}_balance.csv")
    PORTFOLIO_FILE = os.path.join(DATA_DIR, f"{file_prefix}_portfolio.json")
    
    # 初始化
    portfolio = {"cash": 1000, "holdings": [], "last_update": ""}
    trade_logs = []
    balance_history = []
    latest_prices = {}
    commission = 2
    
    dates = pd.date_range(start=start_date, end=end_date)

    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        
        # 更新當日價格 (用於週末市值計算)
        for t in TICKERS:
            if t in data_cache and date in data_cache[t].index:
                latest_prices[t] = data_cache[t].loc[date]['Close']
        
        # --- A. 賣出檢查 ---
        # 倒序遍歷 (方便移除)
        for i in range(len(portfolio['holdings']) - 1, -1, -1):
            holding = portfolio['holdings'][i]
            t = holding['Ticker']
            
            # 只有開盤日才能交易
            if t in data_cache and date in data_cache[t].index:
                price = data_cache[t].loc[date]['Close']
                entry = holding['Entry']
                shares = holding['Shares']
                buy_date = pd.to_datetime(holding['BuyDate'])
                curr_date = pd.to_datetime(date_str)
                days = (curr_date - buy_date).days
                pnl = (price - entry) / entry
                
                # 更新最高價 (Trailing Stop 用)
                highest = holding.get('Highest', entry)
                if price > highest: holding['Highest'] = price
                
                sell_reason = None
                
                # === 策略分支 ===
                
                # 1. 經典禿鷹 (Classic) - Tab 1 保留原本設定
                if strategy_type == "classic":
                    if pnl > 0.20: sell_reason = f"💰 獲利達標 (+{pnl*100:.1f}%)"
                    elif pnl < -0.15: sell_reason = f"💀 止損 (-15%)"
                    elif days > 15 and pnl > -0.05: sell_reason = f"💤 資金卡死 ({days}天)"
                
                # 2. 超級禿鷹 (Super) - Tab 2 壓力測試用
                elif strategy_type == "super":
                    drop_from_high = (highest - price) / highest
                    
                    # 邏輯: 獲利奔跑 + 動態止盈
                    if pnl > 0.05 and drop_from_high > 0.05:
                        sell_reason = f"📉 高點回落鎖利 (最高+{((highest-entry)/entry)*100:.1f}%)"
                    elif pnl < -0.10: # 比經典版更嚴格的止損
                        sell_reason = f"🛡️ 嚴格止損 (-10%)"
                    elif days > 15 and pnl > -0.05:
                        sell_reason = f"💤 資金卡死 ({days}天)"

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

        # --- B. 買入檢查 (All-in 單押) ---
        if len(portfolio['holdings']) < 1:
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
                        
                        # 布林通道下軌
                        ma20 = subset['Close'].rolling(20).mean()
                        std = subset['Close'].rolling(20).std()
                        lower = ma20 - (2*std)
                        
                        # 買入信號 (RSI < 35 且 跌破下軌)
                        if rsi.iloc[-1] < 35 and close < lower.iloc[-1]:
                            candidates.append((t, close, rsi.iloc[-1]))
            
            if candidates:
                candidates.sort(key=lambda x: x[2]) # 選 RSI 最低的
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

        # --- C. 資產結算 (Mark-to-Market) ---
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
    with open(PORTFOLIO_FILE, 'w') as f: json.dump(portfolio, f)
    
    print(f"✅ 完成。最終資產: ${balance_history[-1]['Equity']:.2f}")

# ===========================
# 3. 執行任務排程
# ===========================

# --- 任務 1: Tab 1 經典禿鷹 (保留現狀) ---
run_simulation("classic", "2025-01-01", "2026-02-10", "vulture")

# --- 任務 2: Tab 2 超級禿鷹 (壓力測試多重宇宙) ---
test_years = {
    "2022_bear": ("2022-01-01", "2022-12-31"),      # 熊市
    "2023_recovery": ("2023-01-01", "2023-12-31"),  # 復甦
    "2024_bull": ("2024-01-01", "2024-12-31"),      # 牛市
    "2025_now": ("2025-01-01", "2026-02-10")        # 現況
}

for name, (start, end) in test_years.items():
    run_simulation("super", start, end, f"super_vulture_{name}")
