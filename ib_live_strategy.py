import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
from ib_insync import *
import datetime
import math
import json
import os
import ta
import time

# ===========================
# ⚙️ 策略參數 (完美移植你的設定)
# ===========================
BUY_PROB_THRESHOLD = 0.55 
TIME_STOP_DAYS = 20       
ATR_STOP_LOSS_MULTIPLIER = 2.5 
STRONG_DROP_TOLERANCE = 0.05 
WEAK_DROP_TOLERANCE = 0.025  
ALLOCATION_PCT = 0.33 
TOP_N_MOMENTUM = 3 
MAX_POSITIONS = 3
MARKET_INDEX = 'QQQ'
TICKERS = [
    'NVDA', 'TSLA', 'AMZN', 'MSFT', 'GOOGL', 'META', 'AAPL', 
    'AMD', 'INTC', 'QCOM', 'AVGO', 'MU',                     
    'JPM', 'V', 'DIS', 'NFLX', 'COST', 'PEP', 'KO', 'JNJ'    
]

# 資料路徑
STATE_FILE = "data/live_portfolio_state.json" # 用來記憶每檔股票的最高價和買入日
MODEL_DIR = "saved_models/latest"

# ---------------------------------------------------------
# 🛠 輔助函數：計算指標與讀取狀態
# ---------------------------------------------------------
def add_technical_indicators(df):
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    df['EMA20'] = ta.trend.ema_indicator(df['Close'], window=20)
    df['EMA60'] = ta.trend.ema_indicator(df['Close'], window=60)
    df.fillna(method='bfill', inplace=True)
    return df

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=4)

# ---------------------------------------------------------
# 🤖 核心主程式
# ---------------------------------------------------------
def run_live_strategy():
    print(f"🤖 啟動 AI 實盤交易機器人 ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    
    # 1. 連線 IB
    ib = IB()
    try:
        # 真實帳戶用 4001，模擬帳戶用 4002
        ib.connect('127.0.0.1', 4001, clientId=88) 
    except Exception as e:
        print(f"❌ 無法連線至 IB Gateway: {e}")
        return

    try:
        # 2. 獲取帳戶狀態與目前持倉
        account_values = ib.accountValues()
        total_usd_value = sum(float(v.value) for v in account_values if v.tag == 'NetLiquidation' and v.currency == 'USD')
        # 如果是港幣帳戶，用 NetLiquidationByCurrency 換算
        if total_usd_value == 0:
            for val in account_values:
                if val.tag == 'NetLiquidationByCurrency' and val.currency == 'BASE':
                    total_usd_value = float(val.value) / 7.8 
                    break

        print(f"💰 帳戶估算總資產: ${total_usd_value:.2f} USD")
        
        current_positions = {p.contract.symbol: p.position for p in ib.positions() if p.position > 0}
        print(f"📦 券商實際持倉: {list(current_positions.keys())}")
        
        # 3. 讀取記憶卡 (歷史狀態)
        state = load_state()
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        today_date = datetime.datetime.strptime(today_str, "%Y-%m-%d")

        # 同步狀態：如果券商裡有，但記憶卡沒有，就手動補進去
        for t in current_positions.keys():
            if t not in state:
                state[t] = {
                    "BuyDate": today_str, 
                    "Highest": 0.0, # 稍後會更新
                    "ATR_Stop_Price": 0.0 # 暫時設為 0
                }
        # 同步狀態：如果記憶卡有，但券商裡沒有 (可能你手動賣掉了)，清除記憶
        keys_to_remove = [t for t in state.keys() if t not in current_positions]
        for t in keys_to_remove: del state[t]

        # 4. 下載今日最新數據
        print("📥 正在下載今日最新市場數據...")
        full_data = {}
        for t in list(current_positions.keys()) + TICKERS + [MARKET_INDEX]:
            # 抓 100 天就夠算指標了
            start_d = (today_date - datetime.timedelta(days=100)).strftime("%Y-%m-%d")
            df = yf.download(t, start=start_d, progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                df = add_technical_indicators(df)
                full_data[t] = df

        # ==========================================
        # 🛑 賣出檢查 (Sell Logic)
        # ==========================================
        tickers_to_sell = []
        for ticker in list(current_positions.keys()):
            if ticker not in full_data: continue
            
            df = full_data[ticker]
            curr_price = df['Close'].iloc[-1]
            ema20 = df['EMA20'].iloc[-1]
            
            # 更新記憶卡裡的最高價
            if curr_price > state[ticker]['Highest']:
                state[ticker]['Highest'] = curr_price
                
            highest_price = state[ticker]['Highest']
            atr_stop_price = state[ticker]['ATR_Stop_Price']
            buy_date = datetime.datetime.strptime(state[ticker]['BuyDate'], "%Y-%m-%d")
            held_days = (today_date - buy_date).days
            
            drop_from_peak = (curr_price - highest_price) / highest_price if highest_price > 0 else 0
            is_strong = curr_price > ema20
            sell_reason = None
            
            # 判斷邏輯完全照搬你的 Tab 3
            if curr_price <= atr_stop_price and atr_stop_price > 0:
                sell_reason = f"🛑 觸發 ATR 初始止損"
            else:
                if is_strong:
                    if drop_from_peak <= -STRONG_DROP_TOLERANCE: sell_reason = f"📉 強勢回調止盈"
                else:
                    if drop_from_peak <= -WEAK_DROP_TOLERANCE: sell_reason = f"🏃 弱勢反彈止盈"
            
            if not sell_reason and held_days >= TIME_STOP_DAYS:
                sell_reason = f"⏰ 持有達 {TIME_STOP_DAYS} 天，時間到期"
                
            if sell_reason:
                print(f"⚠️ 觸發賣出條件 [{ticker}]: {sell_reason}")
                tickers_to_sell.append(ticker)
        
        # 執行賣出下單
        for t in tickers_to_sell:
            qty = current_positions[t]
            contract = Stock(t, 'SMART', 'USD')
            ib.qualifyContracts(contract)
            order = MarketOrder('SELL', qty)
            ib.placeOrder(contract, order)
            print(f"🚀 已送出 {t} 賣出單 ({qty} 股)！")
            del state[t] # 賣出後清除記憶
            del current_positions[t] # 從目前持倉中移除
            ib.sleep(1) # 暫停 1 秒避免 API 擁擠

        # ==========================================
        # 🚀 買入檢查 (Buy Logic)
        # ==========================================
        if len(current_positions) < MAX_POSITIONS:
            market_df = full_data.get(MARKET_INDEX)
            if market_df is not None and market_df['Close'].iloc[-1] > market_df['EMA60'].iloc[-1]:
                print(f"🌍 大盤多頭 (QQQ > EMA60)，開始尋找新標的...")
                
                # 計算動能並排序
                momentum_scores = []
                for t in TICKERS:
                    if t in current_positions: continue # 已經有了不買
                    if t in full_data:
                        curr_p = full_data[t]['Close'].iloc[-1]
                        ema60 = full_data[t]['EMA60'].iloc[-1]
                        if ema60 > 0:
                            momentum_scores.append((t, curr_p / ema60))
                            
                momentum_scores.sort(key=lambda x: x[1], reverse=True)
                top_tickers = [x[0] for x in momentum_scores[:TOP_N_MOMENTUM]]
                
                # AI 模型預測
                best_ticker, best_prob = None, 0.0
                from sklearn.preprocessing import MinMaxScaler # 臨時引入給 prepare_data 用
                
                for t in top_tickers:
                    model_path = os.path.join(MODEL_DIR, f"{t}.keras")
                    if os.path.exists(model_path):
                        # 重複利用你的資料準備邏輯，稍微簡化只取最後 60 天
                        df = full_data[t]
                        features = ['Close', 'Volume', 'RSI', 'MACD', 'ATR']
                        data = df[features].values
                        scaler = MinMaxScaler(feature_range=(0, 1))
                        scaled_data = scaler.fit_transform(data)
                        
                        if len(scaled_data) >= 60:
                            last_sequence = scaled_data[-60:].reshape(1, 60, 5)
                            model = load_model(model_path, compile=False)
                            prob = float(model.predict(last_sequence, verbose=0)[0][0])
                            print(f"   🤖 AI 評估 {t}: 勝率 {prob*100:.1f}%")
                            
                            if prob > best_prob:
                                best_prob = prob
                                best_ticker = t
                                
                if best_prob > BUY_PROB_THRESHOLD and best_ticker:
                    curr_p = full_data[best_ticker]['Close'].iloc[-1]
                    curr_atr = full_data[best_ticker]['ATR'].iloc[-1]
                    
                    # 計算要買幾股 (無條件捨去成整數)
                    budget = total_usd_value * ALLOCATION_PCT
                    shares_to_buy = math.floor(budget / curr_p)
                    
                    if shares_to_buy > 0:
                        print(f"🔥 決定買入 {best_ticker}！預算: ${budget:.0f} | 股數: {shares_to_buy}")
                        
                        # 送出買單
                        contract = Stock(best_ticker, 'SMART', 'USD')
                        ib.qualifyContracts(contract)
                        order = MarketOrder('BUY', shares_to_buy)
                        ib.placeOrder(contract, order)
                        print(f"🚀 已送出 {best_ticker} 買入單！")
                        
                        # 寫入記憶卡
                        state[best_ticker] = {
                            "BuyDate": today_str,
                            "Highest": curr_p,
                            "ATR_Stop_Price": curr_p - (curr_atr * ATR_STOP_LOSS_MULTIPLIER)
                        }
            else:
                print("⚠️ 大盤未達多頭條件或資料不足，今日不買新股。")
        else:
            print("🛑 庫存已達滿倉 (3檔)，停止買入。")

        # 保存最新記憶卡狀態
        save_state(state)
        print("💾 系統狀態已儲存。")

    except Exception as e:
        print(f"❌ 發生未預期錯誤: {e}")
    finally:
        ib.disconnect()
        print("🔌 機器人執行完畢，連線已斷開。")

if __name__ == "__main__":
    run_live_strategy()