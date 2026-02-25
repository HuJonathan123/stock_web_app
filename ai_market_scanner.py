import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 隱藏 TensorFlow 的 C++ 層級警告

import tensorflow as tf
# 🔥 [關鍵修復] 禁用 Mac 的 GPU (MPS)，強制使用 CPU。
# 這能完美避開 M1/M2 晶片在載入 LSTM 模型預測時的 mps.slice 底層崩潰問題，且單筆預測速度更快。
tf.config.set_visible_devices([], 'GPU')

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import datetime
import json
import ta

# ===========================
# ⚙️ 掃描器設定
# ===========================
MODEL_DIR = "saved_models/latest"
MARKET_INDEX = 'QQQ'
OUTPUT_FILE = "data/latest_signals.json"

def add_technical_indicators(df):
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    df['EMA20'] = ta.trend.ema_indicator(df['Close'], window=20)
    df['EMA60'] = ta.trend.ema_indicator(df['Close'], window=60)
    df['MA30'] = ta.trend.sma_indicator(df['Close'], window=30)
    df['MA30_Slope'] = df['MA30'].diff()
    df['Price_Change'] = df['Close'].diff()
    df.fillna(method='bfill', inplace=True)
    return df

def prepare_live_data(df, look_back):
    features = ['Close', 'Volume', 'RSI', 'MACD', 'ATR', 'MA30']
    data = df[features].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # 只取最後 look_back 天的數據進行預測
    last_sequence = scaled_data[-look_back:]
    curr_input = last_sequence.reshape(1, look_back, scaled_data.shape[1])
    return curr_input

def scan_market():
    print(f"🚀 啟動 AI 雙引擎市場掃描 ({datetime.date.today()})...")
    
    # 1. 讀取模型設定
    config_path = os.path.join(MODEL_DIR, "config.json")
    if not os.path.exists(config_path):
        print("❌ 找不到 config.json，請先執行回測訓練模型。")
        return
        
    with open(config_path, "r") as f:
        config = json.load(f)
        
    TICKERS = config["TICKERS"]
    LOOK_BACK = config["LOOK_BACK"]
    
    # 下載數據 (多抓一點確保指標計算正確)
    start_date = (datetime.datetime.now() - datetime.timedelta(days=200)).strftime("%Y-%m-%d")
    
    # 2. 檢查大盤 (QQQ)
    # 🔥 [修改] 加入 auto_adjust=True 和 multi_level_index=False 讓格式更穩定
    print(f"🔍 正在下載大盤數據 {MARKET_INDEX}...")
    try:
        market_df = yf.download(MARKET_INDEX, start=start_date, progress=False, auto_adjust=True, multi_level_index=False)
    except Exception as e:
        print(f"❌ 大盤下載發生例外錯誤: {e}")
        return

    # 🔥 [關鍵防呆] 如果下載結果為空，直接結束函數，避免後面計算指標時崩潰
    if market_df.empty:
        print(f"❌ 無法下載大盤數據 {MARKET_INDEX} (數據為空)。可能原因是 yfinance 需要更新或 Yahoo 阻擋。本次掃描終止。")
        return
        
    # 如果下載成功，繼續執行
    market_df = add_technical_indicators(market_df)
    
    is_market_bullish = False
    if not market_df.empty and market_df['Close'].iloc[-1] > market_df['EMA60'].iloc[-1]:
        is_market_bullish = True
        print("🌍 大盤狀態: 多頭 (QQQ > EMA60)")
    else:
        print("⚠️ 大盤狀態: 空頭 (QQQ < EMA60)，建議空手或輕倉")

    # 3. 掃描個股
    full_data = {}
    momentum_scores = []
    
    for t in TICKERS:
        try:
            df = yf.download(t, start=start_date, progress=False, auto_adjust=True, multi_level_index=False)
        except:
            continue

        # 🔥 [防呆] 確保數據不為空且長度足夠
        if df.empty or len(df) < LOOK_BACK + 20: 
            continue

        df = add_technical_indicators(df)
        full_data[t] = df
        # ... (後面的代碼保持不變)
        
        # 計算動能分數
        ema60 = df['EMA60'].iloc[-1]
        if ema60 > 0:
            score = df['Close'].iloc[-1] / ema60
            momentum_scores.append((t, score))

    momentum_scores.sort(key=lambda x: x[1], reverse=True)
    top_3_tickers = [x[0] for x in momentum_scores[:3]]
    
    signals = {
        "scan_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "market_bullish": is_market_bullish,
        "strategy_1_top3": [],    # 策略 1: Top 3 無限奔跑
        "strategy_2_ma30": []     # 策略 2: MA30 強力突破
    }

    print("\n🧠 正在進行 AI 預測...")
    for t, df in full_data.items():
        curr_price = df['Close'].iloc[-1]
        ma30 = df['MA30'].iloc[-1]
        ma30_slope = df['MA30_Slope'].iloc[-1]
        price_change = df['Price_Change'].iloc[-1]
        
        # 載入模型並預測
        model_path = os.path.join(MODEL_DIR, f"{t}.keras")
        if not os.path.exists(model_path): continue
        
        try:
            model = load_model(model_path, compile=False)
            curr_input = prepare_live_data(df, LOOK_BACK)
            prob = float(model.predict(curr_input, verbose=0)[0][0])
        except Exception as e:
            # print(f"預測 {t} 時發生錯誤: {e}")
            continue

        signal_info = {
            "ticker": t,
            "price": round(curr_price, 2),
            "probability": round(prob * 100, 1),
            "ma30_distance": round((curr_price - ma30) / ma30 * 100, 1)
        }

        # 🎯 策略 1: Top 3 動能 + MA30 確認 (機率 > 55%)
        if t in top_3_tickers and ma30_slope > 0 and curr_price > (ma30 * 1.01):
            if prob >= 0.55:
                signals["strategy_1_top3"].append(signal_info)

        # 🎯 策略 2: MA30 強力突破 + 5% 緩衝 (機率 > 55%)
        if ma30_slope > 0 and price_change > 0 and curr_price > (ma30 * 1.05):
            if prob >= 0.55:
                signals["strategy_2_ma30"].append(signal_info)

    # 排序：勝率高的排前面
    signals["strategy_1_top3"].sort(key=lambda x: x["probability"], reverse=True)
    signals["strategy_2_ma30"].sort(key=lambda x: x["probability"], reverse=True)

    # 輸出成 JSON 供網頁使用
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(signals, f, indent=4)
        
    print(f"\n✅ 掃描完成！結果已保存至 {OUTPUT_FILE}")
    print(f"🏅 策略 1 (Top 3) 推薦: {[s['ticker'] for s in signals['strategy_1_top3']]}")
    print(f"💥 策略 2 (MA30突破) 推薦: {[s['ticker'] for s in signals['strategy_2_ma30']]}")

if __name__ == "__main__":
    scan_market()