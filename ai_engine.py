import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os
import json
import datetime

# ===========================
# 設定與參數
# ===========================
DATA_DIR = "data"
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

# 為了節省測試時間，我們先只跑這 5 支重點股票
# 等確認沒問題了，再把整串名單放進去
TICKERS = ['NVDA', 'TSLA', 'AMZN', 'MSFT', 'GOOGL'] 

LOOK_BACK = 60  # 回看過去 60 天
FORECAST_DAYS = 10 # 預測未來 10 天

# ===========================
# 1. 數據準備 (正規化)
# ===========================
def prepare_data(df, look_back):
    data = df.filter(['Close']).values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    x_train, y_train = [], []
    # 建立滑動視窗數據
    for i in range(look_back, len(scaled_data)):
        x_train.append(scaled_data[i-look_back:i, 0])
        y_train.append(scaled_data[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    # LSTM 需要三維輸入 [Samples, Time Steps, Features]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    return x_train, y_train, scaler, scaled_data

# ===========================
# 2. 建立 LSTM 模型架構
# ===========================
def build_model(input_shape):
    model = Sequential()
    # 第一層 LSTM
    model.add(LSTM(50, return_sequences=False, input_shape=input_shape))
    # 全連接層
    model.add(Dense(25))
    # 輸出層
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ===========================
# 3. 執行分析主程序
# ===========================
def run_ai_analysis():
    print(f"🧠 AI 實驗室啟動... 準備分析 {len(TICKERS)} 檔股票")
    results = []
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # 抓取過去 2 年數據 (數據多一點訓練比較準)
    start_date = (datetime.datetime.now() - datetime.timedelta(days=730)).strftime("%Y-%m-%d")
    
    for ticker in TICKERS:
        try:
            print(f"👉 正在訓練模型: {ticker} ...")
            df = yf.download(ticker, start=start_date, progress=False)
            
            # 處理 MultiIndex (yfinance 新版問題)
            if isinstance(df.columns, pd.MultiIndex): 
                df.columns = df.columns.get_level_values(0)
            
            if len(df) < 100: 
                print(f"⚠️ {ticker} 數據不足，跳過")
                continue

            # 準備數據
            x_train, y_train, scaler, scaled_data = prepare_data(df, LOOK_BACK)
            
            # 建立並訓練模型
            # epochs=5 (訓練 5 輪，本機測試比較快)
            model = build_model((x_train.shape[1], 1))
            model.fit(x_train, y_train, batch_size=16, epochs=5, verbose=0)
            
            # --- 開始預測未來 ---
            # 拿出最後 60 天的數據作為起點
            last_sequence = scaled_data[-LOOK_BACK:]
            curr_input = last_sequence.reshape(1, LOOK_BACK, 1)
            
            predicted_prices_scaled = []
            
            # 迭代預測未來 N 天
            for _ in range(FORECAST_DAYS):
                # 預測下一天
                next_pred = model.predict(curr_input, verbose=0)
                predicted_prices_scaled.append(next_pred[0, 0])
                
                # 更新輸入數據：移除第一天，加入剛剛預測出來的一天
                # 這樣才能像接龍一樣往後預測
                curr_input = np.append(curr_input[:, 1:, :], [[next_pred[0]]], axis=1)
            
            # 將預測結果轉回真實價格
            predicted_prices = scaler.inverse_transform(np.array(predicted_prices_scaled).reshape(-1, 1))
            predicted_prices = predicted_prices.flatten().tolist()
            
            # 計算指標
            current_price = float(df['Close'].iloc[-1])
            max_future_price = max(predicted_prices)
            potential_roi = (max_future_price - current_price) / current_price * 100
            
            print(f"   ✅ {ticker} 完成 | 現價: {current_price:.1f} -> 預測高點: {max_future_price:.1f} ({potential_roi:+.2f}%)")
            
            results.append({
                "Ticker": ticker,
                "Current_Price": current_price,
                "Predicted_Max": max_future_price,
                "Potential_ROI": potential_roi,
                "Forecast_Curve": predicted_prices
            })
            
        except Exception as e:
            print(f"❌ {ticker} 失敗: {e}")

    # 排序：潛力最高的排前面
    results.sort(key=lambda x: x['Potential_ROI'], reverse=True)
    
    # 存檔
    output = {
        "analysis_date": today,
        "top_pick": results[0] if results else None,
        "all_rankings": results
    }
    
    with open(os.path.join(DATA_DIR, "ai_lab_result.json"), "w") as f:
        json.dump(output, f)
    
    print("\n🎉 分析全部完成！結果已儲存至 data/ai_lab_result.json")

if __name__ == "__main__":
    run_ai_analysis()