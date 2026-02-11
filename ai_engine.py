import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import os
import json
import datetime

# ===========================
# ⚙️ 統一參數 (與回測一致)
# ===========================
DATA_DIR = "data"
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

TICKERS = [
    'NVDA', 'TSLA', 'AMZN', 'MSFT', 'GOOGL', 'META', 'AAPL', 
    'AMD', 'INTC', 'QCOM', 'AVGO', 'MU',                     
    'JPM', 'V', 'DIS', 'NFLX', 'COST', 'PEP', 'KO', 'JNJ'    
]

LOOK_BACK = 60
FORECAST_DAYS = 10 
EPOCHS = 20 # 預測未來時我們可以訓練久一點，讓線條更準

def prepare_data(df, look_back):
    data = df.filter(['Close']).values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    x_train, y_train = [], []
    for i in range(look_back, len(scaled_data)):
        x_train.append(scaled_data[i-look_back:i, 0])
        y_train.append(scaled_data[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    return x_train, y_train, scaler, scaled_data

def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def run_ai_analysis():
    print(f"🧠 AI 預測引擎啟動 (個別訓練 LookBack={LOOK_BACK})...")
    results = []
    
    # 設定下載起點 (往前推 2 年)
    start_date = (datetime.datetime.now() - datetime.timedelta(days=730)).strftime("%Y-%m-%d")
    
    for ticker in TICKERS:
        try:
            print(f"👉 分析 {ticker}...")
            df = yf.download(ticker, start=start_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            if len(df) < 100: continue

            # 1. 個別準備數據 (確保每個股票有自己的標準)
            x_train, y_train, scaler, scaled_data = prepare_data(df, LOOK_BACK)
            
            # 2. 個別訓練模型 (這是避免 1.65% 複製貼上的關鍵)
            model = build_model((x_train.shape[1], 1))
            model.fit(x_train, y_train, batch_size=16, epochs=EPOCHS, verbose=0)
            
            # 3. 預測未來
            last_sequence = scaled_data[-LOOK_BACK:]
            curr_input = last_sequence.reshape(1, LOOK_BACK, 1)
            predicted_prices_scaled = []
            
            for _ in range(FORECAST_DAYS):
                pred = model.predict(curr_input, verbose=0)
                predicted_prices_scaled.append(pred[0, 0])
                curr_input = np.append(curr_input[:, 1:, :], [[pred[0]]], axis=1)
            
            # 4. 還原價格與計算 ROI
            predicted_prices = scaler.inverse_transform(np.array(predicted_prices_scaled).reshape(-1, 1)).flatten().tolist()
            current_price = float(df['Close'].iloc[-1])
            max_future = max(predicted_prices)
            roi = (max_future - current_price) / current_price * 100
            
            # 5. 抓取歷史數據 (畫圖用)
            history_prices = df['Close'].iloc[-60:].values.tolist()

            results.append({
                "Ticker": ticker,
                "Current_Price": current_price,
                "Predicted_Max": max_future,
                "Potential_ROI": roi,
                "Forecast_Curve": predicted_prices,
                "History_Curve": history_prices
            })
            print(f"   ✅ 預測 ROI: {roi:.2f}%")
            
        except Exception as e:
            print(f"❌ {ticker} 失敗: {e}")

    # 排序並存檔
    results.sort(key=lambda x: x['Potential_ROI'], reverse=True)
    
    output = {
        "analysis_date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "top_pick": results[0] if results else None,
        "all_rankings": results
    }
    
    with open(os.path.join(DATA_DIR, "ai_lab_result.json"), "w") as f:
        json.dump(output, f)
    
    print("🎉 分析全部完成！結果已更新。")

if __name__ == "__main__":
    run_ai_analysis()