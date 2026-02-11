import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os
import json
import glob
import datetime

# ===========================
# 🔮 實戰預測腳本 (JSON 修復版)
# ===========================
MODEL_BASE_DIR = "saved_models"
DATA_DIR = "data"

def find_latest_model_dir():
    # 優先找 latest 資料夾
    latest = os.path.join(MODEL_BASE_DIR, "latest")
    if os.path.exists(latest): return latest
    
    # 否則找最新的
    dirs = glob.glob(os.path.join(MODEL_BASE_DIR, "*"))
    if not dirs: return None
    return max(dirs, key=os.path.getmtime)

def prepare_data(df, look_back):
    if len(df) < look_back: return None, None
    data = df.filter(['Close']).values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    last_sequence = scaled_data[-look_back:].reshape(1, look_back, 1)
    return last_sequence, scaler

def run_prediction():
    print("🤖 AI 正在分析即時市場數據...")
    model_dir = find_latest_model_dir()
    if not model_dir:
        print("❌ 找不到模型，請先執行回測！")
        return
        
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        print("❌ 找不到 config.json")
        return

    with open(config_path, "r") as f:
        config = json.load(f)
        
    tickers = config.get('TICKERS', [])
    look_back = config.get('LOOK_BACK', 60)
    predict_days = config.get('PREDICT_DAYS', 10)
    
    results = []
    
    for t in tickers:
        try:
            model_path = os.path.join(model_dir, f"{t}.keras")
            if not os.path.exists(model_path): continue
            
            model = load_model(model_path)
            # 抓取最近 1.5 年數據以確保有足夠的 Lookback
            df = yf.download(t, period="2y", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            if len(df) < look_back: continue

            input_seq, scaler = prepare_data(df, look_back)
            if input_seq is None: continue
            
            preds = []
            curr_input = input_seq
            for _ in range(predict_days):
                pred = model.predict(curr_input, verbose=0)
                preds.append(pred[0, 0])
                curr_input = np.append(curr_input[:, 1:, :], [[pred[0]]], axis=1)
                
            real_preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
            
            # 🔥 強制轉型為 Python 原生 float (解決 JSON error)
            current_price = float(df['Close'].iloc[-1])
            max_future = float(np.max(real_preds))
            roi = float((max_future - current_price) / current_price * 100)
            
            # 歷史數據 (供畫圖) - 同樣強制轉型
            hist = [float(x) for x in df['Close'].iloc[-60:].values]
            forecast = [float(x) for x in real_preds]
            
            results.append({
                "Ticker": t,
                "Current_Price": current_price,
                "Predicted_High": max_future,
                "ROI": roi,
                "Forecast_Curve": forecast,
                "History_Curve": hist
            })
            print(f"✅ {t}: {roi:+.2f}%")
            
        except Exception as e:
            print(f"❌ {t} 失敗: {e}")
            
    results.sort(key=lambda x: x['ROI'], reverse=True)
    
    output = {
        "analysis_date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "top_pick": results[0] if results else None,
        "all_rankings": results
    }
    
    with open(os.path.join(DATA_DIR, "ai_lab_result.json"), "w") as f:
        json.dump(output, f)
    
    print("🎉 預測完成！結果已更新。")

if __name__ == "__main__":
    run_prediction()