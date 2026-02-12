import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os
import json
import glob
import datetime
import sys
import io

# 🔥 強制設定標準輸出為 UTF-8，解決 Windows 下 Emoji 報錯問題
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ===========================
# 🔮 實戰預測腳本 (含 EMA20 趨勢判斷 + Windows 修復)
# ===========================
MODEL_BASE_DIR = "saved_models"
DATA_DIR = "data"

if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

def find_latest_model_dir():
    # 優先找 latest 資料夾
    latest = os.path.join(MODEL_BASE_DIR, "latest")
    if os.path.exists(latest): return latest
    
    # 否則找時間最新的
    dirs = glob.glob(os.path.join(MODEL_BASE_DIR, "*"))
    if not dirs: return None
    return max(dirs, key=os.path.getmtime)

def prepare_data(df, look_back):
    if len(df) < look_back: return None, None
    data = df.filter(['Close']).values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    last_sequence = scaled_data[-look_back:]
    last_sequence = last_sequence.reshape(1, look_back, 1)
    
    return last_sequence, scaler

def run_prediction():
    print("🤖 AI 正在分析即時市場數據 (含 EMA20 趨勢掃描)...")
    model_dir = find_latest_model_dir()
    
    if not model_dir:
        print("❌ 找不到模型，請先執行 ai_backtest.py！")
        return
        
    print(f"📂 載入模型庫: {model_dir}")
        
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        print("❌ 找不到 config.json")
        return

    with open(config_path, "r") as f:
        config = json.load(f)
        
    tickers = config.get('TICKERS', [])
    look_back = config.get('LOOK_BACK', 60)
    predict_days = config.get('PREDICT_DAYS', 10)
    
    print(f"📊 掃描 {len(tickers)} 支股票...")
    
    results = []
    
    for t in tickers:
        try:
            model_path = os.path.join(model_dir, f"{t}.keras")
            if not os.path.exists(model_path): continue
            
            # 1. 載入模型
            model = load_model(model_path)
            
            # 2. 抓取數據
            df = yf.download(t, period="2y", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            if len(df) < look_back: continue

            # 🔥 新增：計算 EMA20
            df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
            current_price = float(df['Close'].iloc[-1])
            current_ema = float(df['EMA20'].iloc[-1])
            
            # 判斷趨勢
            is_strong = current_price > current_ema
            trend_icon = "🔥" if is_strong else "❄️"
            trend_desc = "強勢多頭" if is_strong else "弱勢回調"

            # 3. 預處理 & 預測
            input_seq, scaler = prepare_data(df, look_back)
            if input_seq is None: continue
            
            preds = []
            curr_input = input_seq
            for _ in range(predict_days):
                pred = model.predict(curr_input, verbose=0)
                preds.append(pred[0, 0])
                curr_input = np.append(curr_input[:, 1:, :], [[pred[0]]], axis=1)
                
            real_preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
            max_future = float(np.max(real_preds))
            roi = float((max_future - current_price) / current_price * 100)
            
            # 4. 準備繪圖數據
            history_curve = [float(x) for x in df['Close'].iloc[-60:].values]
            forecast_curve = [float(x) for x in real_preds]
            
            # 5. 存入結果 (加入 Trend 資訊)
            res_entry = {
                "Ticker": t,
                "Current_Price": current_price,
                "EMA20": current_ema,           
                "Trend": trend_icon,            
                "Trend_Desc": trend_desc,       
                "Predicted_High": max_future,
                "ROI": roi,
                "History_Curve": history_curve,
                "Forecast_Curve": forecast_curve
            }
            results.append(res_entry)
            print(f"✅ {t:<5} {trend_icon}: 現價 ${current_price:<7.2f} (EMA:${current_ema:.2f}) -> 預測漲幅 {roi:+.2f}%")
            
        except Exception as e:
            print(f"❌ {t} 分析失敗: {e}")
            
    # 6. 排序並存檔
    results.sort(key=lambda x: x['ROI'], reverse=True)
    
    output = {
        "analysis_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "top_pick": results[0] if results else None,
        "all_rankings": results
    }
    
    json_path = os.path.join(DATA_DIR, "ai_lab_result.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=4)
    
    print("-" * 50)
    if results:
        top = results[0]
        print(f"🌟 明日最佳推薦: 【{top['Ticker']}】 {top['Trend']}")
        print(f"   理由: 趨勢{top['Trend_Desc']}，AI 預測補漲/續漲 +{top['ROI']:.2f}%")

if __name__ == "__main__":
    run_prediction()