import streamlit as st
import pandas as pd
import json
import os

st.set_page_config(page_title="AI 交易實驗室", layout="wide")

st.title("🦅 禿鷹策略監控儀表板 (Vulture Strategy)")

# 讀取數據
DATA_DIR = "data"
PORTFOLIO_FILE = os.path.join(DATA_DIR, "portfolio.json")
LOG_FILE = os.path.join(DATA_DIR, "trade_log.csv")

# 顯示當前狀態
if os.path.exists(PORTFOLIO_FILE):
    with open(PORTFOLIO_FILE, 'r') as f:
        portfolio = json.load(f)
        
    col1, col2, col3 = st.columns(3)
    
    # 計算當前預估資產
    current_val = portfolio['cash']
    holding_ticker = "無 (空手)"
    if portfolio['holdings']:
        h = portfolio['holdings']
        holding_ticker = h['Ticker']
        # 這裡可以加代碼去抓即時股價來更新市值，這邊先簡化
        current_val = "持倉中 (等待結算)" 
    
    col1.metric("當前持倉", holding_ticker)
    col2.metric("可用現金", f"${portfolio['cash']:.2f}")
    col3.metric("最後更新", portfolio['last_update'])

# 顯示交易日誌
st.subheader("📜 交易記錄")
if os.path.exists(LOG_FILE):
    df_log = pd.read_csv(LOG_FILE)
    if not df_log.empty:
        st.dataframe(df_log.sort_index(ascending=False), use_container_width=True)
        
        # 畫資產曲線
        st.subheader("📈 資產成長曲線")
        # 簡單處理：將 Balance 欄位畫出來
        chart_data = df_log[['Date', 'Balance']].set_index('Date')
        st.line_chart(chart_data)
    else:
        st.info("尚未有交易產生。策略正在等待機會...")
else:
    st.warning("找不到交易日誌。")

# 說明
st.markdown("---")
st.markdown("""
**策略邏輯：**
1. **本金:** $1000 | **手續費:** $2
2. **買入:** RSI < 30 且 跌破布林下軌 (撿屍體)
3. **賣出:** 獲利 > 20% 或 RSI > 75 (過熱) 或 止損 -15%
4. **頻率:** 每日收盤後自動掃描
""")