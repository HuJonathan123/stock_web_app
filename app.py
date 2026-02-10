import streamlit as st
import pandas as pd
import json
import os
import datetime

# 設定頁面
st.set_page_config(page_title="AI 投資戰情室", layout="wide", page_icon="📈")
st.title("📈 Jonathan's AI Investment Dashboard")

# 建立分頁
tab1, tab2, tab3 = st.tabs(["🦅 禿鷹策略 (自動)", "🤖 實驗室模型", "✍️ 手動交易日記"])

# 🔥🔥🔥【修正核心：使用絕對路徑】🔥🔥🔥
# 1. 抓出 app.py 所在的絕對位置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. 設定 data 資料夾的絕對路徑
DATA_DIR = os.path.join(BASE_DIR, "data")
PORTFOLIO_FILE = os.path.join(DATA_DIR, "portfolio.json")
LOG_FILE = os.path.join(DATA_DIR, "trade_log.csv")
MANUAL_LOG_FILE = os.path.join(DATA_DIR, "manual_log.csv")

# 3. 強制建立資料夾 (如果不存在)
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
# 🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥

# ==========================================
# Tab 1: 禿鷹策略 (自動化)
# ==========================================
with tab1:
    st.header("🦅 Vulture Strategy (自動化監控)")
    
    # 初始化預設值 (防止檔案讀取失敗)
    portfolio = {"cash": 1000, "holdings": None, "last_update": "尚未更新"}
    
    # 嘗試讀取檔案
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, 'r') as f:
                # 檢查檔案是否為空
                if os.stat(PORTFOLIO_FILE).st_size > 0:
                    portfolio = json.load(f)
        except json.JSONDecodeError:
            st.warning("⚠️ 投資組合檔案 (portfolio.json) 格式錯誤或為空，已使用預設值。")
    
    # 顯示指標卡片
    col1, col2, col3 = st.columns(3)
    
    status_text = "無 (空手)"
    if portfolio.get('holdings'): # 使用 .get 防止 KeyError
        h = portfolio['holdings']
        status_text = f"{h['Ticker']} ({h['Shares']:.2f} 股)"
    
    col1.metric("當前持倉", status_text)
    col2.metric("可用現金", f"${portfolio['cash']:.2f}")
    col3.metric("最後更新", portfolio.get('last_update', '未知'))

    # 讀取交易日誌
    if os.path.exists(LOG_FILE):
        df_log = pd.read_csv(LOG_FILE)
        if not df_log.empty:
            st.subheader("📜 歷史交易 (自 2026-01-01 起)")
            
            # 讓表格更漂亮
            st.dataframe(
                df_log.sort_index(ascending=False), 
                use_container_width=True,
                column_config={
                    "Price": st.column_config.NumberColumn(format="$%.2f"),
                    "Balance": st.column_config.NumberColumn(format="$%.2f"),
                }
            )
            
            # 畫圖
            st.subheader("📈 資產成長曲線")
            chart_data = df_log[['Date', 'Balance']].copy()
            chart_data['Date'] = pd.to_datetime(chart_data['Date'])
            chart_data = chart_data.set_index('Date')
            st.line_chart(chart_data)
        else:
            st.info("暫無交易紀錄。")

    st.markdown("---")
    st.caption("策略邏輯：本金 $1000 | 每次手續費 $2 | RSI < 30 買入 | 獲利 > 20% 賣出")

# ==========================================
# Tab 2: 其他模型 (預留空間)
# ==========================================
with tab2:
    st.header("🤖 Alpha 實驗室")
    st.write("這裡可以放置你的 Transformer 模型預測結果、回測數據，或是瘋狗流策略的監控。")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("Transformer 預測 (開發中)")
        st.write("今日最強預測：NVDA (+1.2%)")
    with col2:
        st.warning("瘋狗流策略 (開發中)")
        st.write("今日訊號：無 (VIX 過高)")

# ==========================================
# Tab 3: 手動交易日記 (模擬)
# ==========================================
# ... (在 Tab 3 裡面)
with tab3:
    st.header("✍️ 手動模擬交易紀錄")
    st.write("在這裡記錄你自己的模擬操作，系統會幫你計算損益。")

    # 🔥【修正點】先確保 data 資料夾存在
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # 確保手動日誌檔案存在
    if not os.path.exists(MANUAL_LOG_FILE):
        pd.DataFrame(columns=["Date", "Ticker", "Action", "Price", "Shares", "Note"]).to_csv(MANUAL_LOG_FILE, index=False)

    # 輸入區塊
    with st.expander("➕ 新增交易紀錄", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            m_date = st.date_input("日期", datetime.date.today())
            m_ticker = st.text_input("股票代號 (如 TSLA)").upper()
        with c2:
            m_action = st.selectbox("動作", ["BUY", "SELL"])
            m_price = st.number_input("價格", min_value=0.0, step=0.01)
        with c3:
            m_shares = st.number_input("股數", min_value=0.0, step=0.1)
            m_note = st.text_input("筆記 (選填)")
        
        if st.button("提交紀錄"):
            if m_ticker and m_price > 0 and m_shares > 0:
                new_record = {
                    "Date": m_date,
                    "Ticker": m_ticker,
                    "Action": m_action,
                    "Price": m_price,
                    "Shares": m_shares,
                    "Note": m_note
                }
                pd.DataFrame([new_record]).to_csv(MANUAL_LOG_FILE, mode='a', header=False, index=False)
                st.success(f"已儲存：{m_action} {m_ticker}")
                st.rerun() # 重新整理頁面顯示最新數據
            else:
                st.error("請填寫完整資訊")

    # 顯示紀錄與簡單統計
    if os.path.exists(MANUAL_LOG_FILE):
        df_manual = pd.read_csv(MANUAL_LOG_FILE)
        
        if not df_manual.empty:
            # 簡單損益計算 (示意)
            total_invested = 0
            realized_pnl = 0
            
            # 顯示表格
            st.subheader("交易明細")
            st.dataframe(df_manual.sort_index(ascending=False), use_container_width=True)
            
            # 下載功能
            csv = df_manual.to_csv(index=False).encode('utf-8')
            st.download_button("下載 CSV", csv, "my_trade_log.csv", "text/csv")