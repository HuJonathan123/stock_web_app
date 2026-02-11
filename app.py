import streamlit as st
import pandas as pd
import json
import os
import datetime

# ==========================================
# 1. 頁面設定
# ==========================================
st.set_page_config(page_title="投資戰情室", layout="wide", page_icon="📈")
st.title("📈 Jonathan's Investment Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(["🦅 禿鷹 (經典版)", "🚀 超級禿鷹 (壓力測試)", "🧠 AI 實驗室", "✍️ 手動日記"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

# ==========================================
# Tab 1: 禿鷹策略
# ==========================================
with tab1:
    st.header("🦅 Vulture Strategy (經典)")
    st.info("此處顯示您原本的禿鷹策略回測結果 (需上傳對應 csv)")
    # (保留您原本的代碼邏輯，若無檔案則略過)

# ==========================================
# Tab 2: 超級禿鷹
# ==========================================
with tab2:
    st.header("🚀 Super Vulture (動態止盈)")
    st.info("此處顯示超級禿鷹策略結果")

# ==========================================
# Tab 3: AI 實驗室
# ==========================================
with tab3:
    st.header("🧠 AI 趨勢預測 (基於 $3008 獲利模型)")
    
    # 1. 觸發按鈕
    if st.button("⚡️ 執行最新預測 (Run Prediction)"):
        with st.spinner("正在載入模型並分析最新股價..."):
            import subprocess
            subprocess.run(["python", "ai_predict.py"])
        st.success("更新完成！")
        st.rerun()

    # 2. 顯示最新預測
    AI_RES = os.path.join(DATA_DIR, "ai_lab_result.json")
    if os.path.exists(AI_RES):
        with open(AI_RES, 'r') as f:
            res = json.load(f)
        
        st.write(f"📅 分析日期: {res.get('analysis_date', 'N/A')}")
        
        # 最佳推薦
        top = res.get('top_pick')
        if top:
            col1, col2, col3 = st.columns(3)
            col1.metric("🌟 明日首選", top['Ticker'])
            col2.metric("預測漲幅", f"{top['ROI']:.2f}%")
            col3.metric("目標價", f"${top['Predicted_High']:.2f}")
            
            # 畫圖
            chart_data = top['History_Curve'] + top['Forecast_Curve']
            st.line_chart(chart_data)
            st.caption(f"圖表說明: 前段為過去 60 天走勢，後段為未來 10 天預測")

        # 排行榜
        st.subheader("📊 候選清單 (按漲幅排序)")
        ranks = pd.DataFrame(res['all_rankings'])
        if not ranks.empty:
            st.dataframe(ranks[['Ticker', 'Current_Price', 'Predicted_High', 'ROI']], use_container_width=True)

    # 3. 顯示回測績效
    st.divider()
    st.subheader("📜 策略歷史績效 (2025-Now)")
    BT_LOG = os.path.join(DATA_DIR, "ai_backtest_log.csv")
    BT_BAL = os.path.join(DATA_DIR, "ai_backtest_balance.csv")
    
    if os.path.exists(BT_BAL):
        df_bal = pd.read_csv(BT_BAL)
        final_equity = df_bal.iloc[-1]['Equity']
        roi = (final_equity - 1000) / 1000 * 100
        
        st.metric("回測總資產", f"${final_equity:.0f}", f"{roi:.1f}%")
        st.line_chart(df_bal.set_index('Date')['Equity'])
        
        with st.expander("查看詳細交易紀錄"):
            if os.path.exists(BT_LOG):
                st.dataframe(pd.read_csv(BT_LOG), use_container_width=True)

# ==========================================
# Tab 4: 手動日記
# ==========================================
with tab4:
    st.header("✍️ 手動模擬交易")
    MANUAL_LOG = os.path.join(DATA_DIR, "manual_log.csv")
    if not os.path.exists(MANUAL_LOG):
        pd.DataFrame(columns=["Date", "Ticker", "Action", "Price", "Shares", "Note"]).to_csv(MANUAL_LOG, index=False)
    
    with st.form("manual"):
        c1, c2, c3 = st.columns(3)
        d = c1.date_input("日期", datetime.date.today())
        t = c2.text_input("代號").upper()
        act = c3.selectbox("動作", ["BUY", "SELL"])
        p = c1.number_input("價格", min_value=0.0)
        if st.form_submit_button("提交"):
            new = pd.DataFrame([{"Date": d, "Ticker": t, "Action": act, "Price": p}])
            new.to_csv(MANUAL_LOG, mode='a', header=False, index=False)
            st.rerun()
            
    if os.path.exists(MANUAL_LOG):
        st.dataframe(pd.read_csv(MANUAL_LOG), use_container_width=True)