import streamlit as st
import pandas as pd
import json
import os
import datetime
import altair as alt # 引入繪圖庫

# ==========================================
# 1. 頁面設定
# ==========================================
st.set_page_config(page_title="AI 投資戰情室", layout="wide", page_icon="📈")
st.title("📈 Jonathan's AI Investment Dashboard")

# 建立分頁
tab1, tab2, tab3, tab4 = st.tabs(["🦅 禿鷹 (經典版)", "🚀 超級禿鷹 (壓力測試)", "🧠 AI 實驗室", "✍️ 手動日記"])

# 路徑設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

# 顯示最後更新時間
META_FILE = os.path.join(DATA_DIR, "meta.json")
if os.path.exists(META_FILE):
    with open(META_FILE, 'r') as f:
        meta = json.load(f)
        st.caption(f"數據最後更新於：{meta.get('last_updated', '未知')}")

# ==========================================
# Tab 1: 禿鷹策略 (經典版)
# ==========================================
with tab1:
    st.header("🦅 Vulture Strategy (經典 All-in)")
    st.caption("✅ 你的基準策略 | 規則：固定 20% 止盈 | 15% 止損 | 15 天持有上限")
    
    # 年份選擇器
    PERIOD_OPTIONS = {
        "2025-Now (當前)": "2025_now",
        "2024 (AI 牛市)": "2024_bull",
        "2023 (震盪復甦)": "2023_recovery",
        "2022 (崩盤熊市)": "2022_bear"
    }
    
    c_sel, _ = st.columns([1, 3])
    with c_sel:
        v_period = st.selectbox("📅 選擇年份", list(PERIOD_OPTIONS.keys()), key="v_sel")
    
    v_key = PERIOD_OPTIONS[v_period]
    b_file = os.path.join(DATA_DIR, f"vulture_{v_key}_balance.csv")
    l_file = os.path.join(DATA_DIR, f"vulture_{v_key}_log.csv")

    if os.path.exists(b_file):
        df = pd.read_csv(b_file)
        if not df.empty:
            final_eq = df.iloc[-1]['Equity']
            roi = (final_eq - 1000) / 1000 * 100
            color = "green" if roi >= 0 else "red"
            st.metric("最終淨值", f"${final_eq:,.2f}", f"{roi:.2f}%")
            st.line_chart(df.set_index('Date')['Equity'])
    
    if os.path.exists(l_file):
        with st.expander("查看交易紀錄"):
            st.dataframe(pd.read_csv(l_file).sort_index(ascending=False), use_container_width=True)

# ==========================================
# Tab 2: 超級禿鷹 (壓力測試)
# ==========================================
with tab2:
    st.header("🚀 Super Vulture (動態追蹤)")
    st.caption("🧪 實驗規則：不止盈(讓獲利奔跑) | 高點回吐 5% 離場 | 10% 嚴格止損")
    
    c_sel2, _ = st.columns([1, 3])
    with c_sel2:
        sv_period = st.selectbox("📅 選擇年份", list(PERIOD_OPTIONS.keys()), key="sv_sel")
    
    sv_key = PERIOD_OPTIONS[sv_period]
    sb_file = os.path.join(DATA_DIR, f"super_vulture_{sv_key}_balance.csv")
    sl_file = os.path.join(DATA_DIR, f"super_vulture_{sv_key}_log.csv")

    if os.path.exists(sb_file):
        df = pd.read_csv(sb_file)
        if not df.empty:
            final_eq = df.iloc[-1]['Equity']
            roi = (final_eq - 1000) / 1000 * 100
            st.metric("最終淨值", f"${final_eq:,.2f}", f"{roi:.2f}%")
            st.line_chart(df.set_index('Date')['Equity'])

    if os.path.exists(sl_file):
        with st.expander("查看交易紀錄"):
            st.dataframe(pd.read_csv(sl_file).sort_index(ascending=False), use_container_width=True)

# ==========================================
# Tab 3: AI 實驗室 (LSTM Model)
# ==========================================
with tab3:
    st.header("🧠 AI 趨勢預測實驗室 (LSTM)")
    st.caption("實驗原理：利用深度學習分析過去 60 天走勢，預測未來 10 天 (2週) 表現。")
    
    # 檔案路徑
    AI_FILE = os.path.join(DATA_DIR, "ai_lab_result.json")
    
    # 按鈕：手動觸發 AI 分析 (在本機跑很有用)
    if st.button("⚡️ 啟動 AI 運算 (需耗時約 1-2 分鐘)"):
        with st.spinner("正在訓練神經網絡...請稍候 (你可以看終端機的進度)"):
            # 這裡使用 subprocess 呼叫外部 python 腳本
            import subprocess
            subprocess.run(["python", "ai_engine.py"])
        st.success("分析完成！請查看下方結果")
        st.rerun() # 重新整理頁面

    if os.path.exists(AI_FILE):
        try:
            with open(AI_FILE, 'r') as f:
                ai_data = json.load(f)
            
            update_date = ai_data.get('analysis_date', '未知')
            top_pick = ai_data.get('top_pick', {})
            
            st.markdown(f"**最後分析時間：** `{update_date}`")
            st.divider()

            # 1. 顯示冠軍股票
            if top_pick:
                ticker = top_pick['Ticker']
                roi = top_pick['Potential_ROI']
                
                st.subheader(f"🏆 AI 首選：{ticker}")
                
                # 數據卡片
                col1, col2, col3 = st.columns(3)
                col1.metric("當前價格", f"${top_pick['Current_Price']:.2f}")
                col2.metric("預測高點 (10天內)", f"${top_pick['Predicted_Max']:.2f}")
                col3.metric("預期漲幅", f"{roi:.2f}%", delta_color="normal" if roi > 0 else "inverse")
                
                # 繪製預測圖
                st.markdown("#### 🔮 未來 10 天價格走勢預測")
                forecast_data = top_pick['Forecast_Curve']
                
                # 生成未來日期作為 X 軸
                start_dt = datetime.datetime.strptime(update_date, "%Y-%m-%d")
                future_dates = [(start_dt + datetime.timedelta(days=i)).strftime("%m-%d") for i in range(1, 11)]
                
                chart_df = pd.DataFrame({
                    "Date": future_dates,
                    "Predicted Price": forecast_data
                }).set_index("Date")
                
                st.line_chart(chart_df)
                
                # AI 建議
                if roi > 5.0:
                    st.success(f"🚀 強力買入訊號：AI 預測 {ticker} 短期動能強勁！")
                elif roi > 0:
                    st.info(f"👀 觀望訊號：{ticker} 趨勢向上，但幅度不大。")
                else:
                    st.error(f"🐻 空頭訊號：AI 預測 {ticker} 未來兩週可能下跌。")
                    
            # 2. 顯示完整排行榜
            st.divider()
            with st.expander("📊 查看所有股票預測排行", expanded=True):
                rankings = ai_data.get('all_rankings', [])
                if rankings:
                    df_rank = pd.DataFrame(rankings)
                    # 整理表格欄位
                    df_rank = df_rank[['Ticker', 'Current_Price', 'Predicted_Max', 'Potential_ROI']]
                    df_rank.columns = ['代號', '現價', '預測高點', '預期漲幅(%)']
                    
                    st.dataframe(
                        df_rank.style.format({
                            "現價": "${:.2f}", 
                            "預測高點": "${:.2f}", 
                            "預期漲幅(%)": "{:.2f}%"
                        }).background_gradient(subset=['預期漲幅(%)'], cmap='RdYlGn'), 
                        use_container_width=True
                    )
        except Exception as e:
            st.error(f"讀取數據失敗: {e}")
    else:
        st.info("👈 請點擊上方的「啟動 AI 運算」按鈕開始第一次分析。")

    # ... (Tab 3 前面的代碼保持不變)

    # ==========================================
    # 新增：AI 回測報告
    # ==========================================
    st.divider()
    st.subheader("📜 歷史回測驗證 (2025 - Now)")
    st.caption("模擬情境：本金 $1000 | 每次空手時 AI 重新預測 | 買入信心最高的股票 | 賺15%走/賠8%砍")

    BT_LOG = os.path.join(DATA_DIR, "ai_backtest_log.csv")
    BT_BAL = os.path.join(DATA_DIR, "ai_backtest_balance.csv")

    if os.path.exists(BT_BAL) and os.path.exists(BT_LOG):
        df_bal = pd.read_csv(BT_BAL)
        df_log = pd.read_csv(BT_LOG)
        
        if not df_bal.empty:
            # 1. 績效指標
            final_eq = df_bal.iloc[-1]['Equity']
            total_roi = (final_eq - 1000) / 1000 * 100
            
            c1, c2, c3 = st.columns(3)
            c1.metric("最終資產", f"${final_eq:,.2f}")
            c2.metric("總報酬率", f"{total_roi:.2f}%", delta_color="normal" if total_roi > 0 else "inverse")
            c3.metric("總交易次數", len(df_log[df_log['Action']=='SELL']))
            
            # 2. 曲線圖
            st.line_chart(df_bal.set_index('Date')['Equity'])
            
            # 3. 交易明細
            with st.expander("查看 AI 的詳細買賣紀錄"):
                st.dataframe(df_log, use_container_width=True)
    else:
        st.info("尚未執行回測。請在本地執行 `python ai_backtest.py` 來生成報告。")

# ==========================================
# Tab 4: 手動日記
# ==========================================
with tab4:
    st.header("✍️ 手動模擬交易")
    MANUAL_LOG = os.path.join(DATA_DIR, "manual_log.csv")
    if not os.path.exists(MANUAL_LOG):
        pd.DataFrame(columns=["Date", "Ticker", "Action", "Price", "Shares", "Note"]).to_csv(MANUAL_LOG, index=False)
    
    with st.expander("➕ 新增交易", expanded=True):
        with st.form("manual"):
            c1, c2, c3 = st.columns(3)
            d = c1.date_input("日期", datetime.date.today())
            t = c2.text_input("代號").upper()
            act = c3.selectbox("動作", ["BUY", "SELL"])
            p = c1.number_input("價格", min_value=0.0)
            s = c2.number_input("股數", min_value=0.0)
            n = c3.text_input("筆記")
            if st.form_submit_button("提交"):
                if t and p > 0:
                    new = pd.DataFrame([{"Date": d, "Ticker": t, "Action": act, "Price": p, "Shares": s, "Note": n}])
                    new.to_csv(MANUAL_LOG, mode='a', header=False, index=False)
                    st.success("已儲存")
                    st.rerun()
    if os.path.exists(MANUAL_LOG):
        st.dataframe(pd.read_csv(MANUAL_LOG).sort_index(ascending=False), use_container_width=True)