import streamlit as st
import pandas as pd
import json
import os
import datetime
import subprocess

# ==========================================
# ⚙️ 頁面設定
# ==========================================
st.set_page_config(page_title="AI 投資戰情室", layout="wide", page_icon="📈")
st.title("📈 Jonathan's AI Investment Dashboard")

# CSS 優化表格顯示 (修正版：適配深色模式)
st.markdown("""
<style>
    /* 讓 Metric 卡片背景變深色，文字變白色，並加上邊框 */
    div[data-testid="stMetric"] {
        background-color: #262730; /* 深灰色背景 */
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #464b5f; /* 邊框讓它更明顯 */
        color: white; /* 強制文字白色 */
    }
    
    /* 讓 Metric 的 Label (標題) 也清楚顯示 */
    div[data-testid="stMetricLabel"] p {
        color: #b0b0b0 !important; /* 淺灰色標題 */
    }

    div[data-testid="stDataFrame"] {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# 定義 Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🦅 禿鷹 (經典版)", "🚀 超級禿鷹 (進化版)", "🤖 AI 實驗室 (最新推薦)", "✍️ 手動日記"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

# 定義年份選項
PERIOD_OPTIONS = {
    "2025-Now (當前)": "2025_now",
    "2024 (AI 牛市)": "2024_bull",
    "2023 (震盪復甦)": "2023_recovery",
    "2022 (崩盤熊市)": "2022_bear"
}

# ==========================================
# 🔧 共用函數
# ==========================================
def render_strategy_view(strategy_prefix, strategy_title, strategy_desc):
    st.header(strategy_title)
    st.caption(strategy_desc)
    
    col_sel, _ = st.columns([1, 3])
    with col_sel:
        selected_label = st.selectbox(
            "📅 選擇回測年份：", 
            list(PERIOD_OPTIONS.keys()), 
            key=f"sel_{strategy_prefix}"
        )
    
    period_key = PERIOD_OPTIONS[selected_label]
    b_file = os.path.join(DATA_DIR, f"{strategy_prefix}_{period_key}_balance.csv")
    l_file = os.path.join(DATA_DIR, f"{strategy_prefix}_{period_key}_log.csv")
    
    # 1. 顯示資產曲線
    if os.path.exists(b_file):
        df = pd.read_csv(b_file)
        if not df.empty:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            
            final_eq = df.iloc[-1]['Equity']
            roi = (final_eq - 1000) / 1000 * 100
            
            color = "green" if roi >= 0 else "red"
            emoji = "🎉" if roi >= 0 else "🩸"
            
            st.subheader(f"📈 {selected_label} 資產走勢")
            c1, c2 = st.columns(2)
            c1.markdown(f"## 最終淨值: **${final_eq:,.2f}**")
            c2.markdown(f"## 報酬率: :{color}[{emoji} {roi:.2f}%]")
            
            st.line_chart(df['Equity'], color="#29b5e8")
        else:
            st.warning("⚠️ 數據為空。")
    else:
        st.info(f"找不到數據檔案：{b_file}")

    # 2. 顯示交易紀錄
    if os.path.exists(l_file):
        df_log = pd.read_csv(l_file)
        if not df_log.empty:
            with st.expander(f"📜 查看 {selected_label} 詳細交易紀錄", expanded=False):
                st.dataframe(
                    df_log.sort_index(ascending=False), 
                    use_container_width=True,
                    column_config={
                        "Price": st.column_config.NumberColumn(format="$%.2f"),
                        "Profit_USD": st.column_config.NumberColumn(format="$%.2f"),
                        "Profit_Pct": st.column_config.NumberColumn(format="%.2f%%"),
                        "Balance": st.column_config.NumberColumn(format="$%.2f"),
                    }
                )

# ==========================================
# Tab 1 & 2: 舊策略回顧
# ==========================================
with tab1:
    render_strategy_view("vulture", "🦅 Vulture Strategy", "經典 All-in 策略")

with tab2:
    render_strategy_view("super_vulture", "🚀 Super Vulture", "動態追蹤策略")

# ==========================================
# Tab 3: AI 實驗室 (核心功能)
# ==========================================
with tab3:
    st.header("🧠 AI 智能趨勢實驗室")
    st.caption("基於最新 LSTM 模型 + EMA20 趨勢濾網 + 動態止盈策略")
    
    # --- A. 控制區 ---
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button("⚡️ 執行最新預測", type="primary"):
            with st.spinner("🤖 AI 正在掃描市場數據 (含 EMA 趨勢判斷)..."):
                try:
                    # 執行 ai_predict.py
                    result = subprocess.run(["python", "ai_predict.py"], capture_output=True, text=True)
                    if result.returncode == 0:
                        st.success("✅ 預測完成！")
                        st.rerun()
                    else:
                        st.error(f"❌ 執行失敗: {result.stderr}")
                except Exception as e:
                    st.error(f"❌ 錯誤: {e}")

    # --- B. 顯示 AI 預測結果 ---
    AI_RES = os.path.join(DATA_DIR, "ai_lab_result.json")
    
    if os.path.exists(AI_RES):
        with open(AI_RES, 'r') as f:
            res = json.load(f)
        
        st.write(f"📅 分析時間: **{res.get('analysis_date', 'N/A')}**")
        
        # 1. 冠軍卡片
        top = res.get('top_pick')
        if top:
            st.divider()
            st.subheader(f"🌟 明日冠軍推薦：{top['Ticker']}")
            
            # 判斷趨勢顏色
            trend_color = "red" if "🔥" in top.get('Trend', '') else "blue"
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("當前股價", f"${top['Current_Price']:.2f}")
            c2.metric("預測目標", f"${top['Predicted_High']:.2f}", f"{top['ROI']:.2f}%")
            c3.metric("EMA20 趨勢線", f"${top.get('EMA20', 0):.2f}")
            c4.markdown(f"### 趨勢: :{trend_color}[{top.get('Trend', '')} {top.get('Trend_Desc', '')}]")
            
            # 畫圖
            hist_data = top.get('History_Curve', [])
            forecast_data = top.get('Forecast_Curve', [])
            
            if hist_data and forecast_data:
                chart_data = pd.DataFrame({
                    "Price": hist_data + forecast_data,
                    "Type": ["History"] * len(hist_data) + ["Forecast"] * len(forecast_data)
                }).reset_index()
                
                st.line_chart(chart_data, x="index", y="Price", color="Type")
                st.caption("🔵 藍色: 過去60天走勢 | 🟠 橘色: 未來10天AI預測")
        
        # 2. 完整排行榜
        st.divider()
        st.subheader("📊 全市場潛力排行榜")
        
        rankings = res.get('all_rankings', [])
        if rankings:
            df_rank = pd.DataFrame(rankings)
            
            # 🛠️ 防呆：如果資料是舊的，自動補上預設值，避免崩潰
            if 'Trend' not in df_rank.columns: df_rank['Trend'] = "❓"
            if 'EMA20' not in df_rank.columns: df_rank['EMA20'] = 0.0
            
            # 整理顯示欄位
            df_display = df_rank[['Ticker', 'Trend', 'Current_Price', 'EMA20', 'Predicted_High', 'ROI']]
            df_display.columns = ['代號', '趨勢', '現價', 'EMA20均線', 'AI預測高點', '預測漲幅%']
            
            st.dataframe(
                df_display, 
                use_container_width=True,
                column_config={
                    "現價": st.column_config.NumberColumn(format="$%.2f"),
                    "EMA20均線": st.column_config.NumberColumn(format="$%.2f"),
                    "AI預測高點": st.column_config.NumberColumn(format="$%.2f"),
                    "預測漲幅%": st.column_config.NumberColumn(format="%.2f%%"),
                }
            )
    else:
        st.info("👋 尚未有預測數據，請點擊上方按鈕執行預測。")

    # --- C. 顯示回測交易紀錄 ---
    st.divider()
    st.subheader("📜 歷史 AI 自動交易紀錄 (Backtest Log)")
    
    BT_LOG = os.path.join(DATA_DIR, "ai_backtest_log.csv")
    BT_BAL = os.path.join(DATA_DIR, "ai_backtest_balance.csv")
    
    col1, col2 = st.columns([1, 1])
    
    # 左側：資產曲線
    with col1:
        if os.path.exists(BT_BAL):
            df_bal = pd.read_csv(BT_BAL)
            final_equity = df_bal.iloc[-1]['Equity']
            total_ret = (final_equity - 1000) / 1000 * 100
            st.metric("目前模擬倉總資產", f"${final_equity:,.2f}", f"{total_ret:.1f}%")
            st.line_chart(df_bal.set_index('Date')['Equity'], height=250)
    
    # 右側：最新交易列表
    with col2:
        if os.path.exists(BT_LOG):
            df_bt = pd.read_csv(BT_LOG)
            # 顯示最近 10 筆就好
            st.write("最新 10 筆交易:")
            st.dataframe(
                df_bt.tail(10).sort_index(ascending=False), 
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Price": st.column_config.NumberColumn(format="$%.2f"),
                    "Profit_USD": st.column_config.NumberColumn(format="$%.2f"),
                    "Profit_Pct": st.column_config.NumberColumn(format="%.2f%%"),
                }
            )
            
    # 完整紀錄展開
    if os.path.exists(BT_LOG):
        with st.expander("查看完整歷史交易清單"):
            st.dataframe(
                pd.read_csv(BT_LOG).sort_index(ascending=False),
                use_container_width=True,
                column_config={
                    "Price": st.column_config.NumberColumn(format="$%.2f"),
                    "Profit_USD": st.column_config.NumberColumn(format="$%.2f"),
                    "Profit_Pct": st.column_config.NumberColumn(format="%.2f%%"),
                    "Balance": st.column_config.NumberColumn(format="$%.2f"),
                }
            )

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
            st.success("紀錄已儲存")
            st.rerun()
            
    if os.path.exists(MANUAL_LOG):
        st.dataframe(pd.read_csv(MANUAL_LOG), use_container_width=True)