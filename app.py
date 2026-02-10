import streamlit as st
import pandas as pd
import json
import os
import datetime

st.set_page_config(page_title="AI 投資戰情室", layout="wide", page_icon="📈")
st.title("📈 Jonathan's AI Investment Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(["🦅 禿鷹 (經典版)", "🚀 超級禿鷹 (進化版)", "🤖 實驗室", "✍️ 手動日記"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

# 定義共用的年份選項
PERIOD_OPTIONS = {
    "2025-Now (當前)": "2025_now",
    "2024 (AI 牛市)": "2024_bull",
    "2023 (震盪復甦)": "2023_recovery",
    "2022 (崩盤熊市)": "2022_bear"
}

# ==========================================
# 共用顯示函數 (減少重複代碼)
# ==========================================
def render_strategy_view(strategy_prefix, strategy_title, strategy_desc):
    st.header(strategy_title)
    st.caption(strategy_desc)
    
    # 年份選擇器
    col_sel, col_dummy = st.columns([1, 3])
    with col_sel:
        # 使用 unique key 避免兩個 tabs 的 selectbox 衝突
        selected_label = st.selectbox(
            "📅 選擇回測年份：",
            list(PERIOD_OPTIONS.keys()),
            key=f"sel_{strategy_prefix}"
        )
    
    period_key = PERIOD_OPTIONS[selected_label]
    b_file = os.path.join(DATA_DIR, f"{strategy_prefix}_{period_key}_balance.csv")
    l_file = os.path.join(DATA_DIR, f"{strategy_prefix}_{period_key}_log.csv")
    
    # 顯示資產曲線
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
            
            st.line_chart(df['Equity'])
            
            if roi < -20: st.error("⚠️ 警告：此策略在該年份遭受重創。")
            elif roi > 20: st.success("✅ 表現優異！")
        else:
            st.warning("數據為空。")
    else:
        st.info(f"找不到數據檔案：{b_file}")

    # 顯示交易紀錄
    if os.path.exists(l_file):
        df_log = pd.read_csv(l_file)
        if not df_log.empty:
            with st.expander(f"📜 查看 {selected_label} 詳細交易紀錄"):
                st.dataframe(
                    df_log.sort_index(ascending=False),
                    use_container_width=True,
                    column_config={"Price": st.column_config.NumberColumn(format="$%.2f")}
                )
        else:
            st.info("無交易紀錄。")

# ==========================================
# Tab 1: 經典禿鷹
# ==========================================
with tab1:
    render_strategy_view(
        "vulture",
        "🦅 Vulture Strategy (經典 All-in)",
        "規則：固定 20% 止盈 | 15% 止損 | 15 天持有上限 (看看它能否撐過 2022)"
    )

# ==========================================
# Tab 2: 超級禿鷹
# ==========================================
with tab2:
    render_strategy_view(
        "super_vulture",
        "🚀 Super Vulture (動態追蹤)",
        "規則：不止盈(讓獲利奔跑) | 高點回吐 5% 離場 | 10% 嚴格止損"
    )

# ==========================================
# Tab 3 & 4 (保持不變)
# ==========================================
with tab3:
    st.header("🤖 Alpha 實驗室")
    st.info("開發中...")

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
