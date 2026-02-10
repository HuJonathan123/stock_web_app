import streamlit as st
import pandas as pd
import json
import os
import datetime

# ==========================================
# 1. 頁面設定
# ==========================================
st.set_page_config(page_title="AI 投資戰情室", layout="wide", page_icon="📈")
st.title("📈 Jonathan's AI Investment Dashboard")

# 建立分頁
tab1, tab2, tab3, tab4 = st.tabs(["🦅 禿鷹 (經典版)", "🚀 超級禿鷹 (壓力測試)", "🤖 實驗室", "✍️ 手動日記"])

# 路徑設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

# ==========================================
# Tab 1: 禿鷹策略 (經典版 - 你的獲利保證)
# ==========================================
with tab1:
    st.header("🦅 Vulture Strategy (經典 All-in)")
    st.caption("✅ 你的基準策略 | 規則：固定 20% 止盈 | 15% 止損 | 15 天持有上限")
    
    # 讀取檔案
    p_file = os.path.join(DATA_DIR, "vulture_portfolio.json")
    b_file = os.path.join(DATA_DIR, "vulture_balance.csv")
    l_file = os.path.join(DATA_DIR, "vulture_log.csv")
    
    # A. 顯示持倉卡片
    if os.path.exists(p_file):
        try:
            with open(p_file, 'r') as f: port = json.load(f)
            c1, c2, c3 = st.columns(3)
            
            holdings = port.get('holdings', [])
            status = f"{holdings[0]['Ticker']} ({holdings[0]['Shares']:.2f} 股)" if holdings else "空手 (100% 現金)"
            
            c1.metric("當前持倉", status)
            c2.metric("可用現金", f"${port['cash']:.2f}")
            c3.metric("最後更新", port.get('last_update', 'N/A'))
        except: pass

    # B. 顯示曲線圖
    if os.path.exists(b_file):
        df = pd.read_csv(b_file)
        if not df.empty:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            
            last_equity = df.iloc[-1]['Equity']
            roi = (last_equity - 1000) / 1000 * 100
            st.markdown(f"### 目前淨值: **${last_equity:,.2f}** (:green[{roi:.2f}%])")
            st.line_chart(df['Equity'])

    # C. 顯示交易紀錄
    if os.path.exists(l_file):
        df_log = pd.read_csv(l_file)
        if not df_log.empty:
            st.dataframe(df_log.sort_index(ascending=False), use_container_width=True)

# ==========================================
# Tab 2: 超級禿鷹 (壓力測試 - 多重宇宙)
# ==========================================
with tab2:
    st.header("🚀 Super Vulture (穿越牛熊壓力測試)")
    st.caption("🧪 實驗規則：不止盈(讓獲利奔跑) | 高點回吐 5% 離場 | 10% 嚴格止損")
    
    # 年份選擇器
    col_sel, col_dummy = st.columns([1, 3])
    with col_sel:
        period_options = {
            "2025-Now (當前)": "2025_now",
            "2024 (AI 牛市)": "2024_bull",
            "2023 (震盪復甦)": "2023_recovery",
            "2022 (崩盤熊市)": "2022_bear"
        }
        selected_label = st.selectbox("📅 選擇回測年份：", list(period_options.keys()))
    
    period_key = period_options[selected_label]
    
    # 動態組裝檔名
    b_file = os.path.join(DATA_DIR, f"super_vulture_{period_key}_balance.csv")
    l_file = os.path.join(DATA_DIR, f"super_vulture_{period_key}_log.csv")
    
    # 顯示分析結果
    if os.path.exists(b_file):
        df = pd.read_csv(b_file)
        if not df.empty:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            
            final_eq = df.iloc[-1]['Equity']
            roi = (final_eq - 1000) / 1000 * 100
            
            # 根據賺賠變色
            color = "green" if roi >= 0 else "red"
            emoji = "🎉" if roi >= 0 else "🩸"
            
            st.subheader(f"📈 {selected_label} 資產走勢")
            c1, c2 = st.columns(2)
            c1.markdown(f"## 最終淨值: **${final_eq:,.2f}**")
            c2.markdown(f"## 報酬率: :{color}[{emoji} {roi:.2f}%]")
            
            st.line_chart(df['Equity'])
            
            # 熊市警語
            if roi < -20:
                st.error("⚠️ 警告：此策略在該年份遭受重創，不適合空頭市場。")
            elif roi > 20:
                st.success("✅ 完美：此策略在該年份表現優異！")

    if os.path.exists(l_file):
        with st.expander(f"📜 查看 {selected_label} 詳細交易紀錄"):
            df_log = pd.read_csv(l_file)
            st.dataframe(df_log.sort_index(ascending=False), use_container_width=True)

# ==========================================
# Tab 3 & 4: 其他功能
# ==========================================
with tab3:
    st.header("🤖 Alpha 實驗室")
    st.info("開發中：未來可加入 VIX 情緒指標或 Transformer 預測模型。")

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
