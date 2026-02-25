import streamlit as st
import pandas as pd
import json
import os
import datetime
import subprocess

st.set_page_config(page_title="AI 投資戰情室", layout="wide", page_icon="📈")
st.title("📈 Jonathan's AI Investment Dashboard")

# 重新定義 Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🦅 禿鷹 (經典版)", 
    "🚀 超級禿鷹 (進化版)", 
    "🏆 AI 領頭羊 (Top 3)", 
    "💥 AI MA30 突破", 
    "✍️ 手動日記"
])

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

# 共用 Dataframe 顯示格式
DF_CONFIG = {
    "ticker": st.column_config.TextColumn("股票代號"),
    "price": st.column_config.NumberColumn("現價", format="$%.2f"),
    "probability": st.column_config.ProgressColumn("AI 勝率", format="%.1f%%", min_value=0, max_value=100),
    "ma30_distance": st.column_config.NumberColumn("乖離 MA30", format="%+.1f%%")
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
            # AI 策略的初始資金是 10000，其他是 1000，這裡做個簡單判斷
            init_cash = 10000 if "ai" in strategy_prefix else 1000
            roi = (final_eq - init_cash) / init_cash * 100
            
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
# 載入最新的 AI 掃描結果
# ==========================================
ai_signals = {}
market_status = None
scan_time = "尚未掃描"

LATEST_SIGNALS_FILE = os.path.join(DATA_DIR, "latest_signals.json")
if os.path.exists(LATEST_SIGNALS_FILE):
    with open(LATEST_SIGNALS_FILE, 'r') as f:
        ai_signals = json.load(f)
        market_status = ai_signals.get('market_bullish')
        scan_time = ai_signals.get('scan_time', 'N/A')

def display_market_status():
    st.write(f"📅 **最後掃描時間:** `{scan_time}`")
    if market_status is True:
        st.success("🟢 **大盤狀態：多頭 (QQQ > EMA60)** - 大環境安全，可積極建倉！")
    elif market_status is False:
        st.error("🔴 **大盤狀態：空頭 (QQQ < EMA60)** - 系統性風險高，建議空手或極輕倉防守！")
    else:
        st.warning("⚪️ **大盤狀態：未知** - 請先執行市場掃描。")

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
# Tab 3: AI 領頭羊戰法 (策略 1)
# ==========================================
with tab3:
    st.header("🏆 AI 領頭羊戰法 (Top 3 Momentum)")
    st.caption("【核心邏輯】只買全市場動能最強的前三名，並由 AI 確認勝率。無限奔跑不設止盈。")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("⚡️ 執行掃描", key="btn_scan_1"):
            with st.spinner("正在掃描市場..."):
                subprocess.run(["python", "ai_market_scanner.py"])
            st.rerun()
            
    display_market_status()
    st.divider()
    
    st.subheader("🎯 今日推薦清單")
    s1_data = ai_signals.get('strategy_1_top3', [])
    if s1_data:
        st.dataframe(pd.DataFrame(s1_data), use_container_width=True, column_config=DF_CONFIG, hide_index=True)
    else:
        st.info("今日無符合條件的標的，AI 建議觀望。")
        
    st.divider()
    # 這裡預設讀取 ai_backtest_2.py 產生的結果
    # 注意：你需要確保你把 ai_backtest_2.py 產生的 csv 命名規則跟你的網頁一致
    # 這裡為了展示，我直接寫死讀取 data/ai_backtest_balance.csv
    st.subheader("📜 歷史回測績效 (Top 3 動能版)")
    bt_bal_file = os.path.join(DATA_DIR, "ai_backtest_balance.csv")
    bt_log_file = os.path.join(DATA_DIR, "ai_backtest_log.csv")
    
    if os.path.exists(bt_bal_file):
        df_bal = pd.read_csv(bt_bal_file)
        if not df_bal.empty:
            df_bal['Date'] = pd.to_datetime(df_bal['Date'])
            df_bal = df_bal.set_index('Date')
            final_eq = df_bal.iloc[-1]['Equity']
            roi = (final_eq - 10000) / 10000 * 100
            
            c1, c2 = st.columns(2)
            c1.metric("回測總資產", f"${final_eq:,.2f}")
            c2.metric("總報酬率", f"{roi:.1f}%")
            st.line_chart(df_bal['Equity'])
            
            with st.expander("查看詳細交易紀錄"):
                if os.path.exists(bt_log_file):
                    st.dataframe(pd.read_csv(bt_log_file).sort_index(ascending=False), use_container_width=True)


# ==========================================
# Tab 4: AI MA30 突破戰法 (策略 2)
# ==========================================
with tab4:
    st.header("💥 AI MA30 強力突破戰法")
    st.caption("【核心邏輯】等待股價強勢突破 MA30 上方 5% 才進場，確認度高，動態收網止盈。")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("⚡️ 執行掃描", key="btn_scan_2"):
            with st.spinner("正在掃描市場..."):
                subprocess.run(["python", "ai_market_scanner.py"])
            st.rerun()
            
    display_market_status()
    st.divider()
    
    st.subheader("🎯 今日推薦清單")
    s2_data = ai_signals.get('strategy_2_ma30', [])
    if s2_data:
        st.dataframe(pd.DataFrame(s2_data), use_container_width=True, column_config=DF_CONFIG, hide_index=True)
    else:
        st.info("今日無符合條件的標的，尚未出現突破。")
        
    st.divider()
    # 這裡你需要確保執行 ai_backtest_ma30_2.py 後，輸出的 CSV 檔名是什麼
    # 假設你把它命名為 ai_backtest_ma30_balance.csv
    st.subheader("📜 歷史回測績效 (MA30 突破版)")
    bt_bal_file_ma30 = os.path.join(DATA_DIR, "ai_backtest_ma30_balance.csv") # 假設你的檔名
    bt_log_file_ma30 = os.path.join(DATA_DIR, "ai_backtest_ma30_log.csv")     # 假設你的檔名
    
    if os.path.exists(bt_bal_file_ma30):
        df_bal = pd.read_csv(bt_bal_file_ma30)
        if not df_bal.empty:
            df_bal['Date'] = pd.to_datetime(df_bal['Date'])
            df_bal = df_bal.set_index('Date')
            final_eq = df_bal.iloc[-1]['Equity']
            roi = (final_eq - 10000) / 10000 * 100
            
            c1, c2 = st.columns(2)
            c1.metric("回測總資產", f"${final_eq:,.2f}")
            c2.metric("總報酬率", f"{roi:.1f}%")
            st.line_chart(df_bal['Equity'])
            
            with st.expander("查看詳細交易紀錄"):
                if os.path.exists(bt_log_file_ma30):
                    st.dataframe(pd.read_csv(bt_log_file_ma30).sort_index(ascending=False), use_container_width=True)
    else:
        st.info(f"尚未找到 MA30 版本的歷史績效檔案 ({bt_bal_file_ma30})。請先執行回測程式並將結果輸出為此檔名。")

# ==========================================
# Tab 5: 手動日記
# ==========================================
with tab5:
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