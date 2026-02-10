import streamlit as st
import pandas as pd
import json
import os
import datetime

# ==========================================
# 1. 頁面基礎設定
# ==========================================
st.set_page_config(page_title="AI 投資戰情室", layout="wide", page_icon="📈")
st.title("📈 Jonathan's AI Investment Dashboard")

# 建立四個分頁
tab1, tab2, tab3, tab4 = st.tabs(["🦅 禿鷹 (經典版)", "🚀 超級禿鷹 (動態止盈)", "🤖 實驗室", "✍️ 手動日記"])
# ==========================================
# 2. 路徑設定 (使用絕對路徑防止雲端錯誤)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# 強制建立 data 資料夾 (如果不存在)
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# ==========================================
# 3. 核心函數：顯示策略頁面
# ==========================================
def show_strategy_tab(strategy_name, title_text):
    """
    通用函數：讀取並顯示某個策略的數據
    strategy_name: 'vulture' 或 'octopus' (對應檔案前綴)
    title_text: 顯示在網頁上的標題
    """
    # 定義檔案路徑
    portfolio_file = os.path.join(DATA_DIR, f"{strategy_name}_portfolio.json")
    log_file = os.path.join(DATA_DIR, f"{strategy_name}_log.csv")
    balance_file = os.path.join(DATA_DIR, f"{strategy_name}_balance.csv")

    st.header(title_text)
    
    # --- 讀取 Portfolio (持倉狀態) ---
    portfolio = {"cash": 1000, "holdings": [], "last_update": "尚未初始化"}
    if os.path.exists(portfolio_file):
        try:
            with open(portfolio_file, 'r') as f:
                if os.stat(portfolio_file).st_size > 0:
                    portfolio = json.load(f)
        except json.JSONDecodeError:
            st.warning(f"⚠️ {strategy_name}_portfolio.json 格式錯誤，使用預設值。")

    # --- 顯示上方三張卡片 (Metrics) ---
    col1, col2, col3 = st.columns(3)
    
    # 處理持倉顯示文字
    holdings = portfolio.get('holdings', [])
    status_text = "無 (空手)"
    
    if isinstance(holdings, list) and len(holdings) > 0:
        # 如果是列表 (章魚策略)
        tickers = [h['Ticker'] for h in holdings]
        status_text = f"持有 {len(holdings)} 檔 ({', '.join(tickers)})"
    elif isinstance(holdings, dict):
        # 如果是單一字典 (舊版禿鷹策略兼容)
        status_text = f"{holdings['Ticker']} ({holdings.get('Shares', 0):.2f} 股)"
        
    col1.metric("當前持倉", status_text)
    col2.metric("可用現金", f"${portfolio.get('cash', 0):.2f}")
    col3.metric("最後更新", portfolio.get('last_update', 'N/A'))

    # --- 顯示資產曲線圖 ---
    if os.path.exists(balance_file):
        df_bal = pd.read_csv(balance_file)
        if not df_bal.empty:
            st.subheader("📈 資產成長曲線 (含未實現損益)")
            
            # 數據處理
            chart_data = df_bal.copy()
            chart_data['Date'] = pd.to_datetime(chart_data['Date'])
            chart_data = chart_data.set_index('Date')
            
            # 計算報酬率
            latest_val = df_bal.iloc[-1]['Equity']
            roi = (latest_val - 1000) / 1000 * 100
            
            color = "green" if roi >= 0 else "red"
            st.markdown(f"#### 目前總資產淨值: **${latest_val:,.2f}** (:{color}[{roi:.2f}%])")
            
            st.line_chart(chart_data['Equity'])
    else:
        st.info("暫無資產數據 (請在本機執行 run_backtest.py 生成)。")

    # --- 顯示交易紀錄表格 ---
    if os.path.exists(log_file):
        df_log = pd.read_csv(log_file)
        if not df_log.empty:
            st.subheader(f"📜 歷史交易紀錄")
            st.dataframe(
                df_log.sort_index(ascending=False),
                use_container_width=True,
                column_config={
                    "Price": st.column_config.NumberColumn(format="$%.2f"),
                    "Balance": st.column_config.NumberColumn(format="$%.2f"),
                }
            )
    else:
        st.info("暫無交易紀錄。")

# ==========================================
# Tab 1: 禿鷹策略 (All-in)
# ==========================================
with tab1:
    show_strategy_tab("vulture", "🦅 Vulture Strategy (經典 All-in)")
    st.markdown("---")
    st.caption("✅ 40% 報酬率版本 | 規則：固定 20% 止盈 | 15% 止損 | 15 天沒動就換股")

# ==========================================
# Tab 2: 章魚策略 (分散)
# ==========================================
with tab2:
    # 🔥 這裡改成讀取 super_vulture
    show_strategy_tab("super_vulture", "🚀 Super Vulture (動態追蹤)")
    st.markdown("---")
    st.caption("🧪 實驗規則：不止盈(讓獲利奔跑) | 高點回吐 5% 離場 | 10% 嚴格止損")

# ==========================================
# Tab 3: 實驗室 (Placeholder)
# ==========================================
with tab3:
    st.header("🤖 Alpha 實驗室")
    st.info("🚧 開發中：這裡未來可以放置 Transformer 模型預測結果或是情緒分析指標。")
    
    c1, c2 = st.columns(2)
    with c1:
        st.metric("市場情緒 (VIX)", "18.5", "-1.2%")
    with c2:
        st.metric("下週預測", "Bullish", "信心度 75%")

# ==========================================
# Tab 4: 手動日記
# ==========================================
with tab4:
    st.header("✍️ 手動模擬交易紀錄")
    MANUAL_LOG_FILE = os.path.join(DATA_DIR, "manual_log.csv")

    # 確保手動日誌檔案存在
    if not os.path.exists(MANUAL_LOG_FILE):
        pd.DataFrame(columns=["Date", "Ticker", "Action", "Price", "Shares", "Note"]).to_csv(MANUAL_LOG_FILE, index=False)

    # 輸入區塊
    with st.expander("➕ 新增交易紀錄", expanded=True):
        with st.form("manual_entry"):
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
            
            submit_btn = st.form_submit_button("提交紀錄")
            
            if submit_btn:
                if m_ticker and m_price > 0 and m_shares > 0:
                    new_record = {
                        "Date": m_date,
                        "Ticker": m_ticker,
                        "Action": m_action,
                        "Price": m_price,
                        "Shares": m_shares,
                        "Note": m_note
                    }
                    # 讀取舊資料並附加新資料
                    df_old = pd.read_csv(MANUAL_LOG_FILE)
                    df_new = pd.concat([df_old, pd.DataFrame([new_record])], ignore_index=True)
                    df_new.to_csv(MANUAL_LOG_FILE, index=False)
                    st.success(f"已儲存：{m_action} {m_ticker}")
                    st.rerun()
                else:
                    st.error("請填寫完整資訊 (代號、價格、股數)")

    # 顯示紀錄
    if os.path.exists(MANUAL_LOG_FILE):
        df_manual = pd.read_csv(MANUAL_LOG_FILE)
        if not df_manual.empty:
            st.subheader("交易明細")
            st.dataframe(df_manual.sort_index(ascending=False), use_container_width=True)
            
            # 下載按鈕
            csv = df_manual.to_csv(index=False).encode('utf-8')
            st.download_button("下載 CSV", csv, "my_trade_log.csv", "text/csv")
