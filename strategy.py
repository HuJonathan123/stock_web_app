import yfinance as yf
import pandas as pd

def vulture_strategy_check(ticker, current_cash, holding_info):
    """
    檢查單隻股票是否觸發買賣訊號
    holding_info: None (空手) 或 {'Entry': 價格, 'Shares': 股數, 'Ticker': 代號}
    回傳: (Action, Detail)
    """
    # 下載數據 (只取最近 30 天即可計算 RSI)
    df = yf.download(ticker, period="2mo", progress=False)
    if len(df) < 20: return "HOLD", None
    
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # 計算指標
    close = df['Close'].iloc[-1]
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain/loss))
    current_rsi = rsi.iloc[-1]
    
    ma20 = df['Close'].rolling(20).mean()
    std = df['Close'].rolling(20).std()
    lower_band = ma20 - (2 * std)
    current_lower = lower_band.iloc[-1]
    
    # --- 判斷邏輯 ---
    
    # 1. 如果持有中 -> 檢查賣出
    if holding_info:
        entry_price = holding_info['Entry']
        pnl = (close - entry_price) / entry_price
        
        if pnl > 0.20: return "SELL", f"💰 獲利達標 (+{pnl*100:.1f}%)"
        if current_rsi > 75: return "SELL", f"🔥 RSI 過熱 ({current_rsi:.1f})"
        if pnl < -0.15: return "SELL", f"💀 止損 (-15%)"
        
        return "HOLD", f"持倉中 (PnL: {pnl*100:.1f}%)"

    # 2. 如果空手 -> 檢查買入
    else:
        # 禿鷹條件: RSI < 30 且 跌破布林下軌
        if current_rsi < 30 and close < current_lower:
            return "BUY", f"撿屍體 (RSI: {current_rsi:.1f})"
            
    return "WAIT", f"觀望中 (RSI: {current_rsi:.1f})"