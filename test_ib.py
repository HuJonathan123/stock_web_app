from ib_insync import *

print("嘗試連線到 IB Gateway...")
ib = IB()

try:
    # 連線到本機 (127.0.0.1)，Port 填入你剛才記下的號碼 (通常是 4002)
    # clientId 可以隨便設一個數字，例如 1
    ib.connect('127.0.0.1', 4002, clientId=1)
    print("✅ 連線成功！")

    # 獲取帳戶資訊
    account = ib.managedAccounts()[0]
    print(f"你的模擬帳戶號碼是: {account}")

    # 查詢帳戶總資產與可用現金
    account_values = ib.accountValues()
    for val in account_values:
        if val.tag == 'NetLiquidation':
            print(f"💰 總資產淨值: {val.value} {val.currency}")
        elif val.tag == 'AvailableFunds':
            print(f"💵 可用資金: {val.value} {val.currency}")

    # ---------- 測試下單 (買入 1 股蘋果股票) ----------
    print("\n準備測試下單：買入 1 股 AAPL...")
    contract = Stock('AAPL', 'SMART', 'USD')
    ib.qualifyContracts(contract) # 確認合約有效

    # 建立一張市價買單
    order = MarketOrder('BUY', 1)

    # 送出訂單
    trade = ib.placeOrder(contract, order)
    print("🚀 訂單已送出！")
    print(f"訂單狀態: {trade.orderStatus.status}")

except Exception as e:
    print(f"❌ 發生錯誤: {e}")
finally:
    # 記得斷開連線
    ib.disconnect()
    print("連線已斷開。")