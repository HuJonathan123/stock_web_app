from ib_insync import *

print("⚠️ 準備連線至【真實帳戶】...")
ib = IB()

try:
    ib.connect('127.0.0.1', 4001, clientId=99)
    account = ib.managedAccounts()[0]
    print(f"✅ 連線成功！你的真實帳戶是: {account}")

    print("\n準備下單：買入 1 股 Ford (F) 完整股測試...")
    
    # 建立福特汽車的合約
    contract = Stock('F', 'SMART', 'USD')
    ib.qualifyContracts(contract)

    # 🔥 關鍵修改：不使用 cashQty，直接買 1 股完整的 (totalQuantity=1)
    order = MarketOrder('BUY', 1)

    # 送出訂單
    trade = ib.placeOrder(contract, order)
    
    ib.sleep(2)
    print("🚀 訂單已發送至交易所！")
    print(f"訂單狀態: {trade.orderStatus.status}")
    
    if trade.orderStatus.status == 'PendingSubmit':
        print("💡 狀態為 PendingSubmit：目前美股休市中。訂單已掛單排隊，將在今晚開盤時自動成交！")
    elif trade.orderStatus.status == 'Filled':
        print("🎉 狀態為 Filled：訂單已瞬間成交！")

except Exception as e:
    print(f"❌ 發生錯誤: {e}")
finally:
    ib.disconnect()
    print("🔌 連線已斷開。")