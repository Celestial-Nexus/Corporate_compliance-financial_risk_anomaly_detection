import yfinance as yf

ticker = yf.Ticker("AAPL")

info = ticker.info
print(f"Company Name: {info.get('shortName')}")
print(f"Sector: {info.get('sector')}")
print(f"Market Cap: ${info.get('marketCap'):,}\n")

historical_data = ticker.history(period="1mo")

print("Recent Historical Data:")
print(historical_data[['Open', 'High', 'Low', 'Close', 'Volume']].tail())