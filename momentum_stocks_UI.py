import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import urllib.parse
from datetime import datetime

# List of Nifty 50 stock tickers (as of current date)
nifty_50_tickers = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "SBIN.NS",
    "KOTAKBANK.NS", "BAJFINANCE.NS", "BHARTIARTL.NS", "ITC.NS", "HCLTECH.NS", "LT.NS", "ASIANPAINT.NS",
    "AXISBANK.NS", "HDFCLIFE.NS", "MARUTI.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS", "WIPRO.NS", "NESTLEIND.NS",
    "TITAN.NS", "BAJAJFINSV.NS", "ADANIGREEN.NS", "GRASIM.NS", "DIVISLAB.NS", "JSWSTEEL.NS", "TECHM.NS",
    "TATAMOTORS.NS", "TATACONSUM.NS", "BPCL.NS", "SHREECEM.NS", "ONGC.NS", "HEROMOTOCO.NS", "HINDALCO.NS",
    "COALINDIA.NS", "DRREDDY.NS", "POWERGRID.NS", "IOC.NS", "EICHERMOT.NS", "BAJAJ-AUTO.NS", "NTPC.NS",
    "TATASTEEL.NS", "BRITANNIA.NS", "ADANIPORTS.NS", "M&M.NS", "CIPLA.NS", "UPL.NS", "SBILIFE.NS",
    "HINDZINC.NS"
]

# Function to get stock data
def get_stock_data(ticker, start, end):
    stock_data = yf.download(ticker, start=start, end=end)
    return stock_data

# Function to calculate momentum indicators
def calculate_indicators(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['RSI'] = compute_rsi(df['Close'], 14)
    df = compute_macd(df)
    df = compute_supertrend(df)
    return df

# Function to compute RSI
def compute_rsi(series, period):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to compute MACD
def compute_macd(df):
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

# Function to compute SuperTrend
def compute_supertrend(df, atr_period=10, multiplier=3):
    hl2 = (df['High'] + df['Low']) / 2
    df['ATR'] = hl2.rolling(window=atr_period).apply(lambda x: np.mean(np.abs(np.diff(x))), raw=False)
    df['UpperBand'] = hl2 + (multiplier * df['ATR'])
    df['LowerBand'] = hl2 - (multiplier * df['ATR'])
    df['SuperTrend'] = 0.0

    for i in range(1, len(df)):
        if df['Close'].iloc[i-1] <= df['UpperBand'].iloc[i-1]:
            df.loc[df.index[i], 'UpperBand'] = min(df['UpperBand'].iloc[i], df['UpperBand'].iloc[i-1])
        if df['Close'].iloc[i-1] >= df['LowerBand'].iloc[i-1]:
            df.loc[df.index[i], 'LowerBand'] = max(df['LowerBand'].iloc[i], df['LowerBand'].iloc[i-1])
        
        if df['Close'].iloc[i] > df['UpperBand'].iloc[i-1]:
            df.loc[df.index[i], 'SuperTrend'] = df['LowerBand'].iloc[i]
        elif df['Close'].iloc[i] < df['LowerBand'].iloc[i-1]:
            df.loc[df.index[i], 'SuperTrend'] = df['UpperBand'].iloc[i]
        else:
            df.loc[df.index[i], 'SuperTrend'] = df['SuperTrend'].iloc[i-1]
    return df

# Function to identify momentum stocks
def identify_momentum_stocks(tickers, start, end, target_percentage=10):
    momentum_stocks = []
    for ticker in tickers:
        df = get_stock_data(ticker, start, end)
        if df.empty:
            continue
        df = calculate_indicators(df)
        
        # Conditions to identify momentum stocks
        if (df['SMA_50'].iloc[-1] > df['SMA_200'].iloc[-1] and 
            df['RSI'].iloc[-1] > 70 and 
            df['MACD'].iloc[-1] > df['Signal_Line'].iloc[-1] and 
            df['Close'].iloc[-1] > df['SuperTrend'].iloc[-1]):
            buy_price = df['Close'].iloc[-1]
            target_price = buy_price * (1 + target_percentage / 100)
            momentum_stocks.append({
                'Ticker': ticker,
                'Buy Price': buy_price,
                'Target Price': target_price,
                'Last Close': df['Close'].iloc[-1],
                'Signal': 'Buy'
            })
        
        # Sell signal based on SuperTrend
        elif df['Close'].iloc[-1] < df['SuperTrend'].iloc[-1]:
            momentum_stocks.append({
                'Ticker': ticker,
                'Last Close': df['Close'].iloc[-1],
                'Signal': 'Sell'
            })
    
    return momentum_stocks

# Streamlit UI
st.title("Nifty 50 Momentum Stocks")
start_date = st.date_input("Start Date", value=pd.to_datetime('2024-01-01'))
end_date = st.date_input("End Date", value=datetime.today().date())
target_percentage = st.slider("Target Percentage", min_value=5, max_value=20, value=10)
selected_stocks = st.multiselect("Select Stocks", nifty_50_tickers, default=nifty_50_tickers)

if st.button("Identify Momentum Stocks"):
    with st.spinner("Fetching data and calculating indicators..."):
        momentum_stocks = identify_momentum_stocks(selected_stocks, start_date, end_date, target_percentage)
    
    st.success("Momentum Stocks Identified!")
    st.write(f"Target Percentage: {target_percentage}%")
    
    if momentum_stocks:
        share_message = "Momentum Stocks:\n"
        for stock in momentum_stocks:
            if stock['Signal'] == 'Buy':
                st.markdown(f"""
                <div style="border:2px solid green; padding: 10px; margin: 10px;">
                **Ticker**: {stock['Ticker']}<br>
                **Buy Price**: {stock['Buy Price']:.2f}<br>
                **Target Price**: {stock['Target Price']:.2f}<br>
                **Last Close**: {stock['Last Close']:.2f}<br>
                **Signal**: <span style="color:green;">{stock['Signal']}</span>
                </div>
                """, unsafe_allow_html=True)
                share_message += f"{stock['Ticker']} - Buy at {stock['Buy Price']:.2f}, Target {stock['Target Price']:.2f}\n"
            else:
                st.markdown(f"""
                <div style="border:2px solid red; padding: 10px; margin: 10px;">
                **Ticker**: {stock['Ticker']}<br>
                **Last Close**: {stock['Last Close']:.2f}<br>
                **Signal**: <span style="color:red;">{stock['Signal']}</span>
                </div>
                """, unsafe_allow_html=True)
                share_message += f"{stock['Ticker']} - Sell at {stock['Last Close']:.2f}\n"
        
        # Create WhatsApp share link
        encoded_message = urllib.parse.quote(share_message)
        whatsapp_url = f"https://api.whatsapp.com/send?text={encoded_message}"
        
        # Display WhatsApp share button
        st.markdown(f"[Share on WhatsApp]({whatsapp_url})")
    else:
        st.write("No momentum stocks found for the given stocks")
