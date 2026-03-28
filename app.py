import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from groq import Groq
import os

# ---------------- CONFIG ----------------
st.set_page_config(page_title="InvestorSignal AI", layout="wide")

# ---------------- HEADER ----------------
st.title("🛎️ InvestorSignal AI")
st.caption("🚀 Smart Opportunity Radar for Indian Investors | ET Hackathon 2026")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Controls")

symbol = st.sidebar.text_input("Stock Symbol", "RELIANCE.NS")
period = st.sidebar.selectbox("Time Period", ["3mo", "6mo", "1y"], index=1)

# Portfolio Simulator
st.sidebar.subheader("💼 Investment Simulator")
investment = st.sidebar.number_input("Monthly Investment ₹", 1000, 100000, 5000)
years = st.sidebar.slider("Years", 1, 20, 5)

# ---------------- DATA ----------------
@st.cache_data(ttl=300)
def get_stock_data(symbol, period):
    try:
        return yf.download(symbol, period=period, progress=False)
    except:
        return pd.DataFrame()

data = get_stock_data(symbol, period)

# ---------------- MAIN ----------------
if not data.empty and len(data) > 5:

    df = data.copy()
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df = df.dropna()

    current_price = round(float(df['Close'].iloc[-1]), 2)

    try:
        change = round(((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100, 2)
    except:
        change = 0

    # ---------------- METRICS ----------------
    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Current Price", f"₹{current_price}")
    col2.metric("📊 Change", f"{change}%", delta=f"{change}%")
    col3.metric("📅 Period", period)

    # ---------------- CALCULATIONS ----------------

    # Moving Average
    df['MA20'] = df['Close'].rolling(20).mean()

    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Support & Resistance
    support = df['Low'].rolling(20).min().iloc[-1]
    resistance = df['High'].rolling(20).max().iloc[-1]

    # Volume
    avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
    latest_volume = df['Volume'].iloc[-1]

    # ---------------- TABS ----------------
    tab1, tab2, tab3 = st.tabs(["📈 Chart", "🤖 AI Advisor", "🚨 Signals"])

    # ================== TAB 1 ==================
    with tab1:
        st.subheader("📊 Trading Chart")

        fig = go.Figure()

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Price"
        ))

        # Moving Average
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['MA20'],
            mode='lines',
            name='MA20',
        ))

        fig.update_layout(
            height=600,
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            title=f"{symbol} Trading Chart"
        )

        st.plotly_chart(fig, use_container_width=True)

    # ================== TAB 2 ==================
    with tab2:
        st.subheader("🤖 AI Financial Advisor")

        user_query = st.text_input("Ask anything (e.g., Should I buy now?)")

        groq_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

        if st.button("Ask AI"):

            if groq_key and groq_key.startswith("gsk_"):
                try:
                    client = Groq(api_key=groq_key)

                    prompt = f"""
You are an Indian stock advisor.

Stock: {symbol}
Price: ₹{current_price}

Answer in simple Hinglish:
{user_query}
"""

                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        max_tokens=300
                    )

                    st.success(response.choices[0].message.content)

                except Exception as e:
                    st.error(f"AI Error: {str(e)}")

            else:
                # Demo AI
                if "buy" in user_query.lower():
                    st.success("📈 Market stable lag raha hai — SIP ya gradual buy better rahega.")
                elif "sell" in user_query.lower():
                    st.warning("⚠️ Profit hai to partial booking consider karo.")
                else:
                    st.info("💡 Long-term investing is safest strategy.")

    # ================== TAB 3 ==================
    with tab3:
        st.subheader("🚨 Smart Signals")

        # RSI
        latest_rsi = df['RSI'].iloc[-1]
        st.write(f"📊 RSI: {round(latest_rsi,2)}")

        if latest_rsi > 70:
            st.error("🔴 Overbought — Sell Zone")
        elif latest_rsi < 30:
            st.success("🟢 Oversold — Buy Opportunity")
        else:
            st.info("🟡 Neutral RSI")

        # Support Resistance
        st.write(f"🟢 Support: ₹{round(support,2)}")
        st.write(f"🔴 Resistance: ₹{round(resistance,2)}")

        # Breakout
        if current_price > resistance * 0.985:
            st.success("🔥 Breakout Detected")
        elif current_price < support * 1.02:
            st.error("📉 Breakdown Risk")

        # Volume Spike
        if latest_volume > avg_volume * 1.5:
            st.warning("⚡ Volume Spike — Big Move Possible")

        # Sentiment
        price_change = df['Close'].iloc[-1] - df['Close'].iloc[-5]

        st.subheader("📰 Market Sentiment")

        if price_change > 0:
            st.success("🟢 Positive Trend")
        else:
            st.error("🔴 Negative Trend")

    # ---------------- PORTFOLIO ----------------
    st.subheader("💼 Investment Growth")

    rate = 12 / 100 / 12
    months = years * 12

    future_value = investment * (((1 + rate)**months - 1) / rate) * (1 + rate)

    st.success(f"📈 Estimated Value after {years} years: ₹{round(future_value,0)}")

# ---------------- ERROR ----------------
else:
    st.error("❌ Could not fetch data. Try RELIANCE.NS, TCS.NS")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built for ET AI Hackathon 2026 🚀 | AI + Finance Intelligence")