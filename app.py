import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

# ---------------- CONFIG ----------------
st.set_page_config(page_title="InvestorSignal AI", page_icon="🛎️", layout="wide")

PERIOD_DAYS = {"3mo": 63, "6mo": 126, "1y": 252, "2y": 504}
MIN_ROWS = 60  # needed for MA50 / MACD to be meaningful

WATCHLIST = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS", "AXISBANK.NS",
    "KOTAKBANK.NS", "HINDUNILVR.NS", "MARUTI.NS", "SUNPHARMA.NS", "TATAMOTORS.NS",
    "WIPRO.NS", "ADANIENT.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "TITAN.NS",
]

KIND_RENDER = {"good": st.success, "bad": st.error, "warn": st.warning, "info": st.info}


# ---------------- HELPERS ----------------
def normalize_symbol(raw: str) -> str:
    """'reliance' -> 'RELIANCE.NS'. Leaves '.BO', '^NSEI', 'AAPL' style symbols alone if they have a suffix/prefix."""
    s = raw.strip().upper().replace(" ", "")
    if not s:
        return s
    if "." not in s and not s.startswith("^"):
        s += ".NS"
    return s


def inr(x: float) -> str:
    """Format a number with Indian digit grouping: 1234567 -> ₹12,34,567."""
    n = int(round(x))
    s = str(abs(n))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts + [tail])
    return f"{'-' if n < 0 else ''}₹{s}"


def clean_ohlcv(df) -> pd.DataFrame:
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if not all(c in df.columns for c in needed):
        return pd.DataFrame()
    return df[needed].dropna()


# ---------------- DATA ----------------
@st.cache_data(ttl=300, show_spinner=False)
def get_stock_data(symbol: str) -> pd.DataFrame:
    """Always fetch 2y so MA50/MACD are warmed up; the UI slices to the chosen period."""
    try:
        return clean_ohlcv(yf.download(symbol, period="2y", progress=False, auto_adjust=True))
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def get_news(symbol: str) -> list:
    try:
        items = yf.Ticker(symbol).news or []
    except Exception:
        return []
    out = []
    for it in items[:6]:
        content = it.get("content") if isinstance(it.get("content"), dict) else {}
        title = it.get("title") or content.get("title")
        url = it.get("link") or (content.get("canonicalUrl") or {}).get("url")
        if title:
            out.append((title, url))
    return out


# ---------------- INDICATORS ----------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]

    df["MA20"] = close.rolling(20).mean()
    df["MA50"] = close.rolling(50).mean()

    std20 = close.rolling(20).std()
    df["BB_UP"] = df["MA20"] + 2 * std20
    df["BB_LO"] = df["MA20"] - 2 * std20

    # RSI with Wilder's smoothing (standard definition, matches charting platforms)
    delta = close.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.where(avg_loss != 0, 100.0).where(avg_gain.notna())
    df["RSI"] = rsi

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIG"] = df["MACD"].ewm(span=9, adjust=False).mean()
    return df


def analyze(df: pd.DataFrame) -> dict:
    """Turn an indicator frame into levels, a list of signals and an overall score."""
    last, prev = df.iloc[-1], df.iloc[-2]
    window = df.iloc[-21:-1]  # previous 20 sessions, excluding today, so breakouts are real

    price = float(last["Close"])
    support = float(window["Low"].min())
    resistance = float(window["High"].max())
    avg_vol = float(window["Volume"].mean())
    vol_ratio = float(last["Volume"] / avg_vol) if avg_vol > 0 else float("nan")

    info = {
        "price": price,
        "change": float((price / float(prev["Close"]) - 1) * 100),
        "rsi": float(last["RSI"]),
        "ma20": float(last["MA20"]),
        "ma50": float(last["MA50"]),
        "macd": float(last["MACD"]),
        "macd_sig": float(last["MACD_SIG"]),
        "support": support,
        "resistance": resistance,
        "vol_ratio": vol_ratio,
        "high_52w": float(df["High"].tail(252).max()),
        "low_52w": float(df["Low"].tail(252).min()),
    }

    signals, score = [], 0

    rsi = info["rsi"]
    if pd.notna(rsi):
        if rsi > 70:
            signals.append(("bad", f"RSI {rsi:.1f} — overbought, pullback risk"))
            score -= 1
        elif rsi < 30:
            signals.append(("good", f"RSI {rsi:.1f} — oversold, possible bounce"))
            score += 1
        else:
            signals.append(("info", f"RSI {rsi:.1f} — neutral zone"))

    if pd.notna(info["ma20"]) and pd.notna(info["ma50"]):
        if price > info["ma50"] and info["ma20"] > info["ma50"]:
            signals.append(("good", "Uptrend — price above MA50 and MA20 above MA50"))
            score += 1
        elif price < info["ma50"] and info["ma20"] < info["ma50"]:
            signals.append(("bad", "Downtrend — price below MA50 and MA20 below MA50"))
            score -= 1
        else:
            signals.append(("info", "Mixed trend — moving averages disagree"))

    if pd.notna(info["macd"]) and pd.notna(info["macd_sig"]):
        if info["macd"] > info["macd_sig"]:
            signals.append(("good", "MACD above signal line — bullish momentum"))
            score += 1
        else:
            signals.append(("bad", "MACD below signal line — bearish momentum"))
            score -= 1

    if price > resistance:
        confirmed = pd.notna(vol_ratio) and vol_ratio >= 1.5
        signals.append(("good", f"Breakout above 20-day high ₹{resistance:,.2f}"
                                + (" with strong volume" if confirmed else " (volume not confirming)")))
        score += 1 if confirmed else 0
    elif price < support:
        signals.append(("bad", f"Breakdown below 20-day low ₹{support:,.2f}"))
        score -= 1
    elif price >= resistance * 0.985:
        signals.append(("warn", "Testing resistance — watch for a breakout or rejection"))
    elif price <= support * 1.02:
        signals.append(("warn", "Near support — a bounce or breakdown is likely soon"))

    if pd.notna(vol_ratio) and vol_ratio >= 1.5:
        signals.append(("warn", f"Volume spike — {vol_ratio:.1f}× the 20-day average"))

    if score >= 2:
        verdict = "🟢 Bullish"
    elif score <= -2:
        verdict = "🔴 Bearish"
    else:
        verdict = "🟡 Neutral"

    info.update(signals=signals, score=score, verdict=verdict)
    return info


# ---------------- CHART ----------------
def build_chart(view: pd.DataFrame, symbol: str, info: dict) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.15, 0.25], vertical_spacing=0.03,
    )
    fig.add_trace(go.Candlestick(
        x=view.index, open=view["Open"], high=view["High"],
        low=view["Low"], close=view["Close"], name="Price",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=view.index, y=view["MA20"], mode="lines", name="MA20",
                             line=dict(width=1.5, color="#f5a623")), row=1, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["MA50"], mode="lines", name="MA50",
                             line=dict(width=1.5, color="#4aa3ff")), row=1, col=1)
    for col, name in (("BB_UP", "Bollinger"), ("BB_LO", "Bollinger")):
        fig.add_trace(go.Scatter(x=view.index, y=view[col], mode="lines", name=name,
                                 legendgroup="bb", showlegend=(col == "BB_UP"),
                                 line=dict(width=1, dash="dot", color="rgba(180,180,180,0.5)")),
                      row=1, col=1)

    fig.add_hline(y=info["support"], line_dash="dash", line_color="#2ecc71",
                  annotation_text="Support", annotation_position="bottom right", row=1, col=1)
    fig.add_hline(y=info["resistance"], line_dash="dash", line_color="#e74c3c",
                  annotation_text="Resistance", annotation_position="top right", row=1, col=1)

    vol_colors = np.where(view["Close"] >= view["Open"], "#2ecc71", "#e74c3c")
    fig.add_trace(go.Bar(x=view.index, y=view["Volume"], marker_color=vol_colors,
                         name="Volume", showlegend=False), row=2, col=1)

    fig.add_trace(go.Scatter(x=view.index, y=view["RSI"], mode="lines", name="RSI",
                             line=dict(color="#b388ff", width=1.5), showlegend=False), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#e74c3c", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#2ecc71", row=3, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)

    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])  # hide weekends
    fig.update_layout(
        height=760, template="plotly_dark", title=f"{symbol} — Price, Volume, RSI",
        xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", y=1.04),
    )
    return fig


# ---------------- AI ----------------
def get_groq_key():
    try:
        key = st.secrets["GROQ_API_KEY"]
    except Exception:  # no secrets.toml or key missing
        key = None
    return key or os.getenv("GROQ_API_KEY")


def build_context(symbol: str, info: dict) -> str:
    sig = "; ".join(t for _, t in info["signals"]) or "none"
    return (
        f"Stock: {symbol}\n"
        f"Price: ₹{info['price']:.2f} ({info['change']:+.2f}% today)\n"
        f"RSI(14): {info['rsi']:.1f} | MA20: {info['ma20']:.2f} | MA50: {info['ma50']:.2f}\n"
        f"MACD: {info['macd']:.2f} vs signal {info['macd_sig']:.2f}\n"
        f"20-day support: {info['support']:.2f} | resistance: {info['resistance']:.2f}\n"
        f"52-week range: {info['low_52w']:.2f} – {info['high_52w']:.2f}\n"
        f"Technical verdict: {info['verdict']} (score {info['score']})\n"
        f"Signals: {sig}"
    )


def ask_ai(symbol: str, info: dict, question: str) -> str:
    key = get_groq_key()
    if key:
        from groq import Groq  # imported lazily so the app still runs without the package/key

        client = Groq(api_key=key)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": (
                    "You are a careful Indian stock market assistant. Reply in simple Hinglish, "
                    "under 150 words. Base your answer ONLY on the technical data provided; do not "
                    "invent news, fundamentals or price targets. Never promise returns. Mention "
                    "risk and that this is educational, not SEBI-registered investment advice.")},
                {"role": "user", "content": f"{build_context(symbol, info)}\n\nQuestion: {question}"},
            ],
            temperature=0.4,
            max_tokens=350,
        )
        return resp.choices[0].message.content

    # Offline fallback: rule-based answer driven by the real signals
    v = info["verdict"]
    q = question.lower()
    if "sell" in q or "book" in q:
        base = ("Technicals weak hain — risk kam karne ke liye partial booking consider karo."
                if info["score"] <= -1 else
                "Trend abhi itna weak nahi hai; stop-loss ke saath hold karna bhi option hai.")
    elif "buy" in q or "invest" in q:
        base = ("Momentum positive hai — gradual entry (SIP / staggered buy) behtar rahegi."
                if info["score"] >= 1 else
                "Abhi clear buy signal nahi hai — wait karo ya chhote hisson me enter karo.")
    else:
        base = "Long-term ke liye diversified SIP sabse safe approach hai."
    return (f"{base}\n\nTechnical view: {v} (score {info['score']}), RSI {info['rsi']:.0f}. "
            f"Support ₹{info['support']:.0f}, resistance ₹{info['resistance']:.0f}.\n\n"
            "_Demo mode: GROQ_API_KEY set karo for full AI answers. Educational only, not advice._")


# ---------------- RADAR ----------------
@st.cache_data(ttl=600, show_spinner=False)
def scan_watchlist(tickers: tuple) -> pd.DataFrame:
    try:
        raw = yf.download(list(tickers), period="1y", group_by="ticker",
                          progress=False, auto_adjust=True, threads=True)
    except Exception:
        return pd.DataFrame()
    rows = []
    for t in tickers:
        try:
            d = clean_ohlcv(raw[t])
        except KeyError:
            continue
        if len(d) < MIN_ROWS:
            continue
        info = analyze(add_indicators(d))
        rows.append({
            "Stock": t.replace(".NS", ""),
            "Price ₹": round(info["price"], 2),
            "Change %": round(info["change"], 2),
            "RSI": round(info["rsi"], 1),
            "Vol ×avg": round(info["vol_ratio"], 2),
            "Score": info["score"],
            "Verdict": info["verdict"],
            "Top signal": next((s for k, s in info["signals"] if k in ("good", "bad", "warn")), "—"),
        })
    return pd.DataFrame(rows)


# ---------------- SIP ----------------
def sip_projection(monthly: float, years: int, annual_return: float) -> pd.DataFrame:
    r = annual_return / 100 / 12
    months = np.arange(1, years * 12 + 1)
    invested = monthly * months
    value = monthly * (((1 + r) ** months - 1) / r) * (1 + r) if r > 0 else invested.astype(float)
    return pd.DataFrame({"Month": months, "Invested": invested, "Value": value})


# ---------------- HEADER ----------------
st.title("🛎️ InvestorSignal AI")
st.caption("🚀 Smart Opportunity Radar for Indian Investors | ET Hackathon 2026")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Controls")
symbol = normalize_symbol(st.sidebar.text_input("Stock Symbol", "RELIANCE.NS",
                                                help="NSE symbols auto-get .NS (e.g. TCS → TCS.NS). Use .BO for BSE."))
period = st.sidebar.selectbox("Chart Period", list(PERIOD_DAYS), index=1)
if st.sidebar.button("🔄 Refresh data"):
    st.cache_data.clear()
    st.rerun()
st.sidebar.info("Educational tool only — not SEBI-registered investment advice.")

# ---------------- DATA ----------------
if not symbol:
    st.warning("Enter a stock symbol in the sidebar.")
    st.stop()

with st.spinner(f"Loading {symbol}…"):
    raw_df = get_stock_data(symbol)

if raw_df.empty or len(raw_df) < MIN_ROWS:
    st.error(f"❌ Could not fetch enough data for **{symbol}** (need ≥ {MIN_ROWS} trading days). "
             "Try RELIANCE.NS, TCS.NS or INFY.NS.")
    st.stop()

df = add_indicators(raw_df)
info = analyze(df)
view = df.tail(PERIOD_DAYS[period])

# ---------------- METRICS ----------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("💰 Current Price", f"₹{info['price']:,.2f}", delta=f"{info['change']:+.2f}%")
m2.metric("🧭 Technical Verdict", info["verdict"], help=f"Score {info['score']} (range −4 to +4)")
m3.metric("📉 52W Low", f"₹{info['low_52w']:,.2f}")
m4.metric("📈 52W High", f"₹{info['high_52w']:,.2f}")

# ---------------- TABS ----------------
tab_chart, tab_signals, tab_ai, tab_radar, tab_sip = st.tabs(
    ["📈 Chart", "🚨 Signals", "🤖 AI Advisor", "📡 Opportunity Radar", "💼 SIP Simulator"])

with tab_chart:
    st.plotly_chart(build_chart(view, symbol, info), width="stretch")

with tab_signals:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("🚨 Smart Signals")
        for kind, text in info["signals"]:
            KIND_RENDER[kind](text)
        st.write(f"🟢 Support (20d): **₹{info['support']:,.2f}**  |  🔴 Resistance (20d): **₹{info['resistance']:,.2f}**")
    with right:
        st.subheader("📰 Latest Headlines")
        news = get_news(symbol)
        if news:
            for title, url in news:
                st.markdown(f"- [{title}]({url})" if url else f"- {title}")
        else:
            st.caption("No headlines available for this symbol.")

with tab_ai:
    st.subheader("🤖 AI Financial Advisor")
    st.caption("The AI sees live indicators for the selected stock, not just the price.")
    if not get_groq_key():
        st.caption("ℹ️ Running in demo mode — add `GROQ_API_KEY` to `.streamlit/secrets.toml` for real AI answers.")
    question = st.text_input("Ask anything (e.g., Should I buy now?)", key="ai_q")
    if st.button("Ask AI"):
        if not question.strip():
            st.warning("Type a question first.")
        else:
            with st.spinner("Thinking…"):
                try:
                    st.markdown(ask_ai(symbol, info, question.strip()))
                except Exception as e:
                    st.error(f"AI error: {e}")

with tab_radar:
    st.subheader("📡 Opportunity Radar — NIFTY large-cap scan")
    st.caption("Scores every stock on RSI, trend, MACD and breakouts. Higher score = stronger bullish setup.")
    if st.button("Scan watchlist"):
        with st.spinner("Scanning 20 stocks…"):
            st.session_state["radar"] = scan_watchlist(tuple(WATCHLIST))
    radar = st.session_state.get("radar")
    if radar is not None:
        if radar.empty:
            st.error("Scan failed — data source unreachable. Try again in a moment.")
        else:
            radar = radar.sort_values("Score", ascending=False).reset_index(drop=True)
            st.dataframe(radar, width="stretch", hide_index=True)

with tab_sip:
    st.subheader("💼 SIP Simulator")
    c1, c2, c3 = st.columns(3)
    monthly = c1.number_input("Monthly Investment ₹", 500, 1_000_000, 5000, step=500)
    years = c2.slider("Years", 1, 30, 10)
    annual = c3.slider("Expected return (% p.a.)", 4, 20, 12)
    proj = sip_projection(monthly, years, annual)
    invested, final = proj["Invested"].iloc[-1], proj["Value"].iloc[-1]
    s1, s2, s3 = st.columns(3)
    s1.metric("Total Invested", inr(invested))
    s2.metric("Estimated Value", inr(final))
    s3.metric("Estimated Gain", inr(final - invested), delta=f"{(final / invested - 1) * 100:.0f}%")
    fig_sip = go.Figure()
    fig_sip.add_trace(go.Scatter(x=proj["Month"] / 12, y=proj["Invested"], name="Invested", fill="tozeroy"))
    fig_sip.add_trace(go.Scatter(x=proj["Month"] / 12, y=proj["Value"], name="Estimated value", fill="tonexty"))
    fig_sip.update_layout(template="plotly_dark", height=380, xaxis_title="Years",
                          yaxis_title="₹", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_sip, width="stretch")
    st.caption("Assumes a constant return compounded monthly. Actual market returns vary and are not guaranteed.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built for ET AI Hackathon 2026 🚀 | Data: Yahoo Finance (may be delayed) | "
           "For education only — not investment advice.")
