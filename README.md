# 🛎️ InvestorSignal AI

**Opportunity Radar for Indian Investors**  
Built for **ET AI Hackathon 2026 – Problem Statement 6**

---

## 🚀 Overview

InvestorSignal AI is an AI-powered multi-agent system that transforms raw stock market data into **actionable investment signals**.

Unlike traditional dashboards, this system focuses on:
- Signal detection (not just charts)
- Automated technical analysis
- Clear Buy/Sell/Neutral decisions
- Simple Hinglish explanations for users

---

## 🎯 Problem

India has **14+ crore retail investors**, but most:
- Rely on tips or news
- Miss key signals like breakouts or volume spikes
- Spend hours researching manually

---

## 💡 Solution

InvestorSignal AI acts as an **Opportunity Radar** that:
- Continuously analyzes stock data
- Detects trading signals automatically
- Explains insights in simple language
- Helps users make smarter investment decisions

---

## ⚙️ Features

✅ Live NSE stock data (yfinance)  
✅ Trading-style charts (candlestick + MA20)  
✅ RSI (Overbought / Oversold detection)  
✅ Support & Resistance levels  
✅ Breakout detection  
✅ Volume spike detection  
✅ 🤖 AI Advisor (Hinglish insights using Groq LLM)  
✅ 📊 SIP Investment Growth Simulator  
✅ Clean tab-based UI  

---

## 🧠 System Architecture

The system follows a **multi-agent architecture**:

1. **Data Ingestion Agent**  
   Fetches real-time stock data  

2. **Technical Analysis Agent**  
   Calculates indicators (RSI, MA, Support/Resistance)  

3. **Opportunity Radar Agent**  
   Detects actionable signals (breakouts, trends)  

4. **AI Reasoning Agent**  
   Converts signals into simple advice  

5. **Portfolio Simulator Agent**  
   Estimates future SIP growth  

---

## 🔄 Data Flow

User Input → Data Fetch → Analysis → Signal Detection → AI Explanation → UI Output

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit  
- **Data Source**: yfinance (NSE stocks)  
- **Visualization**: Plotly  
- **AI Engine**: Groq (LLaMA 3.3 70B)  
- **Processing**: Pandas  

---

## 🌐 Live Demo

🚀 Try the app here:  
👉 https://omnandre07-investorsignal-ai-app-ryr936.streamlit.app/
No installation required — works directly in browser
