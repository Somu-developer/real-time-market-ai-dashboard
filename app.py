import streamlit as st
import joblib
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
from backend import fetch_data, create_features

#Page config
st.set_page_config(page_title="Middle East Market Predictor", layout="wide")

#CSS
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0e1117;
}
[data-testid="stMetric"] {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}
[data-testid="stMetricLabel"] {
    color: #aaaaaa;
}
[data-testid="stMetricValue"] {
    color: white;
    font-size: 28px;
}
.stProgress > div > div {
    background-color: #00c6ff;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Dashboard Controls")
stock_option = st.sidebar.selectbox(
    "Select Stock",
    ["2222.SR", "AAPL", "MSFT"]
)
stock_names = {
    "2222.SR": "Saudi Aramco",
    "AAPL": "Apple",
    "MSFT": "Microsoft"
}

# Get selected name
selected_name = stock_names.get(stock_option, stock_option)
st.title("AI Market Dashboard")

# Display under title
st.markdown(f"### Stock Name: {selected_name} ({stock_option})")
refresh_rate = st.sidebar.slider(
    "Refresh Rate (seconds)",
    30, 300, 60
)
if st.sidebar.button("Refresh Now"):
    st.cache_data.clear()
    st.rerun()

#Auto Refresh
st_autorefresh(interval=refresh_rate * 1000, key="refresh")

#Load Models
@st.cache_resource
def load_models():
    rf = joblib.load("rf_model.pkl")
    xgb = joblib.load("xgb_model.pkl")
    return rf, xgb

rf_model, xgb_model = load_models()

#Fetch Data
@st.cache_data(ttl=60)
def load_data(symbol):
    return fetch_data(symbol)

base_df = load_data(stock_option)

#Sentiment Query
query_map = {
    "2222.SR": "Saudi Aramco OR oil prices OR Middle East economy",
    "AAPL": "Apple stock OR iPhone OR tech market",
    "MSFT": "Microsoft OR AI OR cloud computing"
}
query = query_map.get(stock_option, stock_option)

#Feature Section
feature_df, headlines = create_features(base_df.copy(), query)
##st.write("Rows after features:", len(feature_df))

#Safety Check
if len(feature_df) < 5:
    st.warning("Waiting for enough data...")
    st.stop()

#Prediction Section
features = ['return', 'ma_5', 'rsi', 'sentiment', 'volatility', 'momentum']
latest = feature_df.iloc[-1][features].values.reshape(1, -1)
rf_pred = rf_model.predict(latest)[0]
rf_prob = rf_model.predict_proba(latest)[0][1]
xgb_pred = xgb_model.predict(latest)[0]
xgb_prob = xgb_model.predict_proba(latest)[0][1]

#MetricsSection
price = float(base_df.iloc[-1]['price'])
sentiment = float(feature_df.iloc[-1]['sentiment'])


col1, col2, col3, col4 = st.columns(4)
col1.metric("Price", f"{price:.2f}")
col2.metric("RF", "UP" if rf_pred else "DOWN")
col3.metric("XGB", "UP" if xgb_pred else "DOWN")
col4.metric("Sentiment", f"{sentiment:.2f}")

#Confidence Section
st.subheader("Model Confidence")
c1, c2 = st.columns(2)
with c1:
    st.write(f"Random Forest: {rf_prob:.2%}")
    st.progress(int(rf_prob * 100))
with c2:
    st.write(f"XGBoost: {xgb_prob:.2%}")
    st.progress(int(xgb_prob * 100))

#Chart Section

st.subheader("📈 Price Chart")
fig = px.line(base_df, y='price', title="Live Price Trend")
st.plotly_chart(fig, use_container_width=True)

#News Section
st.subheader("Latest News")
for h in headlines[:5]:
    st.markdown(f" - {h}")
st.caption("Auto-refresh enabled")