# Real-time-market-ai-dashboard
AI-powered market prediction dashboard using live data, sentiment analysis, and machine learning models (Random Forest &amp; XGBoost), with a focus on Middle East markets.

I built this project to understand how market behaviour changes in real time and how different signals can be combined to make better predictions. The goal is simple: use live data, identify patterns, and estimate what might happen next.

What this project does

- Fetches live stock data for selected companies such as Saudi Aramco, Apple, and Microsoft
- Collects recent news and calculates a simple sentiment score
- Generates technical indicators such as returns, moving averages, RSI, volatility, and momentum
- Trains machine learning models to predict short term price movement
- Displays everything in an interactive dashboard

How it works

Instead of relying only on price, the project combines multiple signals:

- Price movement to understand recent trends
- RSI to identify overbought or oversold conditions
- Momentum and volatility to capture short term behaviour
- News sentiment to reflect market perception

These features are passed into machine learning models.

Two models are used:

- Random Forest for stability and handling noisy data
- XGBoost for capturing more complex relationships

Both models learn from historical patterns and predict whether the next movement is likely to go up or down.

Output

The dashboard shows:

- Live price chart
- Current price and sentiment score
- Model predictions (up or down)
- Confidence levels from both models
- Latest related news headlines

Important note

This is not a trading system and should not be used as financial advice. The goal is to understand how data, models, and simple signals can work together in a real world scenario.

Tech stack

- Python
- Pandas
- yfinance
- XGBoost
- Random Forest
- Streamlit

Why I built this

I wanted to move beyond theory and build something practical using live data. This project helped me understand how:

- real time data is collected and handled
- features are engineered from raw data
- models are trained and used for predictions
- dashboards are built for user interaction

Next steps

- Improve sentiment analysis using more advanced NLP models
- Add prediction tracking to compare model performance
- Expand to more markets and data sources
- Improve visualization with more advanced charts
