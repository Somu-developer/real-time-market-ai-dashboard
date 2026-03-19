import pandas as pd
import numpy as np
import requests
import yfinance as yf
from config import News_API_KEY, SYMBOL, NEWS_QUERY

# Fetching data from Yahoo Finance

def fetch_data(symbol=SYMBOL):
    import yfinance as yf
    if symbol.endswith(".SR"):
        df = yf.download(symbol, interval="1d", period="3mo")
    else:
        df = yf.download(symbol, interval="1m", period="5d")
    if df.empty:
        raise ValueError("No data fetched")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={"Close": "price"})
    df = df[['price']].copy()

    return df.dropna()

# Fetching News Data from the News API

def fetch_news(query):
    url = "https://newsapi.org/v2/everything"
    
    params = {
        "q": query,
        "apiKey": News_API_KEY,
        "language": "en",
        "pageSize": 10
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"News API Error: {response.status_code}")
        return []
    data = response.json()
    articles = data.get('articles', [])    
    headlines = [a['title'] for a in articles if a.get('title')]    
    return headlines

# Sentiment Analysis Function (Placeholder)

def compute_sentiment(headlines):
    pos=['gain', 'rise', 'growth', 'positive', 'surge']
    neg=['fall', 'drop', 'decline', 'negative', 'loss']
    score = 0
    for h in headlines:
        h = h.lower()
        score += sum(1 for w in pos if w in h)
        score -= sum(1 for w in neg if w in h)
    return score / len(headlines) if headlines else 0 

#Features RSI 

def compute_rsi(series, window=7):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    avg_loss = avg_loss.replace(0, 1e-10)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def create_features(df, query):
    headlines = fetch_news(query)
    sentiment = compute_sentiment(headlines)
    df['return'] = df['price'].pct_change()
    df['ma_5'] = df['price'].rolling(5).mean()
    df['rsi'] = compute_rsi(df['price'])
    df['sentiment'] = sentiment
#New features
    df['volatility'] = df['price'].rolling(5).std()
    df['momentum'] = df['price'] - df['price'].shift(5)
    df = df.bfill().ffill()
    return df.dropna(), headlines

