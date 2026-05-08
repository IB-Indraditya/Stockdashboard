import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def get_prediction(ticker):
    try:
        # 🔥 Fetch data
        df = yf.download(ticker, period="6mo", auto_adjust=True)

        if df.empty or len(df) < 50:
            print("Not enough data")
            return None

        # Fix column format
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        if 'Close' not in df.columns:
            print("Close column missing")
            return None

        # =========================
        # 🔥 FEATURE ENGINEERING
        # =========================
        df['Return'] = df['Close'].pct_change()
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()

        df = df.dropna()

        if len(df) < 30:
            print("Not enough processed data")
            return None

        # =========================
        # 🔥 INPUT / OUTPUT
        # =========================
        X = df[['Close', 'Return', 'MA5', 'MA10']]
        y = df['Close'].shift(-1)

        X = X[:-1]
        y = y[:-1]

        # =========================
        # 🔥 MODEL
        # =========================
        model = LinearRegression()
        model.fit(X, y)

        # =========================
        # 🔮 PREDICTION
        # =========================
        last_data = df[['Close', 'Return', 'MA5', 'MA10']].iloc[-1].values.reshape(1, -1)
        prediction = model.predict(last_data)[0]

        print("Last Price:", df['Close'].iloc[-1])
        print("Predicted:", prediction)

        return round(float(prediction), 2)

    except Exception as e:
        print("MODEL ERROR:", e)
        return None