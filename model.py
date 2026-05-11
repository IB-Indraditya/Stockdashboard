import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# =========================
# 🔥 RSI FUNCTION
# =========================

def calculate_rsi(close, period=14):

    delta = close.diff()

    gain = (
        delta.where(delta > 0, 0)
        .rolling(window=period)
        .mean()
    )

    loss = (
        -delta.where(delta < 0, 0)
        .rolling(window=period)
        .mean()
    )

    rs = gain / loss

    rsi = 100 - (100 / (1 + rs))

    return rsi

# =========================
# 🔥 MACD FUNCTION
# =========================

def calculate_macd(close):

    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    macd = ema12 - ema26

    signal = macd.ewm(
        span=9,
        adjust=False
    ).mean()

    return macd, signal

# =========================
# 🔥 MAIN PREDICTION FUNCTION
# =========================

def get_prediction(ticker):

    try:

        # =========================
        # 🔥 FETCH 6 MONTH DATA
        # =========================

        df = yf.download(
            ticker,
            period="6mo",
            interval="1d",
            auto_adjust=True,
            progress=False
        )

        # =========================
        # 🔥 VALIDATION
        # =========================

        if df.empty or len(df) < 60:

            print("Not enough stock data")

            return None

        # =========================
        # 🔥 FIX MULTI INDEX
        # =========================

        df.columns = [

            col[0] if isinstance(col, tuple)
            else col

            for col in df.columns
        ]

        # =========================
        # 🔥 CHECK CLOSE COLUMN
        # =========================

        if 'Close' not in df.columns:

            print("Close column missing")

            return None

        # =========================
        # 🔥 FEATURE ENGINEERING
        # =========================

        # DAILY RETURN

        df['Return'] = df['Close'].pct_change()

        # MOVING AVERAGES

        df['MA5'] = (
            df['Close']
            .rolling(window=5)
            .mean()
        )

        df['MA10'] = (
            df['Close']
            .rolling(window=10)
            .mean()
        )

        df['MA20'] = (
            df['Close']
            .rolling(window=20)
            .mean()
        )

        df['MA50'] = (
            df['Close']
            .rolling(window=50)
            .mean()
        )

        # RSI

        df['RSI'] = calculate_rsi(
            df['Close']
        )

        # MACD

        macd, signal = calculate_macd(
            df['Close']
        )

        df['MACD'] = macd

        df['Signal_Line'] = signal

        # VOLATILITY

        df['Volatility'] = (
            df['Return']
            .rolling(window=10)
            .std()
        )

        # MOMENTUM

        df['Momentum'] = (
            df['Close'] - df['Close'].shift(5)
        )

        # =========================
        # 🔥 CLEAN DATA
        # =========================

        df = df.dropna()

        if len(df) < 30:

            print("Not enough processed data")

            return None

        # =========================
        # 🔥 FEATURE COLUMNS
        # =========================

        feature_columns = [

            'Close',
            'Return',
            'MA5',
            'MA10',
            'MA20',
            'MA50',
            'RSI',
            'MACD',
            'Signal_Line',
            'Volatility',
            'Momentum'

        ]

        # =========================
        # 🔥 INPUT / OUTPUT
        # =========================

        X = df[feature_columns]

        # NEXT DAY CLOSE

        y = df['Close'].shift(-1)

        # REMOVE LAST ROW

        X = X[:-1]

        y = y[:-1]

        # =========================
        # 🔥 FEATURE SCALING
        # =========================

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        # =========================
        # 🔥 MODEL TRAINING
        # =========================

        model = LinearRegression()

        model.fit(X_scaled, y)

        # =========================
        # 🔥 LAST DAY DATA
        # =========================

        last_data = df[
            feature_columns
        ].iloc[-1:]

        last_scaled = scaler.transform(
            last_data
        )

        # =========================
        # 🔥 PREDICTION
        # =========================

        prediction = model.predict(
            last_scaled
        )[0]

        # =========================
        # 🔥 OUTPUT
        # =========================

        latest_price = round(
            float(df['Close'].iloc[-1]),
            2
        )

        prediction = round(
            float(prediction),
            2
        )

        # =========================
        # 🔥 DEBUG PRINT
        # =========================

        print("\n========================")

        print("Ticker:", ticker)

        print("Latest Price:", latest_price)

        print("Predicted Price:", prediction)

        print("RSI:", round(float(df['RSI'].iloc[-1]), 2))

        print("MACD:", round(float(df['MACD'].iloc[-1]), 2))

        print("========================\n")

        # =========================
        # 🔥 RETURN PREDICTION
        # =========================

        return prediction

    except Exception as e:

        print("MODEL ERROR:", e)

        return None