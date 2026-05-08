from flask import Flask, render_template, request
import yfinance as yf
from model import get_prediction

app = Flask(__name__)


# 🔥 RSI FUNCTION
def calculate_rsi(close, period=14):
    delta = close.diff()

    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


# 🔥 MACD FUNCTION
def calculate_macd(close):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()

    return macd, signal


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        ticker = request.form.get("ticker").upper().strip()

        # Auto add .NS
        if ".NS" not in ticker:
            ticker += ".NS"

        # Fetch data
        data = yf.download(ticker, period="1mo", auto_adjust=True)

        if data.empty:
            return render_template("index.html", error="Stock not found")

        # Fix column format
        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

        close = data['Close'].dropna()

        if close.empty:
            return render_template("index.html", error="No price data")

        latest_price = round(float(close.iloc[-1]), 2)

        # 🔥 ML Prediction
        prediction = get_prediction(ticker)
        if prediction is None:
            prediction = latest_price

        # 🔥 BUY / SELL SIGNAL (ML)
        signal = "BUY" if prediction > latest_price else "SELL"

        # =========================
        # 🔥 RSI
        # =========================
        rsi_series = calculate_rsi(close).dropna()

        if rsi_series.empty:
            rsi_value = 0
            rsi_signal = "N/A"
            rsi_data = []
        else:
            rsi_value = round(float(rsi_series.iloc[-1]), 2)

            if rsi_value < 30:
                rsi_signal = "OVERSOLD (BUY)"
            elif rsi_value > 70:
                rsi_signal = "OVERBOUGHT (SELL)"
            else:
                rsi_signal = "NEUTRAL"

            rsi_data = list(rsi_series.tail(30))

        # =========================
        # 🔥 MACD
        # =========================
        macd_line, signal_line = calculate_macd(close)

        macd_line = macd_line.dropna()
        signal_line = signal_line.dropna()

        if macd_line.empty or signal_line.empty:
            macd_value = 0
            macd_signal_value = 0
            macd_trade = "N/A"
            macd_data = []
            signal_data = []
        else:
            macd_value = round(float(macd_line.iloc[-1]), 2)
            macd_signal_value = round(float(signal_line.iloc[-1]), 2)

            if macd_value > macd_signal_value:
                macd_trade = "BUY"
            else:
                macd_trade = "SELL"

            macd_data = list(macd_line.tail(30))
            signal_data = list(signal_line.tail(30))

        # =========================
        # 📊 PRICE CHART
        # =========================
        prices = list(close.tail(30))
        labels = list(range(len(prices)))

        return render_template(
            "index.html",
            ticker=ticker,
            price=latest_price,
            prediction=prediction,
            signal=signal,
            prices=prices,
            labels=labels,
            rsi=rsi_value,
            rsi_signal=rsi_signal,
            rsi_data=rsi_data,
            macd=macd_value,
            macd_signal=macd_signal_value,
            macd_trade=macd_trade,
            macd_data=macd_data,
            signal_data=signal_data
        )

    except Exception as e:
        print("APP ERROR:", e)
        return render_template("index.html", error="Something went wrong")


if __name__ == "__main__":
    app.run(debug=True)
    # 99397432de0cacddf6b91385253908ba