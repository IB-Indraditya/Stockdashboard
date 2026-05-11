from flask import Flask, render_template, request, jsonify
import yfinance as yf
from model import get_prediction

app = Flask(__name__)

# =========================
# 🔥 POPULAR STOCKS
# =========================

POPULAR_STOCKS = [

    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "ITC.NS",
    "LT.NS",
    "WIPRO.NS",
    "TATAMOTORS.NS",
    "AXISBANK.NS",
    "BAJFINANCE.NS"

]

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
# 🔥 SIDEBAR STOCK DATA
# =========================

def get_sidebar_data():

    sidebar_data = []

    for stock in POPULAR_STOCKS:

        try:

            data = yf.download(

                stock,

                period="1d",

                interval="1m",

                auto_adjust=True,

                progress=False

            )

            if not data.empty:

                data.columns = [

                    col[0]
                    if isinstance(col, tuple)
                    else col

                    for col in data.columns

                ]

                latest_price = round(

                    float(
                        data['Close'].dropna().iloc[-1]
                    ),

                    2

                )

                open_price = round(

                    float(
                        data['Open'].dropna().iloc[0]
                    ),

                    2

                )

                change = round(
                    latest_price - open_price,
                    2
                )

                sidebar_data.append({

                    "ticker": stock.replace(".NS", ""),

                    "price": latest_price,

                    "change": change

                })

        except Exception as e:

            print("Sidebar Error:", e)

    return sidebar_data

# =========================
# 🔥 LIVE SIDEBAR API
# =========================

@app.route("/live-stocks")
def live_stocks():

    return jsonify(
        get_sidebar_data()
    )

# =========================
# 🔥 LIVE SINGLE STOCK API
# =========================

@app.route("/live-data/<ticker>")
def live_data(ticker):

    try:

        data = yf.download(

            ticker,

            period="1d",

            interval="1m",

            progress=False

        )

        if data.empty:

            return jsonify({
                "price": 0
            })

        data.columns = [

            col[0]
            if isinstance(col, tuple)
            else col

            for col in data.columns

        ]

        latest_price = round(

            float(
                data['Close'].dropna().iloc[-1]
            ),

            2

        )

        return jsonify({

            "price": latest_price

        })

    except Exception as e:

        print("Live API Error:", e)

        return jsonify({
            "price": 0
        })

# =========================
# 🔥 HOME PAGE
# =========================

@app.route("/")
def home():

    return render_template(

        "index.html",

        sidebar_data=get_sidebar_data(),

        selected_period="6mo",

        selected_interval="1d"

    )

# =========================
# 🔥 PREDICTION ROUTE
# =========================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        ticker = request.form.get(
            "ticker"
        ).upper().strip()

        period = request.form.get(
            "period",
            "6mo"
        )

        interval = request.form.get(
            "interval",
            "1d"
        )

        # AUTO ADD .NS

        if ".NS" not in ticker:

            ticker += ".NS"

        # =========================
        # FETCH STOCK DATA
        # =========================

        data = yf.download(

            ticker,

            period=period,

            interval=interval,

            auto_adjust=True,

            progress=False

        )

        if data.empty:

            return render_template(

                "index.html",

                error="Stock not found",

                sidebar_data=get_sidebar_data()

            )

        # FIX MULTI INDEX

        data.columns = [

            col[0]
            if isinstance(col, tuple)
            else col

            for col in data.columns

        ]

        close = data['Close'].dropna()

        if close.empty:

            return render_template(

                "index.html",

                error="No price data",

                sidebar_data=get_sidebar_data()

            )

        # =========================
        # CURRENT PRICE
        # =========================

        latest_price = round(

            float(close.iloc[-1]),

            2

        )

        # =========================
        # ML PREDICTION
        # =========================

        prediction = get_prediction(ticker)

        if prediction is None:

            prediction = latest_price

        prediction = round(
            float(prediction),
            2
        )

        # =========================
        # BUY SELL SIGNAL
        # =========================

        signal = (
            "BUY"
            if prediction > latest_price
            else "SELL"
        )

        # =========================
        # RSI
        # =========================

        rsi_series = calculate_rsi(close).dropna()

        if rsi_series.empty:

            rsi_value = 0

            rsi_signal = "N/A"

            rsi_data = []

        else:

            rsi_value = round(

                float(
                    rsi_series.iloc[-1]
                ),

                2

            )

            if rsi_value < 30:

                rsi_signal = "OVERSOLD (BUY)"

            elif rsi_value > 70:

                rsi_signal = "OVERBOUGHT (SELL)"

            else:

                rsi_signal = "NEUTRAL"

            rsi_data = [

                round(float(x), 2)

                for x in rsi_series.tail(200)

            ]

        # =========================
        # MACD
        # =========================

        macd_line, signal_line = calculate_macd(close)

        macd_line = macd_line.dropna()

        signal_line = signal_line.dropna()

        if macd_line.empty:

            macd_value = 0

            macd_trade = "N/A"

            macd_data = []

            signal_data = []

        else:

            macd_value = round(

                float(
                    macd_line.iloc[-1]
                ),

                2

            )

            signal_latest = round(

                float(
                    signal_line.iloc[-1]
                ),

                2

            )

            macd_trade = (

                "BUY"
                if macd_value > signal_latest
                else "SELL"

            )

            macd_data = [

                round(float(x), 2)

                for x in macd_line.tail(200)

            ]

            signal_data = [

                round(float(x), 2)

                for x in signal_line.tail(200)

            ]

        # =========================
        # PRICE CHART
        # =========================

        prices = [

            round(float(x), 2)

            for x in close.tail(200)

        ]

        labels = [

            str(
                date.strftime("%d-%m-%Y")
            )

            for date in close.tail(
                len(prices)
            ).index

        ]

        # =========================
        # RENDER PAGE
        # =========================

        return render_template(

            "index.html",

            sidebar_data=get_sidebar_data(),

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

            macd_trade=macd_trade,

            macd_data=macd_data,

            signal_data=signal_data,

            selected_period=period,

            selected_interval=interval

        )

    except Exception as e:

        print("APP ERROR:", e)

        return render_template(

            "index.html",

            error="Something went wrong",

            sidebar_data=get_sidebar_data()

        )

# =========================
# 🔥 RUN APP
# =========================

if __name__ == "__main__":

    app.run(
        debug=True
    )