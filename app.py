"""
Stock Market Analyzer (Alpha Vantage API)
-----------------------------------------
Fetches daily stock prices, calculates indicators (moving averages, RSI,
volatility, volume spikes) and prints plain-English insights.
It also saves a chart as <SYMBOL>_analysis.png.
Get a free API key: ZUI2WM887DQ01UAV. 

"""

import os
import requests
import pandas as pd
import matplotlib.pyplot as plt

BASE_URL = "https://www.alphavantage.co/query"


# ---------------------------------------------------------------
# 1. FETCH DATA
# ---------------------------------------------------------------
def fetch_daily(symbol, api_key):
    """Download the last ~100 trading days for a symbol as a DataFrame."""
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": "compact",
        "apikey": api_key,
    }
    response = requests.get(BASE_URL, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()

    if "Time Series (Daily)" not in data:
        # Alpha Vantage sends errors / rate-limit messages under these keys
        message = (
            data.get("Error Message")
            or data.get("Note")
            or data.get("Information")
            or "Unexpected response from API."
        )
        raise ValueError(message)

    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
    df = df.astype(float)
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index = pd.to_datetime(df.index)
    return df.sort_index()  # oldest -> newest


# ---------------------------------------------------------------
# 2. CALCULATE INDICATORS
# ---------------------------------------------------------------
def add_indicators(df):
    df = df.copy()
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    df["return_pct"] = df["close"].pct_change() * 100
    df["volatility"] = df["return_pct"].rolling(20).std()
    df["avg_volume20"] = df["volume"].rolling(20).mean()

    # RSI (14 days)
    change = df["close"].diff()
    gain = change.clip(lower=0).rolling(14).mean()
    loss = (-change.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


# ---------------------------------------------------------------
# 3. GENERATE INSIGHTS
# ---------------------------------------------------------------
def generate_insights(df):
    insights = []
    last = df.iloc[-1]
    prev = df.iloc[-2]

    # Latest price move
    insights.append(
        f"Last close: {last['close']:.2f} "
        f"({last['return_pct']:+.2f}% vs previous day)."
    )

    # 30-day performance
    if len(df) > 30:
        month_ago = df["close"].iloc[-31]
        change = (last["close"] - month_ago) / month_ago * 100
        direction = "up" if change > 0 else "down"
        insights.append(f"Over the last 30 trading days the stock is {direction} {abs(change):.1f}%.")

    # Trend using moving averages
    if pd.notna(last["sma50"]):
        if last["sma20"] > last["sma50"] and last["close"] > last["sma20"]:
            insights.append("Trend: UPTREND (price above 20-day average, which is above 50-day average).")
        elif last["sma20"] < last["sma50"] and last["close"] < last["sma20"]:
            insights.append("Trend: DOWNTREND (price below 20-day average, which is below 50-day average).")
        else:
            insights.append("Trend: SIDEWAYS / MIXED signals.")

        # Crossover in the last 10 days
        recent = df.tail(10)
        diff = recent["sma20"] - recent["sma50"]
        if (diff.iloc[0] < 0) and (diff.iloc[-1] > 0):
            insights.append("Pattern: Golden cross - 20-day average crossed ABOVE 50-day average recently (bullish).")
        elif (diff.iloc[0] > 0) and (diff.iloc[-1] < 0):
            insights.append("Pattern: Death cross - 20-day average crossed BELOW 50-day average recently (bearish).")

    # RSI
    if pd.notna(last["rsi"]):
        if last["rsi"] >= 70:
            insights.append(f"RSI is {last['rsi']:.1f}: OVERBOUGHT zone - price may be stretched, a pullback is possible.")
        elif last["rsi"] <= 30:
            insights.append(f"RSI is {last['rsi']:.1f}: OVERSOLD zone - price may be due for a bounce.")
        else:
            insights.append(f"RSI is {last['rsi']:.1f}: neutral zone.")

    # Volatility
    if pd.notna(last["volatility"]):
        level = "high" if last["volatility"] > 2.5 else "moderate" if last["volatility"] > 1.2 else "low"
        insights.append(f"Volatility (20-day std of daily returns): {last['volatility']:.2f}% -> {level}.")

    # Volume spike
    if pd.notna(last["avg_volume20"]) and last["volume"] > 1.5 * last["avg_volume20"]:
        insights.append("Volume today is more than 1.5x the 20-day average - unusual activity.")

    # Support and resistance (last 30 days)
    window = df.tail(30)
    insights.append(
        f"Approx. support (30-day low): {window['low'].min():.2f} | "
        f"resistance (30-day high): {window['high'].max():.2f}."
    )

    # Best and worst day
    best = df["return_pct"].idxmax()
    worst = df["return_pct"].idxmin()
    insights.append(
        f"Best day: {best.date()} ({df.loc[best, 'return_pct']:+.2f}%) | "
        f"Worst day: {worst.date()} ({df.loc[worst, 'return_pct']:+.2f}%)."
    )
    return insights


# ---------------------------------------------------------------
# 4. CHART
# ---------------------------------------------------------------
def save_chart(df, symbol):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax1.plot(df.index, df["close"], label="Close", linewidth=1.5)
    ax1.plot(df.index, df["sma20"], label="SMA 20", linestyle="--")
    ax1.plot(df.index, df["sma50"], label="SMA 50", linestyle="--")
    ax1.set_title(f"{symbol} - Price and Moving Averages")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(df.index, df["rsi"], color="purple")
    ax2.axhline(70, color="red", linestyle=":")
    ax2.axhline(30, color="green", linestyle=":")
    ax2.set_ylabel("RSI")
    ax2.set_ylim(0, 100)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    filename = f"{symbol}_analysis.png"
    plt.savefig(filename, dpi=120)
    plt.close()
    return filename


# ---------------------------------------------------------------
# 5. COMPARE TWO STOCKS
# ---------------------------------------------------------------
def compare(symbol_a, symbol_b, api_key):
    a = fetch_daily(symbol_a, api_key)["close"]
    b = fetch_daily(symbol_b, api_key)["close"]
    both = pd.concat([a, b], axis=1, keys=[symbol_a, symbol_b]).dropna()
    growth = (both.iloc[-1] / both.iloc[0] - 1) * 100
    correlation = both[symbol_a].pct_change().corr(both[symbol_b].pct_change())

    print(f"\n--- {symbol_a} vs {symbol_b} (last {len(both)} trading days) ---")
    print(f"{symbol_a}: {growth[symbol_a]:+.1f}%")
    print(f"{symbol_b}: {growth[symbol_b]:+.1f}%")
    print(f"Correlation of daily returns: {correlation:.2f} "
          f"({'move together' if correlation > 0.6 else 'move fairly independently'})")


# ---------------------------------------------------------------
# 6. MAIN MENU
# ---------------------------------------------------------------
def analyze(symbol, api_key):
    print(f"\nFetching data for {symbol}...")
    df = add_indicators(fetch_daily(symbol, api_key))
    print(f"\n=== {symbol} ANALYSIS ({df.index[0].date()} to {df.index[-1].date()}) ===")
    for line in generate_insights(df):
        print(" -", line)
    chart = save_chart(df, symbol)
    print(f"\nChart saved as {chart}")
    print("Note: educational project only, not financial advice.")


def main():
    api_key = os.getenv("ALPHAVANTAGE_API_KEY") or input("Enter your Alpha Vantage API key: ").strip()

    while True:
        print("\n1. Analyze a stock\n2. Compare two stocks\n3. Exit")
        choice = input("Choose: ").strip()
        try:
            if choice == "1":
                analyze(input("Stock symbol (e.g. IBM, AAPL, MSFT): ").strip().upper(), api_key)
            elif choice == "2":
                s1 = input("First symbol: ").strip().upper()
                s2 = input("Second symbol: ").strip().upper()
                compare(s1, s2, api_key)
            elif choice == "3":
                print("Goodbye!")
                break
            else:
                print("Please choose 1, 2 or 3.")
        except requests.exceptions.RequestException as e:
            print("Network problem:", e)
        except ValueError as e:
            print("API message:", e)


if __name__ == "__main__":
    main()