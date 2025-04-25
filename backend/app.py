import json
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS


import pandas as pd
from helpers.query_system import EthicalInvestmentQuerySystem, load_stock_data


os.environ["ROOT_PATH"] = os.path.abspath(os.path.join("..", os.curdir))


current_directory = os.path.dirname(os.path.abspath(__file__))


json_file_path = os.path.join(current_directory, "sp500.json")


with open(json_file_path, "r") as file:
    json_text = file.read()
    stocks_data = load_stock_data(json_text)


# Load the new sentiment data - and convert it to a list which is what the class expects
new_sentiment_file_path = os.path.join(current_directory, "final_sentiment_summary.json")
sentiment_data = []
try:
    with open(new_sentiment_file_path, "r") as file:
        sentiment_data = json.load(file)

    print(f"Loaded sentiment data for {len(sentiment_data)} stocks")
except FileNotFoundError:
    print("Sentiment data file not found. Proceeding without sentiment data.")
except json.JSONDecodeError:
    print("Error parsing sentiment data JSON. Proceeding without sentiment data.")


query_system = EthicalInvestmentQuerySystem(stocks_data, sentiment_data)

app = Flask(__name__)
CORS(app)

MAX_RESULTS = 24


@app.route("/query", methods=["GET"])
def query_endpoint():
    user_query = request.args.get("query", default="", type=str)
    # Call rank_stocks for the initial, unrefinded results
    results = query_system.rank_stocks(user_query)[:MAX_RESULTS]

    # # Log debug info about the first result
    # if results and len(results) > 0:
    #     first = results[0]
    #     # print(f"First result: {first['symbol']}")
    #     if "sentiment" in first:
    #         print(f"Sentiment included: {list(first['sentiment'].keys())}")
    #     else:
    #         print("No sentiment data for first result")

    return jsonify(results)


@app.route("/refine", methods=["POST"])
def refine_endpoint():
    """Handles user feedback to refine search results using Rocchio"""
    data = request.get_json()

    original_query = data.get("original_query")
    relevant_symbols = data.get("relevant_symbols", [])
    displayed_symbols = data.get("displayed_symbols", [])
    nonrelevant_symbols = list(set(displayed_symbols) - set(relevant_symbols))

    refined_results = query_system.refine_results_with_feedback(
        original_query, relevant_symbols, nonrelevant_symbols
    )[:MAX_RESULTS]

    return jsonify(refined_results)


@app.route("/debug-sentiment", methods=["GET"])
def debug_sentiment():
    """Debug endpoint to check sentiment data"""
    ticker = request.args.get("ticker", default="AAPL", type=str)

    # Find sentiment for this ticker
    ticker_sentiment = None
    for item in sentiment_data:
        if item["ticker"] == ticker:
            ticker_sentiment = item
            break

    # Get a sample of available tickers
    available_tickers = [item["ticker"] for item in sentiment_data[:10]]

    # Get sample query result
    sample_result = None
    results = query_system.rank_stocks("tech")[:5]
    for result in results:
        if result["symbol"] == ticker:
            sample_result = result
            break

    response = {
        "ticker": ticker,
        "has_sentiment": ticker_sentiment is not None,
        "sentiment_data": ticker_sentiment,
        "sentiment_count": len(sentiment_data),
        "available_tickers_sample": available_tickers,
        "sample_result": sample_result,
    }

    return jsonify(response)


@app.route("/")
def home():
    return render_template("base.html", title="Robingood - Ethical Investing")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
