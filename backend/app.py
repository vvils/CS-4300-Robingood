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


sentiment_file_path = os.path.join(current_directory, "sentiment_analysis.json")
sentiment_data = []
try:
    with open(sentiment_file_path, "r") as file:
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
    # Call rank_stocks for the initial, unrefined results
    results = query_system.rank_stocks(user_query)[:MAX_RESULTS]
    return jsonify(results)

@app.route("/refine", methods=["POST"])
def refine_endpoint():
    """Handles user feedback to refine search results using Rocchio"""
    data = request.get_json()

    original_query = data.get("original_query")
    # print(original_query)
    relevant_symbols = data.get("relevant_symbols", []) 
    # print(f"Relevant symbols: {relevant_symbols}")
    displayed_symbols = data.get("displayed_symbols", [])
    # print(f"Displayed symbols: {displayed_symbols}")
    nonrelevant_symbols = list(set(displayed_symbols) - set(relevant_symbols))

    refined_results = query_system.refine_results_with_feedback(original_query, relevant_symbols, nonrelevant_symbols)[:MAX_RESULTS] 
    # for stock in refined_results:
    #     stock["symbol"] = stock["symbol"].upper()
    #     print(stock["symbol"])
    return jsonify(refined_results)

@app.route("/")
def home():
    return render_template("base.html", title="sample html")


if "DB_NAME" not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)
