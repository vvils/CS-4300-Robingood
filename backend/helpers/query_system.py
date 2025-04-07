import json
import re
from collections import defaultdict
import math
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np

class EthicalInvestmentQuerySystem:
    def __init__(self, stocks_data, sentiment_data=None):

        self.sentiment_data = {}
        if sentiment_data:
            for item in sentiment_data:
                self.sentiment_data[item["ticker"]] = item

        self.field_mappings = {
            "environmental": [
                "environmentScore",
                "environment",
                "eco",
                "green",
                "sustainable",
                "carbon",
                "climate",
            ],
            "social": [
                "socialScore",
                "society",
                "community",
                "people",
                "ethical",
                "human rights",
                "social responsibility",
            ],
            "governance": [
                "governanceScore",
                "management",
                "leadership",
                "board",
                "transparency",
                "corporate governance",
            ],
            "esg": [
                "totalEsg",
                "sustainability",
                "responsible",
                "ethical investing",
                "sustainable investing",
            ],
            "risk": [
                "overallRisk",
                "risky",
                "danger",
                "safe",
                "safety",
                "volatility",
                "stability",
            ],
            "controversy": [
                "highestControversy",
                "controversial",
                "scandal",
                "dispute",
                "issue",
                "problems",
            ],
            "market cap": [
                "marketCap",
                "size",
                "capitalization",
                "market value",
                "company size",
                "large cap",
                "small cap",
            ],
            "beta": [
                "beta",
                "volatility",
                "stable",
                "stability",
                "market risk",
                "market sensitivity",
            ],
            "percentile": [
                "percentile",
                "rank",
                "standing",
                "position",
                "rating",
                "relative performance",
            ],
            "sector": ["GICS Sector", "industry", "field", "domain", "market segment"],
            "sentiment": [
                "sentiment",
                "social sentiment",
                "public opinion",
                "market sentiment",
                "tweets",
                "social media",
                "twitter",
                "buzz",
                "popular opinion",
            ],
        }
        self.reverse_mappings = {}
        for key, values in self.field_mappings.items():
            for value in values:
                self.reverse_mappings[value] = key

        self.modifiers = {
            "high": [
                "high",
                "good",
                "strong",
                "great",
                "impressive",
                "positive",
                "large",
                "big",
                "higher",
                "better",
            ],
            "low": [
                "low",
                "bad",
                "weak",
                "poor",
                "negative",
                "minimal",
                "small",
                "lower",
                "worse",
            ],
        }
        self.reverse_modifiers = {}
        for key, values in self.modifiers.items():
            for value in values:
                self.reverse_modifiers[value] = 1.0 if key == "high" else -1.0

        self.intensifiers = {
            "very": 1.5,
            "extremely": 2.0,
            "highly": 1.7,
            "incredibly": 1.8,
            "somewhat": 0.7,
            "slightly": 0.5,
            "a bit": 0.6,
            "a lot": 1.6,
            "tremendously": 1.9,
            "exceptionally": 1.8,
            "moderately": 0.8,
        }

        self.negations = ["not", "no", "never", "neither", "nor", "barely", "hardly"]

        self.sectors = [
            "information technology",
            "tech",
            "health care",
            "healthcare",
            "financials",
            "financial",
            "banks",
            "consumer discretionary",
            "retail",
            "communication services",
            "telecom",
            "media",
            "industrials",
            "manufacturing",
            "consumer staples",
            "food",
            "beverages",
            "energy",
            "oil",
            "gas",
            "utilities",
            "real estate",
            "property",
            "materials",
            "basic materials",
            "mining",
        ]

        self.sector_mapping = {
            "tech": "information technology",
            "technology": "information technology",
            "healthcare": "health care",
            "medical": "health care",
            "pharma": "health care",
            "pharmaceutical": "health care",
            "financial": "financials",
            "banks": "financials",
            "banking": "financials",
            "insurance": "financials",
            "retail": "consumer discretionary",
            "telecom": "communication services",
            "media": "communication services",
            "telecommunications": "communication services",
            "manufacturing": "industrials",
            "industrial": "industrials",
            "food": "consumer staples",
            "beverages": "consumer staples",
            "household": "consumer staples",
            "oil": "energy",
            "gas": "energy",
            "petroleum": "energy",
            "renewable energy": "energy",
            "property": "real estate",
            "mining": "materials",
            "basic materials": "materials",
            "chemicals": "materials",
        }

        self.normalized_data = self.normalize_stock_data(stocks_data)
        self.original_stocks_map = {
            stock['Symbol']: stock for stock in stocks_data if 'Symbol' in stock
        }
        self.normalized_stocks_map = {
            stock['Symbol']: stock for stock in self.normalized_data if 'Symbol' in stock
        }

        self.available_sectors = set()
        for stock in stocks_data:
            if "GICS Sector" in stock and stock["GICS Sector"]:
                self.available_sectors.add(stock["GICS Sector"].lower())

        try:
            self.stopwords = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords")
            self.stopwords = set(stopwords.words("english"))

    def tokenize(self, text):
        """Simple tokenization by splitting on spaces and removing punctuation"""
        text = text.lower()
        text = re.sub(r"\s+", " ", text)

        for multi_word in ["a bit", "a lot"]:
            if multi_word in text:
                text = text.replace(multi_word, multi_word.replace(" ", "_"))

        text = re.sub(r"[^\w\s]", "", text)
        tokens = text.split()

        tokens = [t.replace("_", " ") for t in tokens]

        return tokens

    def normalize_stock_data(self, stocks_data):
        """
        Normalize stock data to make features comparable
        """

        features = [
            "environmentScore",
            "socialScore",
            "governanceScore",
            "totalEsg",
            "overallRisk",
            "highestControversy",
            "marketCap",
            "beta",
            "percentile",
        ]

        feature_arrays = {}
        for feature in features:
            feature_arrays[feature] = [
                float(stock.get(feature, 0)) for stock in stocks_data
            ]

        feature_min_max = {}
        for feature in features:
            values = feature_arrays[feature]
            if values:
                feature_min_max[feature] = (min(values), max(values))
            else:
                feature_min_max[feature] = (0, 1)

        normalized_data = []
        for stock in stocks_data:
            normalized_stock = stock.copy()
            for feature in features:
                if feature in stock:

                    value = float(stock[feature])
                    min_val, max_val = feature_min_max[feature]

                    if max_val > min_val:
                        if feature in ["highestControversy", "overallRisk", "beta"]:

                            normalized_stock[feature] = (max_val - value) / (
                                max_val - min_val
                            )
                        else:
                            normalized_stock[feature] = (value - min_val) / (
                                max_val - min_val
                            )
                    else:
                        normalized_stock[feature] = 0.5

            normalized_data.append(normalized_stock)

        return normalized_data

    def parse_query(self, query_text):
        """
        Parse a natural language query and convert it to a weighted vector
        representing the importance of different factors
        """
        tokens = self.tokenize(query_text)
        query_vector = defaultdict(float)
        negation_active = False
        window_size = 4
        specified_sectors = []

        sentiment_mentioned = any(
            token in self.field_mappings["sentiment"] for token in tokens
        )

        for token in tokens:

            if token in self.sectors:

                sector = self.sector_mapping.get(token, token)
                specified_sectors.append(sector)

            elif token in self.sector_mapping:
                specified_sectors.append(self.sector_mapping[token])

        if not specified_sectors:
            for sector in self.available_sectors:
                sector_tokens = self.tokenize(sector)
                if any(token in sector_tokens for token in tokens):
                    specified_sectors.append(sector)

        for i in range(len(tokens)):
            if tokens[i] in self.negations:
                negation_active = True
                continue

            if (
                tokens[i] in self.stopwords
                and tokens[i] not in self.intensifiers
                and tokens[i] not in self.negations
            ):
                continue

            field_match = None
            for field_key, synonyms in self.field_mappings.items():
                if tokens[i] in synonyms or tokens[i] == field_key:
                    field_match = field_key
                    break

            if field_match:

                field_value = self.field_mappings[field_match][0]

                modifier_value = 0.0
                intensifier_value = 1.0

                for j in range(max(0, i - window_size), i):

                    if tokens[j] in self.intensifiers:
                        intensifier_value *= self.intensifiers[tokens[j]]
                    if tokens[j] in self.reverse_modifiers:
                        modifier_value = self.reverse_modifiers[tokens[j]]

                # if field_value == "overallRisk":
                #     print("hi1")
                #     print(modifier_value, intensifier_value)

                if modifier_value == 0.0:
                    if field_match in ["environmental", "social", "governance", "esg"]:
                        modifier_value = 1.0
                    elif field_match in ["risk", "controversy", "beta"]:
                        modifier_value = -1.0
                    else:
                        modifier_value = 1.0
                
                if field_match in ["risk", "controversy", "beta"]:
                    if modifier_value != 0.0:
                        modifier_value *= -1
                    else:
                        modifier_value = 1.0 # default to low risk

                if negation_active:
                    modifier_value *= -1
                    negation_active = False

                # if field_value == "overallRisk":
                #     print(modifier_value, intensifier_value)

                query_vector[field_value] = modifier_value * intensifier_value

        if specified_sectors:
            query_vector["specified_sectors"] = specified_sectors

        if sentiment_mentioned:
            query_vector["include_sentiment"] = True

        return dict(query_vector)

    def calculate_similarity(self, stock, query_vector):
        """
        Calculate the similarity score between a stock and the query vector
        """
        score = 0.0
        total_weight = 0.0

        if "specified_sectors" in query_vector:
            if (
                stock.get("GICS Sector", "").lower()
                not in query_vector["specified_sectors"]
            ):
                return 0.0

        for field, weight in query_vector.items():
            if field in ["specified_sectors", "include_sentiment"]:
                continue
            if field in stock and weight != 0.0:
                if field in ["overallRisk", "highestControversy", "beta"]:
                    if weight < 0:
                        field_value = 1 - float(stock[field])
                        score += abs(weight) * field_value
                    else:
                        score += weight * float(stock[field])
                else:
                    score += weight * float(stock[field])
                total_weight += abs(weight)

        if total_weight > 0:
            score = score / total_weight

        return score

    def calculate_sector_similarity(self, query, stock):
        """
        Calculate how well a stock's sector matches the query.
        Returns a score between 0 and 1, with higher values indicating better matches.
        """
        if not stock.get("GICS Sector"):
            return 0.0

        stock_sector = stock["GICS Sector"].lower()
        query_tokens = self.tokenize(query)

        for token in query_tokens:

            if token in self.sectors and (
                token == stock_sector
                or self.sector_mapping.get(token, "") == stock_sector
            ):
                return 1.0

            if (
                token in self.sector_mapping
                and self.sector_mapping[token] == stock_sector
            ):
                return 1.0

        sector_tokens = self.tokenize(stock_sector)
        matches = set(query_tokens).intersection(set(sector_tokens))
        if matches:
            return 0.8 * (len(matches) / len(sector_tokens))

        return 0.0

    def calculate_name_similarity(self, query, stock_name):
        """
        Calculate text similarity between query and stock name
        Returns a score between 0 and 1
        """
        query = query.lower()
        stock_name = stock_name.lower()

        query_tokens = [t for t in self.tokenize(query) if t not in self.stopwords]

        if not query_tokens:
            return 0.0

        direct_match = 0.0
        for token in query_tokens:
            if token in stock_name:
                direct_match += 1.0
        direct_match = direct_match / len(query_tokens) if query_tokens else 0.0

        stock_tokens = self.tokenize(stock_name)
        common_tokens = set(query_tokens).intersection(set(stock_tokens))
        token_similarity = (
            len(common_tokens) / len(query_tokens) if query_tokens else 0.0
        )

        return (direct_match * 0.6) + (token_similarity * 0.4)

    def calculate_content_similarity(self, query, stock):
        """
        Calculate overall similarity between query and stock metadata
        with priority: sector > name > symbol
        """

        symbol = stock.get("Symbol", "").lower()
        query_lower = query.lower()
        symbol_score = 1.0 if query_lower == symbol.lower() else 0.0

        sector_score = self.calculate_sector_similarity(query, stock)

        name_score = self.calculate_name_similarity(query, stock["Full Name"])

        if symbol_score > 0:
            return {"score": symbol_score, "match_type": "symbol_match"}

        elif sector_score > 0:
            return {"score": sector_score, "match_type": "sector_match"}

        elif name_score > 0:
            return {"score": name_score, "match_type": "name_match"}

        else:
            return {"score": 0.0, "match_type": "no_match"}

    def _rocchio_update(self, original_query, relevant_stocks, nonrelevant_stocks, features, alpha=1.0, beta=0.75, gamma=0.15):
        updated_query = {}

        for feature in features:
            orig_weight = original_query.get(feature, 0.0)

            rel_centroid = 0.0
            if relevant_stocks:
                rel_values = [float(stock.get(feature, 0.5)) for stock in relevant_stocks]
                if rel_values: 
                    rel_centroid = np.mean(rel_values)

            nonrel_centroid = 0.0
            if nonrelevant_stocks:
                nonrel_values = [float(stock.get(feature, 0.5)) for stock in nonrelevant_stocks]
                if nonrel_values:
                    nonrel_centroid = np.mean(nonrel_values)

            # Rocchio update formula
            updated_query[feature] = alpha * orig_weight + beta * rel_centroid - gamma * nonrel_centroid

        # Carry over non-feature keys unchanged
        for key, value in original_query.items():
            if key not in features:
                updated_query[key] = value

        return updated_query

    def rank_stocks(self, query_text):
        """
        Rank stocks based on how well they match the query with priority:
        1. ESG terms (highest)
        2. Sector names (middle)
        3. Stock names (lowest)
        """
        if isinstance(query_text, dict):
            query_vector = query_text
        else:
            query_vector = self.parse_query(query_text)

        has_esg_keywords = False
        for field, weight in query_vector.items():
            if (
                field not in ["specified_sectors", "include_sentiment"]
                and weight != 0.0
            ):
                has_esg_keywords = True
                break

        scores = []
        for stock in self.normalized_data:
            if has_esg_keywords:

                score = self.calculate_similarity(stock, query_vector)
                match_type = "esg_factors"
            else:

                result = self.calculate_content_similarity(query_text, stock)
                score = result["score"]
                match_type = result["match_type"]

            if score > 0:
                stock_symbol = stock["Symbol"]
                original_stock = self.original_stocks_map.get(stock_symbol, {})
                result = {
                    "symbol": stock_symbol,
                    "name": stock["Full Name"],
                    "score": score,
                    "sector": stock.get("GICS Sector", "Unknown"),
                    "match_type": match_type,
                    "environmentScore": float(stock.get("environmentScore", 0)),
                    "socialScore": float(stock.get("socialScore", 0)),
                    "governanceScore": float(stock.get("governanceScore", 0)),
                    "totalEsg": float(stock.get("totalEsg", 0)),
                    "overallRisk": int(float(original_stock.get("overallRisk", 0))),
                }

                ticker = stock["Symbol"]
                if ticker in self.sentiment_data:
                    sentiment_info = self.sentiment_data[ticker]
                    result["sentiment"] = {
                        "description": sentiment_info["sentiment"]["description"],
                        "positive_percentage": sentiment_info["sentiment"][
                            "positive_percentage"
                        ],
                        "negative_percentage": sentiment_info["sentiment"][
                            "negative_percentage"
                        ],
                        "total_tweets": sentiment_info["sentiment"]["total_tweets"],
                        "summary": sentiment_info["summary"],
                    }

                scores.append(result)

        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores

    def rank_stocks_with_rocchio(self, query_text, alpha=1.0, beta=0.75, gamma=0.15):
        """
        Ranks stocks using an initial ranking followed by Rocchio feedback.
        """
        initial_ranked_results = self.rank_stocks(query_text)

        if not initial_ranked_results or len(initial_ranked_results) < 2:
            # Not enough initial results for Rocchio feedback
            return initial_ranked_results

        original_query = self.parse_query(query_text)
        feature_keys = [
            k for k, v in original_query.items()
            if k not in ["specified_sectors", "include_sentiment"] and isinstance(v, (int, float)) and v != 0.0
        ]

        if not feature_keys:
            # No weighted numerical factors in the query for Rocchio feedback
            return initial_ranked_results 

        # Split initial results and get corresponding stock data
        mid_index = len(initial_ranked_results) // 2
        relevant_symbols = {res["symbol"] for res in initial_ranked_results[:mid_index]}
        nonrelevant_symbols = {res["symbol"] for res in initial_ranked_results[mid_index:]}

        relevant_stocks = [self.normalized_stocks_map[sym] for sym in relevant_symbols if sym in self.normalized_stocks_map]
        nonrelevant_stocks = [self.normalized_stocks_map[sym] for sym in nonrelevant_symbols if sym in self.normalized_stocks_map]

        # Update the query vector 
        updated_query = self._rocchio_update(
            original_query,
            relevant_stocks,
            nonrelevant_stocks,
            feature_keys,
            alpha=alpha,
            beta=beta,
            gamma=gamma
        )

        return self.rank_stocks(updated_query)

def parse_json_file(file_path):
    """Parse a JSON file into a list of stock objects"""
    with open(file_path, "r") as file:
        content = file.read()
    return load_stock_data(content)


def load_stock_data(json_text):
    """Parse JSON text into a list of stock objects"""
    cleaned_json = json_text.strip()
    if cleaned_json.startswith("{"):
        cleaned_json = "[" + cleaned_json
    elif not cleaned_json.startswith("["):
        cleaned_json = "[" + cleaned_json
    if not cleaned_json.endswith("]"):
        cleaned_json = cleaned_json + "]"
    cleaned_json = re.sub(r",\s*}", "}", cleaned_json)
    cleaned_json = re.sub(r",\s*]", "]", cleaned_json)
    try:
        return json.loads(cleaned_json)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")

        if "," in cleaned_json:

            last_comma_index = cleaned_json.rindex(",")
            last_bracket_index = cleaned_json.rindex("]")
            if last_comma_index > last_bracket_index:
                cleaned_json = (
                    cleaned_json[:last_comma_index]
                    + cleaned_json[last_comma_index + 1 :]
                )

        try:
            return json.loads(cleaned_json)
        except json.JSONDecodeError as e:
            print(f"Still cannot parse JSON after cleanup: {e}")

            print("Attempting to parse individual objects...")
            try:
                parts = re.split(r"},\s*{", cleaned_json.strip("[]"))
                result = []
                for i, part in enumerate(parts):

                    if not part.startswith("{"):
                        part = "{" + part
                    if not part.endswith("}"):
                        part = part + "}"

                    try:
                        obj = json.loads(part)
                        result.append(obj)
                    except json.JSONDecodeError:
                        print(f"Could not parse object {i+1}")

                if result:
                    print(
                        f"Successfully parsed {len(result)} out of {len(parts)} objects"
                    )
                    return result
                else:
                    print("Could not parse any objects, returning empty list")
                    return []
            except Exception as e:
                print(f"Error during manual parsing: {e}")
                return []


def load_sentiment_data(file_path):
    """Load sentiment data from a JSON file"""
    with open(file_path, "r") as file:
        return json.load(file)
