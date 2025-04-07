from query_system import EthicalInvestmentQuerySystem, load_stock_data
import os
import glob
from flask import Flask

# to run, run "python -m backend.helpers.test"


def format_result(result, index, query_vector):
    """Format a single result with score calculation details"""

    output = [
        f"{index}. {result['symbol']} - {result['name']} (Score: {result['score']:.3f})",
        f"   Sector: {result['sector']}",
        f"   ESG Scores: Environmental={result['environmentScore']:.2f}, "
        f"Social={result['socialScore']:.2f}, Governance={result['governanceScore']:.2f}",
        f"   Total ESG: {result['totalEsg']:.2f}, Risk Level: {result['overallRisk']}",
    ]

    return "\n".join(output)


def run_query(query_system, stocks_data, query, top_n=3):
    """Run a query and display the top N results with scoring details"""
    print(f"\n{'='*80}")
    print(f'QUERY: "{query}"')
    print(f"{'='*80}")

    query_vector = query_system.parse_query(query)

    print("\nQUERY VECTOR:")
    if not query_vector:
        print("  No specific criteria identified in query")
    else:
        for field, weight in query_vector.items():
            if isinstance(weight, (int, float)):
                print(f"  {field}: {weight:.2f}")
            else:
                print(f"  {field}: {weight}")

    results = query_system.rank_stocks_with_rocchio(query)

    if not results:
        print("\nNo matching stocks found for this query.")
        return

    print(f"\nTop {min(top_n, len(results))} matches:")
    for i, result in enumerate(results[:top_n]):
        print(format_result(result, i + 1, query_vector))
        print("-" * 40)


def main():

    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Move up one level to the backend folder and join the filename
    json_file_path = os.path.join(current_directory, "..", "sp500.json")
    json_file_path = os.path.abspath(json_file_path)

    with open(json_file_path, "r") as file:
        json_text = file.read()
        stocks_data = load_stock_data(json_text)
        # print(f"Successfully loaded data for {len(stocks_data)} stocks.")

    query_system = EthicalInvestmentQuerySystem(stocks_data)

    example_queries = [
        # "Find stocks with low risk and high environmental scores",
        # "High Governance",
        "Low Risk",
        "High Risk",
        # "Show me companies with excellent governance and social responsibility",
        # "I want to invest in companies that are environmentally friendly with minimal controversy",
        # "Which companies have the best overall ESG ratings?",
        # "Find technology companies with strong social scores",
        # "I need stocks with very low risk that are still good for the environment",
        # "Show me health care companies with strong governance",
        # "Which companies have the highest environmental scores but also low risk?",
        # "Find companies with extremely good social responsibility",
        # "I want to invest in utilities with strong environmental practices",
    ]

    for query in example_queries:
        run_query(query_system, stocks_data, query)


if __name__ == "__main__":
    main()
