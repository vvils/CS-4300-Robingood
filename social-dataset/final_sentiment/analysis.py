import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
import warnings
warnings.filterwarnings("ignore")
import pandas_datareader.data as web
import datetime as dt
import azureml.core
from azureml.core import Workspace
from yahoo_fin import news
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
import time

# Load AzureML Workspace
ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))

'''
# Get S&P 500 tickers
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)
    sp500_df = table[0]
    sp500_tickers = sp500_df["Symbol"].str.replace(".", "-").tolist()
    return sorted(sp500_tickers)

sp500_list = get_sp500_tickers()
print(f"Loaded {len(sp500_list)} tickers.")

# Initialize empty list for all news
all_rows = []


# Main Loop
for ticker in sp500_list:
    try:
        all_news_list = news.get_yf_rss(ticker)
        for i, article in enumerate(all_news_list):
            all_rows.append({'Ticker': ticker, 'NewsNum': i, 'Value': article['summary']})

    except Exception as e:
        print(f"Error fetching {ticker}: {e}. Sleeping for 5 minutes...")
        time.sleep(300)  # Wait 5 minutes
        try:
            all_news_list = news.get_yf_rss(ticker)
            for i, article in enumerate(all_news_list):
                all_rows.append({'Ticker': ticker, 'NewsNum': i, 'Value': article['summary']})
        except Exception as e2:
            print(f"Second error on {ticker}: {e2}. Skipping...")

    # Optional: Save partial progress every 50 tickers
    if sp500_list.index(ticker) % 50 == 0:
        temp_df = pd.DataFrame(all_rows)
        temp_df.to_csv("partial_sp500_news.csv", index=False)
        print(f"Saved partial progress at {ticker}")

# Final DataFrame
combined_snp500_news = pd.DataFrame(all_rows)

# Save Final CSV
combined_snp500_news.to_csv("combined_sp500_news.csv", index=False)
print("Saved all news to combined_sp500_news.csv")
'''

'''
input_file = "combined_sp500_news.csv"
output_file = "combined_sp500_news_with_sentiment.csv"

df = pd.read_csv(input_file)

# If output already exists, resume from there
if os.path.exists(output_file):
    df_out = pd.read_csv(output_file)
    processed_indices = set(df_out.index)
else:
    df_out = df.copy()
    df_out['Sentiment'] = None
    processed_indices = set()

load_dotenv()
endpoint = "https://sentiment-text-analysis-v2.cognitiveservices.azure.com/"
key = os.getenv("AZURE_KEY")
credential = AzureKeyCredential(key)
client = TextAnalyticsClient(endpoint=endpoint, credential=credential)

def analyze_sentiment(text):
    try:
        text = text[:5000]  
        response = client.analyze_sentiment(documents=[text])[0]
        return response.sentiment
    except Exception as e:
        print(f"Error analyzing text: {e}")
        return "error"

for idx, row in df.iterrows():
    if idx in processed_indices:
        continue  
    
    sentiment = analyze_sentiment(row['Value'])
    df_out.at[idx, 'Sentiment'] = sentiment
    
    df_out.to_csv(output_file, index=False)
    print(f"Processed {idx+1}/{len(df)}: {row['Ticker']} - Sentiment: {sentiment}")
'''

# Load CSV
df = pd.read_csv("combined_sp500_news_with_sentiment.csv")

# Initialize output list
sentiment_summary = []

# Define sentiments
sentiment_labels = ['positive', 'negative', 'neutral', 'mixed']

# Group by ticker
grouped = df.groupby('Ticker')

for ticker, group in grouped:
    total = len(group)
    counts = group['Sentiment'].value_counts().to_dict()
    
    percentages = {}
    for label in sentiment_labels:
        count = counts.get(label, 0)
        percentages[label] = round((count / total) * 100, 1)  # One decimal place
        
    ticker_sentiment = {
        "ticker": ticker,
        "sentiment": {
            "positive_percentage": percentages['positive'],
            "negative_percentage": percentages['negative'],
            "mixed_percentage": percentages['mixed'],
            "neutral_percentage": percentages['neutral'],
            "total_news": total
        }
    }
    
    sentiment_summary.append(ticker_sentiment)

with open('final_sentiment_summary.json', 'w') as f:
    json.dump(sentiment_summary, f, indent=2)