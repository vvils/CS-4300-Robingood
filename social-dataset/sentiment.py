import json
import requests
import time
from collections import defaultdict

def get_tweet_text_from_embed(tweet_id):
    url = f"https://cdn.syndication.twimg.com/tweet-result?id={tweet_id}&token=a"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("text", "[No text found]")
        else:
            print(f"[{tweet_id}] HTTP {response.status_code}")
            return "[Error fetching tweet]"
    except Exception as e:
        print(f"[{tweet_id}] Exception: {e}")
        return "[Exception occurred]"

def clean_json(input_path, output_path):
  with open(input_path, "r") as infile:
      data = json.load(infile)

  cleaned_data = [item for item in data if item.get("Text") != "[No text found]"]

  with open(output_path, "w") as outfile:
      json.dump(cleaned_data, outfile, indent=2)

  print(f"Cleaned {len(data) - len(cleaned_data)} items. Saved to {output_path}")

def summarize_sentiments(input_path, output_path):
    with open(input_path, "r") as f:
        data = json.load(f)

    sentiment_counts = defaultdict(lambda: {"POSITIVE": 0, "NEGATIVE": 0})

    for item in data:
        ticker = item.get("Target_Ticker")
        sentiment = item.get("Sentiment")
        if sentiment in ["POSITIVE", "NEGATIVE"]:
            sentiment_counts[ticker][sentiment] += 1

    summary = {}
    for ticker, counts in sentiment_counts.items():
        total = counts["POSITIVE"] + counts["NEGATIVE"]
        if total > 0:
            summary[ticker] = {
                "Positive_Percentage": round(100 * counts["POSITIVE"] / total, 2),
                "Negative_Percentage": round(100 * counts["NEGATIVE"] / total, 2)
            }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Sentiment summary saved to {output_path}")

def main():
    with open("TweetFinSent_Test.json", "r") as f:
        tweets = json.load(f)

    for i, tweet in enumerate(tweets):
        tweet_id = tweet["Tweet_ID"]
        text = get_tweet_text_from_embed(tweet_id)
        tweet["Text"] = text
        print(f"[{i+1}/{len(tweets)}] {tweet_id} → {text[:80]}")
        time.sleep(0.5)  

    with open("TweetFinSent_WithText.json", "w") as f:
        json.dump(tweets, f, indent=2)

    print("Done. Output saved to TweetFinSent_WithText.json")

if __name__ == "__main__":
  #main()
  #clean_json("TweetFinSent_WithText.json", "TweetFinSent_Cleaned.json")
  summarize_sentiments("TweetFinSent_Cleaned.json", "TickerSentimentSummary.json")

