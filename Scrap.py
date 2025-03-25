import requests
import pandas as pd

# 🔑 Replace with your actual Bearer Token
BEARER_TOKEN = ""
SAVE_PATH = "/content/drive/My Drive/twitter_sentiment_data4.csv"

# 🚀 Function to fetch tweets (single request, max 50 results)
def get_tweets():
    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    query = "(I feel so lost OR empty OR hopeless) OR (I'm so stressed OR anxious OR nervous) OR (Having a normal day OR feeling okay)"
    params = {
        "query": query,
        "tweet.fields": "text",
        "max_results": 50  # Fetch up to 50 tweets
    }

    response = requests.get(url, headers=headers, params=params, verify=False)

    if response.status_code == 200:
        return [tweet["text"] for tweet in response.json().get("data", [])]
    else:
        print(f"❌ Error {response.status_code}: {response.text}")
        return []

# Fetch tweets
tweets = get_tweets()

# Assign sentiment labels (basic keyword-based classification)
sentiments = []
for tweet in tweets:
    if any(word in tweet.lower() for word in ["lost", "empty", "hopeless"]):
        sentiments.append("depressed")
    elif any(word in tweet.lower() for word in ["stressed", "anxious", "nervous"]):
        sentiments.append("anxious")
    else:
        sentiments.append("neutral")

# Save dataset to Google Drive
df = pd.DataFrame({"tweet": tweets, "sentiment": sentiments})
df.to_csv(SAVE_PATH, index=False)

print(f"✅ Scraping complete. Data saved to Google Drive at: {SAVE_PATH}")
