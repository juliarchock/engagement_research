from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, json, time, progressbar

from emoji import demojize
from nltk.tokenize import TweetTokenizer


tokenizer = TweetTokenizer()


def normalizeToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token


def normalizeTweet(tweet):
    tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
    normTweet = " ".join([normalizeToken(token) for token in tokens])

    normTweet = (
        normTweet.replace("cannot ", "can not ")
        .replace("n't ", " n't ")
        .replace("n 't ", " n't ")
        .replace("ca n't", "can't")
        .replace("ai n't", "ain't")
    )
    normTweet = (
        normTweet.replace("'m ", " 'm ")
        .replace("'re ", " 're ")
        .replace("'s ", " 's ")
        .replace("'ll ", " 'll ")
        .replace("'d ", " 'd ")
        .replace("'ve ", " 've ")
    )
    normTweet = (
        normTweet.replace(" p . m .", "  p.m.")
        .replace(" p . m ", " p.m ")
        .replace(" a . m .", " a.m.")
        .replace(" a . m ", " a.m ")
    )

    return " ".join(normTweet.split())

tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")
print(model.config.num_labels)
model.eval()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

def analyze_sentiment_bert(tweets: list[str]) -> list[dict]:
    sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    norm_tweets = [normalizeTweet(t["text"]) for t in tweets]

    inputs = tokenizer(norm_tweets, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)

    # Get prediction
    probs = torch.softmax(outputs.logits, dim=-1)
    predicted = torch.argmax(probs, dim=-1)

    results = []

    for i in range(len(tweets)):
        result = {
            "politician": tweets[i]["politician"],
            "text": tweets[i]["text"],
            "predicted_sentiment": sentiment_labels[predicted[i].item()],
            "confidence_scores": {
                "negative": probs[i][0].item(),
                "neutral": probs[i][1].item(),
                "positive": probs[i][2].item()
            },
            "date_published": tweets[i]["time"],
        }
        results.append(result)
    
    return results
try:
    files = {
        "Data/AmyKlobuchar.json": "Amy Klobuchar (D-MN)",
        "Data/TinaSmith.json": "Tina Smith (D-MN)",
        "Data/BettyMcCollum.json": "Betty McCollum (D-MN)",
        "Data/IlhanOmar.json": "Ilhan Omar (D-MN)",
        "Data/AngieCraig.json": "Angie Craig (D-MN)",
        "Data/DeanPhillips.json": "Dean Phillips (D-MN)",
        "Data/TomEmmer.json": "Tom Emmer (R-MN)"
    }
    start_time = time.time()
    all_data = []
    for file in files:
        data = json.load(open(file))
        politician = files[file]
        for d in data:
            all_data.append({
                "text": d["Text"],
                "time": d["Time"],
                "politician": politician
            })
    BATCH_SIZE = 500
    progressbar = progressbar.ProgressBar(max_value=len(all_data))
    outputs = []

    for i in range(0, len(all_data), BATCH_SIZE):
        batch = all_data[i:i+BATCH_SIZE]
        results = analyze_sentiment_bert(batch)
        outputs.extend(results)
        progress = min(i + BATCH_SIZE, len(all_data))
        progressbar.update(progress)
    time_taken = time.time() - start_time
    print(f"Time taken: {time_taken / 60:.2f} minutes")

    with open("output.json", "w") as f:
        json.dump(outputs, f, indent=4)
except KeyboardInterrupt:
    print("Process interrupted by user.")


# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

# # Example dataset: Replace this with your actual sentiment analysis results
# # Convert to DataFrame
# df = pd.DataFrame(outputs)

# df['negative'] = df['confidence_scores'].apply(lambda x: x['negative'])
# df['positive'] = df['confidence_scores'].apply(lambda x: x['positive'])


# # Calculate neutrality threshold: If confidence scores are close, it's neutral
# df['neutral'] = abs(df['negative'] - df['positive']) < 0.07
# df['sentiment'] = df.apply(lambda row: "neutral" if row['neutral'] else ("positive" if row['positive'] > row['negative'] else "negative"), axis=1)

# # Plot probability distributions
# plt.figure(figsize=(10, 5))
# sns.histplot(df['negative'], bins=20, label='Negative Confidence', kde=True, color='red', alpha=0.5)
# sns.histplot(df['positive'], bins=20, label='Positive Confidence', kde=True, color='blue', alpha=0.5)
# plt.axvline(0.5, color='gray', linestyle="dashed", label="Neutral Threshold")
# plt.legend()
# plt.title("Sentiment Confidence Score Distributions")
# plt.xlabel("Confidence Score")
# plt.ylabel("Frequency")
# plt.show()





