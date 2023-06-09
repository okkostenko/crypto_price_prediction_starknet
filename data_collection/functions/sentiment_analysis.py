import pandas as pd
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer


print("Loading Sentiment Models...")
sentiment_model_name = "ElKulako/cryptobert"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name, use_fast=True)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name, num_labels = 3)
sentiment_pipe = TextClassificationPipeline(model=sentiment_model, tokenizer=sentiment_tokenizer, max_length=64, truncation=True, padding = 'max_length')


def calculate_sentiments(summaries):
    sentiments = []
    sentiment_scores = []
    for i, summary in enumerate(summaries):
        print("Calculating sentiment for article", i, "of", len(summaries))
        print(summary)
        sentiment = sentiment_pipe(list(summary))[0]
        sentiments.append(sentiment["label"])
        sentiment_scores.append(sentiment["score"])
        print(sentiment)
        print("\n")
    return sentiments, sentiment_scores

def main():

    print("Reading data...")
    df = pd.read_csv("data_collection/datasets/sentiments/news_database_clean.csv")

    print("Summarizing articles...")
    df["fin_summary"] = df["content"].apply(lambda x: " ".join(x.split(" ")[:100])+" "+" ".join(x.split(" ")[-250:]))
    print("Calculating sentiments...")
    sentiments, sentiment_scores = calculate_sentiments(df["summary"])
    df["sentiment"] = sentiments
    df["sentiment_score"] = sentiment_scores
    df.to_csv("data_collection/datasets/sentiments/news_with_financial_summary_and_sentiment.csv", index=False)
    print("Data Saved!")


if __name__ == "__main__":
    main()



