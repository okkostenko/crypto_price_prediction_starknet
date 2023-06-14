import pandas as pd
from typing import Tuple
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer

# load the sentiment model
print("Loading Sentiment Models...")
sentiment_model_name = "ElKulako/cryptobert" # set the model name to the CryptoBERT model
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name, use_fast=True) # load the tokenizer
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name, num_labels = 3) # load the model
sentiment_pipe = TextClassificationPipeline(model=sentiment_model, tokenizer=sentiment_tokenizer, max_length=64, truncation=True, padding = 'max_length') # create a pipeline


def calculate_sentiments(summaries: pd.Series) -> Tuple[pd.Series, pd.Series]:

    """Calculate sentiments for the summaries of the articles."""

    sentiments = []
    sentiment_scores = []

    # loop over all summaries
    for i, summary in enumerate(summaries):
        print("Calculating sentiment for article", i, "of", len(summaries))
        print(summary)
        sentiment = sentiment_pipe(list(summary))[0] # run the sentiment analysis pipeline on the summary
        sentiments.append(sentiment["label"]) # get the sentiment label
        sentiment_scores.append(sentiment["score"]) # get the sentiment score

        print(sentiment)
        print("\n")

    return sentiments, sentiment_scores

def main():

    """Main function."""

    print("Reading data...")
    df = pd.read_csv("data_collection/datasets/sentiments/news_database_clean.csv") # read the news data

    print("Summarizing articles...")
    df["fin_summary"] = df["content"].apply(lambda x: " ".join(x.split(" ")[:100])+" "+" ".join(x.split(" ")[-250:])) # summarize the articles
    print("Calculating sentiments...")
    sentiments, sentiment_scores = calculate_sentiments(df["summary"]) # calculate the sentiments for the summaries
    
    df["sentiment"] = sentiments # add the sentiments to the data
    df["sentiment_score"] = sentiment_scores # add the sentiment scores to the data
    df.to_csv("data_collection/datasets/sentiments/news_with_financial_summary_and_sentiment.csv", index=False) # save the data
    print("Data Saved!")


if __name__ == "__main__":
    main()



