import pandas as pd
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, pipeline, TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer
from bs4 import BeautifulSoup
import requests
import re
import csv

# print("Loading Summarization Models...")
# #summarization model
# summarization_model_name = "human-centered-summarization/financial-summarization-pegasus"
# summarization_tokanizer=PegasusTokenizer.from_pretrained(summarization_model_name)
# summarization_model=PegasusForConditionalGeneration.from_pretrained(summarization_model_name)
# # sentiment = pipeline("sentiment-analysis")

print("Loading Sentiment Models...")
sentiment_model_name = "ElKulako/cryptobert"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name, use_fast=True)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name, num_labels = 3)
sentiment_pipe = TextClassificationPipeline(model=sentiment_model, tokenizer=sentiment_tokenizer, max_length=64, truncation=True, padding = 'max_length')


# #summarize articles
# print('Summarizing articles...')
# def summarize_articles(articles):
#     summaries=[]
#     for i, article in enumerate(articles):
#         print("Summarizing article", i, "of", len(articles))
#         article = article[:500]
#         print(article)
#         try:
#             input_ids=summarization_tokanizer.encode(article, return_tensors='pt')
#             output=summarization_model.generate(input_ids, max_length=55, num_beams=5, early_stopping=True)
#             summary=summarization_tokanizer.decode(output[0], skip_special_tockens=True)
#         except:
#             summary = None
#         summaries.append(summary)
#         print(summary)
#     return summaries

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
    # summaries = summarize_articles(df['content'])
    # df["fin_summary"] = summaries
    # df["fin_summary"].fillna(df["summary"], inplace=True)
    # df.to_csv("data_collection/datasets/sentiments/news_with_financial_summary.csv", index=False)
    # print("Data Saved!")
    df["fin_summary"] = df["content"].apply(lambda x: " ".join(x.split(" ")[:100])+" "+" ".join(x.split(" ")[-250:]))
    print("Calculating sentiments...")
    sentiments, sentiment_scores = calculate_sentiments(df["summary"])
    df["sentiment"] = sentiments
    df["sentiment_score"] = sentiment_scores
    df.to_csv("data_collection/datasets/sentiments/news_with_financial_summary_and_sentiment.csv", index=False)
    print("Data Saved!")


if __name__ == "__main__":
    main()

# #exporting result
# print('Exporting results...')
# def output(urls, summaries, scores):
#     output=[]
#     for ticker in monitored_tickers:
#         for i in range(len(summaries[ticker])):
#             ticker_output=[
#                 ticker, 
#                 summaries[ticker][i],
#                 scores[ticker][i]['label'],
#                 scores[ticker][i]['score'],
#                 urls[ticker][i]]
#             output.append(ticker_output)
#     return(output)

# final_output=output(cleaned_urls, summaries, scores)
# final_output.insert(0, ['Ticker', 'Summary', 'Sentiment Score', 'Url'])

# with open('financial_summaries.csv', mode='w', newline='') as f:
#     csv_writer=csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     csv_writer.writerows(final_output)

# print('Done')


