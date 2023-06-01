from typing import Union
import math
import os
import numpy as np
import pandas as pd
from constants import *

OTPUT_DIR = "./data_collection/datasets"

# def get_split(split_name:str) -> float:
#     match split_name:
#         case "train":
#             return TRAINING_SPLIT
#         case "validation":
#             return VALIDATION_SPLIT
#         case "test":
#             return TEST_SPLIT
#         case _:
#             raise ValueError(f"There is no such split as {split_name}. Please input one of following: 'train', 'validation' or 'test'.")

def split_dataset(features:np.ndarray, labels:np.ndarray, task:str) -> None:
    train_len = int(len(features)*TRAINING_SPLIT)


    if task == "class":
        labels = labels_classification(labels, features)

    training_features = features[:train_len]
    training_labels = labels[:train_len]

    test_features = features[train_len:]
    test_labels = labels[train_len:]

    save_dataset(training_features, "features", "training", task)
    save_dataset(training_labels, "labels", "training", task)
    save_dataset(test_features, "features", "test", task)
    save_dataset(test_labels, "labels", "test", task)

def save_dataset(data:np.ndarray, type:str, split:str|None="full", task:str|None="full") -> None:
    if not os.path.exists(OTPUT_DIR+"/"+split+"/"+task+"/"+type+".npy"):
        os.makedirs(filepath:=os.path.join(OTPUT_DIR, split, task), exist_ok=True)
        with open(OTPUT_DIR+"/"+split+"/"+task+"/"+type+".npy", "wb") as f:
            np.save(f, data)


def create_dataset(symbol:str, periods:int|None=PERIODS, task:str|None="reg") -> Union[np.ndarray[np.ndarray], np.ndarray]:

    """Splits dataset into training, validation and testing ones."""

    # df = pd.read_csv(f'./data_collection/datasets/full/data_with_statistics_{symbol}_{TIMEFRAME}_full.csv')
    df = pd.read_csv("data_collection/datasets/sentiments/data_with_sentiments.csv")
    df.set_index("date", inplace=True)
    del df["Unnamed: 0"]
    df = df.sort_index(ascending=True)
    df_changes=(df.iloc[:, :-1]-df.iloc[:, :-1].shift(1))/df.iloc[:, :-1]
    df_changes["growth"] = df["growth"]/df["open"]
    df_changes["label"] = df["label"]/df["close"]
    df_changes[["Bearish", "Bullish", "Neutral", "gf-index"]] = df[["Bearish", "Bullish", "Neutral", "gf-index"]]
    df_changes = df_changes.sort_index(ascending=True)

    df_changes.to_csv("data_collection/datasets/dcomplete_dataset.csv")

    df_y = df_changes["label"]
    df_x = df_changes.iloc[:, :-1]

    row_number = len(df)
    print(df_x.iloc[-2, 7])

    df_x_np = df_x.to_numpy()
    df_y_np = df_y.to_numpy()
    features = []
    labels = []

    for row_idx in range(row_number-periods):
        features_data = df_x_np[row_idx:row_idx+periods]
        labels_data = df_y_np[row_idx+periods-1]
        features.append(features_data)
        labels.append(labels_data)
    print("Features and labels are created.")
    # features = np.array(features)[:, :, 1:]
    features = np.array(features)
    labels = np.array(labels)
    save_dataset(features, "features")
    save_dataset(labels, "labels")

    split_dataset(features, labels, task=task)
    
def get_stats(sample, multiplier=1):
    sample_max = np.max(sample)*multiplier
    sample_min = np.min(sample)*multiplier
    sample_diff = (sample_max - sample_min)
    sample_mean = np.mean(sample)*multiplier
    sample_sigma = np.std(sample)*multiplier
    return sample_max, sample_min, sample_diff, sample_mean, sample_sigma

def classify_label(sample:list, label:float) -> float:
    sample_max, sample_min, sample_diff, sample_mean, sample_sigma = get_stats(sample)
    def normalize(sample_max, sample_min, sample_diff, sample_mean, sample_sigma):
        return (sample - sample_mean) / sample_sigma

    sample_norm = normalize(sample_max, sample_min, sample_diff, sample_mean, sample_sigma)
    sample_max_norm, sample_min_norm, sample_diff_norm, sample_mean_norm, sample_sigma_norm = get_stats(sample_norm, multiplier=1)
    if label > (sample_sigma_norm+sample_mean_norm)*sample_sigma:
        return BUY_MAX
    elif label > (sample_mean_norm+sample_sigma_norm/2)*sample_sigma:
        return BUY
    elif label > (sample_mean_norm-sample_sigma_norm/2)*sample_sigma:
        return HOLD
    elif label > (sample_mean_norm-sample_sigma_norm)*sample_sigma:
        return SELL
    else:
        return SELL_MAX


def labels_classification(labels:np.ndarray, features:np.ndarray) -> np.ndarray:

    """Classifies labales for a dataset by standard deviation using percentile."""
    # labels_min = np.min(labels)
    # labels_max = np.max(labels)
    # labels_diff = labels_max - labels_min
    # for i, label in enumerate(labels):
    #     # if label > np.percentile(labels, 84.4):
    #     if label > labels_diff*0.8+labels_min:
    #         labels[i] = BUY_MAX
    #     # elif label > np.percentile(labels, 66.7):
    #     elif label > labels_diff*0.6+labels_min:
    #         labels[i] = BUY
    #     # elif label > np.percentile(labels, 33.7):
    #     elif label > labels_diff*0.4+ labels_min:
    #         labels[i] = HOLD
    #     # elif label > np.percentile(labels, 16.7):
    #     elif label > labels_diff*0.2+labels_min:
    #         labels[i] = SELL
    #     else:
    #         labels[i] = SELL_MAX
    # labels = labels[:10]
    for i, label in enumerate(labels):
        labels[i] = classify_label(features[i, :, 7], label)
    print(labels)
    
    return labels

if __name__ == "__main__":
    # create_dataset(ETH_TOKEN)
    create_dataset(ETH_TOKEN, task="class")
    print("Datasets are created.")