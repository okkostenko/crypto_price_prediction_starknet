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

def split_dataset(features:np.ndarray, labels:np.ndarray) -> None:
    train_len = int(len(features)*TRAINING_SPLIT)
    print(train_len)

    training_features = features[:train_len]
    training_labels = labels[:train_len]

    test_features = features[train_len:]
    test_labels = labels[train_len:]

    save_dataset(training_features, "features", "training")
    save_dataset(training_labels, "labels", "training")
    save_dataset(test_features, "features", "test")
    save_dataset(test_labels, "labels", "test")

def save_dataset(data:np.ndarray, type:str, split:str|None="full") -> None:
    if not os.path.exists(OTPUT_DIR+"/"+split+"/"+type+".npy"):
        os.makedirs(filepath:=os.path.join(OTPUT_DIR, split), exist_ok=True)
        with open(OTPUT_DIR+"/"+split+"/"+type+".npy", "wb") as f:
            np.save(f, data)


def create_dataset(symbol:str, periods:int|None=PERIODS, task:str|None="reg") -> Union[np.ndarray[np.ndarray], np.ndarray]:

    """Splits dataset into training, validation and testing ones."""

    df = pd.read_csv(f'./data_collection/datasets/full/data_with_statistics_{symbol}_{TIMEFRAME}_full.csv')
    df_y = df["label"]
    df_x = df.iloc[:, :-1]

    row_number = len(df)

    df_x_np = df_x.to_numpy()
    df_y_np = df_y.to_numpy()
    features = []
    labels = []

    for row_idx in range(row_number-periods):
        features_data = df_x_np[row_idx:row_idx+periods]
        labels_data = df_y_np[row_idx+periods-1]
        features.append(features_data)
        labels.append(labels_data)

    if task == "class":
        labels = labels_classification(labels)

    save_dataset(features, "features")
    save_dataset(labels, "labels")

    split_dataset(features, labels)
    


def labels_classification(labels:np.ndarray) -> np.ndarray:

    """Classifies labales for a dataset by standard deviation using percentile."""
    labels_min = np.min(labels)
    labels_max = np.max(labels)
    labels_diff = labels_max - labels_min
    for i, label in enumerate(labels):
        # if label > np.percentile(labels, 84.4):
        if label > labels_diff*0.844+labels_min:
            print(np.percentile(labels, 84.4))
            labels[i] = BUY_MAX
        # elif label > np.percentile(labels, 66.7):
        elif label > labels_diff*0.667+labels_min:
            labels[i] = BUY
        # elif label > np.percentile(labels, 33.7):
        elif label > labels_diff*0.337+ labels_min:
            labels[i] = HOLD
        # elif label > np.percentile(labels, 16.7):
        elif label > labels_diff*0.167+labels_min:
            labels[i] = SELL
        else:
            labels[i] = SELL_MAX
    
    return labels

if __name__ == "__main__":
    create_dataset(BTC_TOKEN)
    create_dataset(BTC_TOKEN, task="class")
    print("Datasets are created.")