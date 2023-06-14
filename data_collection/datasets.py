from typing import Union
import math
import os
import numpy as np
import pandas as pd
from constants import *

OTPUT_DIR = "./data_collection/datasets" # define the output directory

def split_dataset(features:np.ndarray, labels:np.ndarray, task:str) -> None:

    """Split the dataset into training and test sets."""

    train_len = int(len(features)*TRAINING_SPLIT) # calculate the length of the training set

    if task == "class":
        labels = labels_classification(labels, features) # convert the labels to classification labels

    training_features = features[:train_len] # split the features into training and test sets
    training_labels = labels[:train_len] # split the labels into training and test sets

    test_features = features[train_len:] # split the features into training and test sets
    test_labels = labels[train_len:] # split the labels into training and test sets

    save_dataset(training_features, "features", "training", task) # save the training features
    save_dataset(training_labels, "labels", "training", task) # save the training labels
    save_dataset(test_features, "features", "test", task) # save the test features
    save_dataset(test_labels, "labels", "test", task) # save the test labels

def save_dataset(data:np.ndarray, type:str, split:str|None="full", task:str|None="full") -> None:

    """Save the dataset."""

    if not os.path.exists(OTPUT_DIR+"/"+split+"/"+task+"/"+type+".npy"): # if the file doesn't exist, create it
        os.makedirs(filepath:=os.path.join(OTPUT_DIR, split, task), exist_ok=True) # create a folder for the data
        with open(OTPUT_DIR+"/"+split+"/"+task+"/"+type+".npy", "wb") as f: # save the data
            np.save(f, data)


def create_dataset(symbol:str, periods:int|None=PERIODS, task:str|None="reg") -> Union[np.ndarray[np.ndarray], np.ndarray]:

    """Splits dataset into training, validation and testing ones."""

    df = pd.read_csv("data_collection/datasets/sentiments/data_with_sentiments.csv") # read the data
    df.set_index("date", inplace=True) # set the index to the date
    del df["Unnamed: 0"]
    df = df.sort_index(ascending=True) # sort the data by date in ascending order
    columns = np.delete(df.columns, np.where(df.columns == "label")) # get the columns of the dataframe
    columns = np.append(columns, "label") # append the label column to the columns
    df=df[columns] # reorder the columns
    df["sentiment"]=df["sentiment"].apply(lambda x: int(x) if len(x)==1 else int(0)) # convert the sentiment to numeric
    df_changes=(df.iloc[:, :-1]-df.iloc[:, :-1].shift(1))/df.iloc[:, :-1] # calculate the changes in the data
    df_changes["growth"] = df["growth"]/df["open"] # calculate the change of growth
    df_changes["label"] = df["label"]/df["close"]  # calculate the change of label
    df_changes[["sentiment", "gf-index"]] = df[["sentiment", "gf-index"]] # add the sentiment and gf-index columns
    df_changes = df_changes.sort_index(ascending=True).dropna() # sort the data by date in ascending order and drop the NaN values

    df_changes.to_csv("data_collection/datasets/complete_dataset.csv") # save the dataset

    df_y = df_changes["label"] # get the labels
    df_x = df_changes.iloc[:, :-1] # get the features

    row_number = len(df) # get the number of rows in the dataframe
    print(df_x.iloc[-2, 7])

    df_x_np = df_x.to_numpy() # convert the features to numpy
    df_y_np = df_y.to_numpy() # convert the labels to numpy
    features = []
    labels = []

    for row_idx in range(row_number-periods): # iterate through the rows
        features_data = df_x_np[row_idx:row_idx+periods] # get the features for the period
        labels_data = df_y_np[row_idx+periods-1] # get the labels
        features.append(features_data) # append the features
        labels.append(labels_data) # append the labels

    print("Features and labels are created.")

    features = np.array(features) # convert the features to numpy
    labels = np.array(labels) # convert the labels to numpy

    save_dataset(features, "features") # save the features
    save_dataset(labels, "labels") # save the labels

    split_dataset(features, labels, task=task) # split the dataset into training and test sets
    
def get_stats(sample, multiplier=1):

    """Get the statistics of the sample."""

    sample_max = np.max(sample)*multiplier # get the maximum value of the sample
    sample_min = np.min(sample)*multiplier # get the minimum value of the sample
    sample_diff = (sample_max - sample_min) # get the difference between the maximum and minimum values of the sample
    sample_mean = np.mean(sample)*multiplier # get the mean of the sample
    sample_sigma = np.std(sample)*multiplier # get the standard deviation of the sample

    return sample_max, sample_min, sample_diff, sample_mean, sample_sigma

def classify_label(sample:list, label:float) -> float:

    """Classify the label."""

    _, _, _, sample_mean, sample_sigma = get_stats(sample) # get the statistics of the sample

    def normalize(sample_mean):
        """Normalize the sample."""
        return sample - sample_mean
    
    print(f"Sample mean: {sample_mean}, sample sigma: {sample_sigma}")
    sample_norm = normalize(sample_mean) # normalize the sample
    _, _, _, sample_mean_norm, sample_sigma_norm = get_stats(sample_norm, multiplier=1) # get the statistics of the normalized sample
    print(f"Sample mean norm: {sample_mean_norm}, sample sigma norm: {sample_sigma_norm}")

    if label > sample_sigma_norm+sample_mean_norm: # if the label is greater than the mean plus the standard deviation
        return BUY_MAX
    elif label > sample_mean_norm+sample_sigma_norm/2: # if the label is greater than the mean plus half of the standard deviation
        return BUY
    elif label > sample_mean_norm-sample_sigma_norm/2: # if the label is greater than the mean minus half of the standard deviation
        return HOLD
    elif label > sample_mean_norm-sample_sigma_norm: # if the label is greater than the mean minus the standard deviation
        return SELL
    else: # if the label is less than the mean minus the standard deviation
        return SELL_MAX

def labels_classification(labels:np.ndarray, features:np.ndarray) -> np.ndarray:

    """Classifies labales for a dataset by standard deviation using percentile."""

    for i, label in enumerate(labels):
        labels[i] = classify_label(features[i, :, 7], label) # classify the label
    print(labels)
    
    return labels

if __name__ == "__main__":
    create_dataset(ETH_TOKEN, task="class")
    print("Datasets are created.")