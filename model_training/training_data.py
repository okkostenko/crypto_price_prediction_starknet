from typing import Union
import math
import numpy as np
import pandas as pd
from data_collection.constants import PERIODS, TIMEFRAME, TRAINING_SPLIT, VALIDATION_SPLIT, TEST_SPLIT

def get_split(split_name:str) -> float:
    match split_name:
        case "train":
            return TRAINING_SPLIT
        case "validation":
            return VALIDATION_SPLIT
        case "test":
            return TEST_SPLIT
        case _:
            raise ValueError(f"There is no such split as {split_name}. Please input one of following: 'train', 'validation' or 'test'.")

def split_dataset(symbol:str, split_name:str, periods:int|None=PERIODS/2) -> Union[np.ndarray[np.ndarray], np.ndarray]:

    """Splits dataset into training, validation and testing ones."""

    df = pd.read_csv(f"/{symbol.lower()}/data/datasets/data_with_statistics_{symbol}_{TIMEFRAME}_full.csv")

    row_number = len(df)
    split = get_split(split_name)
    split_row_number = math.ceil((row_number-periods)*split)
    