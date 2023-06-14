import requests
import json
import datetime

import pandas as pd
import numpy as np

def get_gf_index():

    """Get the Fear and Greed Index from alternative.me."""

    url = "https://api.alternative.me/fng/?limit=0"

    response = json.loads(requests.get(url).text) # get the response from the API
    fg_index_df = pd.DataFrame(response["data"]) # convert the response to a dataframe
    fg_index_df["timestamp"] = fg_index_df["timestamp"].apply(lambda x: datetime.datetime.fromtimestamp(int(x))).dt.date # convert the timestamp to date
    fg_index_df.set_index("timestamp", inplace=True) # set timestamp as index

    val_class = {"Extreme Fear": 0, "Fear": 1, "Neutral": 2, "Greed": 3, "Extreme Greed": 4} # define the value classification
    fg_index_df["value_classification"] = np.array(list(map(lambda x: val_class[x], fg_index_df["value_classification"]))) # convert the value classification to numeric

    return fg_index_df[["value_classification"]].rename(columns={"value_classification":"gf-index"}) 

