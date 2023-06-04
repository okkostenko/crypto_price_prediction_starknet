import requests
import json
import datetime

import pandas as pd
import numpy as np

def get_gf_index():
    url = "https://api.alternative.me/fng/?limit=0"

    response = json.loads(requests.get(url).text)
    fg_index_df = pd.DataFrame(response["data"])
    fg_index_df["timestamp"] = fg_index_df["timestamp"].apply(lambda x: datetime.datetime.fromtimestamp(int(x))).dt.date
    fg_index_df.set_index("timestamp", inplace=True)

    val_class = {"Extreme Fear": 0, "Fear": 1, "Neutral": 2, "Greed": 3, "Extreme Greed": 4}
    fg_index_df["value_classification"] = np.array(list(map(lambda x: val_class[x], fg_index_df["value_classification"])))

    return fg_index_df[["value_classification"]].rename(columns={"value_classification":"gf-index"})

