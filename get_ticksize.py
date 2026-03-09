import os
import pandas as pd

TICK_SIZES = {}

data_path = "Data/"
for folders in os.listdir(data_path):
    folder_path = data_path + folders
    tick_list = []
    symbol = ""
    for filename in os.listdir(folder_path):
        if (".csv" in filename) and ("AI" not in filename):
            symbol = filename[0:2]
            df = pd.read_csv(folder_path+"/"+filename)
            tick_list.append(
                df["open"]
                .sort_values()
                .diff()
                .loc[lambda x: x > 0]
                .min()
            )
            tick_list.append(
                df["high"]
                .sort_values()
                .diff()
                .loc[lambda x: x > 0]
                .min()
            )
            tick_list.append(
                df["low"]
                .sort_values()
                .diff()
                .loc[lambda x: x > 0]
                .min()
            )
            tick_list.append(
                df["close"]
                .sort_values()
                .diff()
                .loc[lambda x: x > 0]
                .min()
            )
    if len(tick_list) > 0:
        TICK_SIZES[symbol] = min(tick_list)
print(TICK_SIZES)