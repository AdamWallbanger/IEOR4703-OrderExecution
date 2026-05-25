import pandas as pd
from tqdm import tqdm
import numpy as np

def backtest(result_df,data_path,symbol_dict,tau,result_address):
    filled_price = []
    for idx, row in tqdm(result_df.iterrows(), total=len(result_df), desc="Order by Order"):
        data = row["main_contract_clean"]
        instrument = str(data)[0:2]
        time = pd.Timestamp(row["timestamp"])
        path = data_path + symbol_dict[instrument] + "/" + data
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"])
        price = row["opti_price"]
        placement_end = time + pd.Timedelta(str(tau) + "min")
        placement_df = df[(df["time"] >= time) & (df["time"] <= placement_end)]
        next_row = df[df["time"] > placement_end].head(1)
        if len(placement_df) == 0:
            if len(next_row) == 0:
                filled_price.append(np.nan)
                continue
            price = next_row["open"].iloc[0]
            filled_price.append(price)
            continue
        mask = (placement_df["low"] <= price) & (placement_df["high"] >= price)
        filled = mask.any()
        if filled:
            filled_price.append(price)
        else:
            filled_price.append(list(placement_df["close"])[-1])
    result_df["filled_price"] = filled_price
    result_df = result_df.dropna(subset=["filled_price"]).copy()
    result_df["benchmark_amount"] = result_df["price"] * result_df["signal"].abs()
    result_df["amount"] = result_df["filled_price"] * result_df["signal"].abs()
    result_df["slippage"] = (result_df["benchmark_amount"] - result_df["amount"]) * np.sign(result_df["signal"])
    result_df.to_csv(result_address,index=False)