import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from epdf import ePDFCalculator

def execution(signal,filename,data_path,symbol_dict,tau,M,N,K,risk_percentage,tick_dict,ewma_halflife=10,estimation_method='smoothed',smoothing_alpha=0.5):
    result_path = "Result_" + str(tau) + "min/"
    if not os.path.isdir(result_path):
        os.mkdir(result_path)
    signal["timestamp"] = pd.to_datetime(signal["timestamp"])
    p = []
    for idx, row in tqdm(signal.iterrows(), total=len(signal), desc="Order by Order"):
        data = row["main_contract_clean"]
        instrument = str(data)[0:2]
        calc = ePDFCalculator(
            instrument=instrument,
            tau=tau,
            M=M,
            N=N,
            K=K,
            ewma_halflife=ewma_halflife,
            estimation_method=estimation_method,
            smoothing_alpha=smoothing_alpha
        )
        time = row["timestamp"]
        path = data_path + symbol_dict[instrument] + "/" + data
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)

        calc.fit(path, train_end_date=time)
        df["volume_ewma"] = (
            df["volume"]
            .shift(1)
            .ewm(halflife=ewma_halflife, adjust=False)
            .mean()
        )
        tick = tick_dict[instrument]

        df["range_R"] = (df["high"] - df["low"]) / tick

        df["volatility_ewma"] = (
            df["range_R"]
            .shift(1)
            .ewm(halflife=ewma_halflife, adjust=False)
            .mean()
        )
        df["open_delta"] = df["open"].diff()

        df["ewma_delta_x"] = (
            df["open_delta"]
            .shift(1)
            .ewm(halflife=ewma_halflife, adjust=False)
            .mean()
        )
        hist = df[df["time"] < row["timestamp"]]

        if hist.empty:
            continue
        pre_row = hist.iloc[-1]
        if (
                pd.isna(pre_row["volume_ewma"])
                or pd.isna(pre_row["volatility_ewma"])
                or pd.isna(pre_row["ewma_delta_x"])
        ):
            continue
        state = calc.get_current_state(
            pre_row["volume_ewma"],
            pre_row["volatility_ewma"],
            pre_row["ewma_delta_x"]
        )
        if row["signal"] > 0:
            direction = 'range_dn'
        else:
            direction = 'range_up'
        placement = (0,0)
        for l in range (10):
            cdf = calc.query_cdf(l, direction, state)
            if cdf >= risk_percentage:
                placement = (l,cdf)
            else:
                break
        if direction == "range_dn":
            price = pre_row["close"] - (l * tick_dict[instrument])
        else:
            price = pre_row["close"] + (l * tick_dict[instrument])
        p.append(price)
    signal["opti_price"] = p
    signal.to_csv(result_path + filename,index=False)