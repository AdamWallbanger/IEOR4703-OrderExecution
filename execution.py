import pandas as pd
import numpy as np
import os

from epdf import ePDFCalculator

def execution(signal_path,data_path,symbol_dict,tau,M,N,K,risk_percentage,ewma_halflife=10,estimation_method='smoothed',smoothing_alpha=0.5):
    for filename in os.listdir(signal_path):
        signal = pd.read_csv(signal_path+filename)
        signal["timestamp"] = pd.to_datetime(signal["timestamp"])
        for idx, row in signal.iterrows():
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
            date = str(row["timestamp"].date())
            path = data_path + symbol_dict[instrument] + "/" + data
            df = pd.read_csv(path)
            df["time"] = pd.to_datetime(df["time"])
            calc.fit(path, train_end_date=date)
            df["volume_ewma"] = df["volume"].ewm(halflife=ewma_halflife, adjust=False).mean()
            df["ret"] = df["close"].pct_change()
            df["volatility_ewma"] = np.sqrt(
                df["ret"].pow(2).ewm(halflife=ewma_halflife,  adjust=False).mean()
            )
            df["delta_x"] = df["close"].diff()
            df["delta_x"] = df["delta_x"].dropna()
            df["ewma_delta_x"] = df["delta_x"].ewm(span=20, adjust=False).mean()
            pre_row = df[df["time"] < row["timestamp"]].iloc[-1]
            state = calc.get_current_state(pre_row["volume_ewma"], pre_row["volatility_ewma"], pre_row["ewma_delta_x"])
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
            price
