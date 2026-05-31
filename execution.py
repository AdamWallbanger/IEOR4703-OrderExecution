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
        tick = tick_dict[instrument]

        try:
            calc.fit(path, train_end_date=time)

            df_proc = calc.data_processor.process_pipeline(
                filepath=path,
                tick_size=tick,
                tau=tau,
                min_completeness=0.9,
                train_end_date=None
            )

            df_proc = calc.state_classifier.compute_all_ewma_features(
                df_proc,
                ewma_halflife
            )

            if not isinstance(df_proc.index, pd.DatetimeIndex):
                if "time" in df_proc.columns:
                    df_proc["time"] = pd.to_datetime(df_proc["time"])
                    df_proc = df_proc.set_index("time")
                elif "timestamp" in df_proc.columns:
                    df_proc["timestamp"] = pd.to_datetime(df_proc["timestamp"])
                    df_proc = df_proc.set_index("timestamp")
                else:
                    p.append(np.nan)
                    continue

            df_proc = df_proc.sort_index()

            hist = df_proc[df_proc.index < time]

            if hist.empty:
                p.append(np.nan)
                continue

            pre_row = hist.iloc[-1]

            state_cols = ["v_ewma", "sigma_ewma", "delta_x_ewma"]

            if not all(col in df_proc.columns for col in state_cols):
                p.append(np.nan)
                continue

            if pre_row[state_cols].isna().any():
                p.append(np.nan)
                continue

            state = calc.get_current_state(
                pre_row["v_ewma"],
                pre_row["sigma_ewma"],
                pre_row["delta_x_ewma"]
            )

            if row["signal"] > 0:
                direction = "range_dn"
            else:
                direction = "range_up"

            placement = (0, np.nan)

            for level in range(10):
                cdf = calc.query_cdf(level, direction, state)

                if cdf >= risk_percentage:
                    placement = (level, cdf)
                else:
                    break

            placement_level = placement[0]

            if direction == "range_dn":
                price = pre_row["close"] - placement_level * tick
            else:
                price = pre_row["close"] + placement_level * tick

            p.append(price)

        except Exception as e:
            print(f"Failed on idx={idx}, data={data}, time={time}, error={e}")
            p.append(np.nan)

    signal["opti_price"] = p
    signal.to_csv(result_path + filename,index=False)