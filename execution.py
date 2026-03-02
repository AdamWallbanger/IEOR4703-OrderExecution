import pandas as pd
import os
from epdf import ePDFCalculator

def execution(signal_path,data_path,tau,M,N,K,risk_percentage,ewma_halflife=10,estimation_method='smoothed',smoothing_alpha=0.5):
    for filename in os.listdir(signal_path):
        signal = pd.read_csv(signal_path+filename)
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
