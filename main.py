from Signal_gen import signal_gen
from execution import execution

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    data_path = "Data/"
    signal_path = "Signal/"
    tau = 5
    M = 3
    N = 3
    K = 2
    risk_percentage = 0.6
    symbol_dict = {
        "VG" : "EuroStoxx",
        "BP" : "GBP - British Pound",
        "RX" : "German Bunds - German Government Bonds",
        "GC" : "Gold",
        "HO" : "HeatingOil",
        "JY" : "JPY - Japanese Yen",
        "NQ" : "Nasdaq"
    }
    #signal_gen(data_path, signal_path)
    execution(signal_path,data_path,symbol_dict,tau, M, N, K,risk_percentage)