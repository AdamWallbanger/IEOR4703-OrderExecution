from Signal_gen import signal_gen
from execution import execution
from multiprocessing import Pool
import multiprocessing as mp
from functools import partial
import pandas as pd
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import builtins

builtins.print = lambda *args, **kwargs: None

def execution_worker(filename,signal_path,data_path,symbol_dict,tau,M,N,K,risk_percentage,tick_dict,ewma_halflife,estimation_method,smoothing_alpha):
    signal = pd.read_csv(signal_path+filename)
    execution(signal,filename,data_path, symbol_dict, tau, M, N, K, risk_percentage,tick_dict, ewma_halflife, estimation_method, smoothing_alpha)

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
    tick_dict = {
        'NQ': 0.25,
        'HO': 0.01,
        'GC': 0.10,
        'BP': 0.01,
        'JY': 0.005,
        'RX': 0.01,
        'VG': 0.50
    }
    #signal_gen(data_path, signal_path)
    func = partial(
        execution_worker,
        signal_path = signal_path,
        data_path=data_path,
        symbol_dict = symbol_dict,
        tau = tau,
        M = M,
        N = N,
        K = K,
        risk_percentage = risk_percentage,
        tick_dict = tick_dict,
        ewma_halflife = 10,
        estimation_method = 'smoothed',
        smoothing_alpha = 0.5
    )

    n_workers = os.cpu_count() - 2
    filenames = os.listdir(signal_path)
    with Pool(processes=n_workers) as pool:
        for _ in tqdm(pool.imap_unordered(func, filenames), total=len(filenames), desc="Placement"):
            pass