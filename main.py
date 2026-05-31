from Signal_gen import signal_gen
from execution import execution
from multiprocessing import Pool
import multiprocessing as mp
from backtest import backtest
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

def backtest_worker(filename,result_path,data_path,symbol_dict,tau):
    result_df = pd.read_csv(result_path + filename)
    result_address = result_path + filename
    backtest(result_df,data_path,symbol_dict,tau,result_address)

if __name__ == '__main__':

    t = [5,10,15,30,60]
    for i in range(len(t)):
        data_path = "Data/"
        signal_path = "Signal/"
        tau = t[i]
        M = 3
        N = 3
        K = 2
        risk_percentage = 0.9
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

        result_path = "Result_" + str(tau) + "min/"
        func = partial(
            backtest_worker,
            result_path=result_path,
            data_path = data_path,
            symbol_dict = symbol_dict,
            tau = tau
        )

        filenames = os.listdir(result_path)
        with Pool(processes=n_workers) as pool:
            for _ in tqdm(pool.imap_unordered(func, filenames), total=len(filenames), desc="Backtest"):
                pass

