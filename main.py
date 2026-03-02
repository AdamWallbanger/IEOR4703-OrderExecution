from Signal_gen import signal_gen
from execution import execution

if __name__ == '__main__':
    data_path = "Data/"
    signal_path = "Signal/"
    tau = 5
    M = 3
    N = 3
    K = 2
    risk_percentage = 0.7
    #signal_gen(data_path, signal_path)
    execution(signal_path,data_path,tau, M, N, K)