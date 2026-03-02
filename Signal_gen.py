import os
import pandas as pd

def signal_gen(data_path,signal_path):
    for folder_name in sorted(os.listdir(data_path)):
        folder_path = os.path.join(data_path, folder_name)
        for filename in os.listdir(folder_path):
            if "AI" in filename:
                file_path = os.path.join(folder_path,filename)
                inventory = pd.read_csv(file_path)
                inventory["signal"] = inventory["inventory"].diff()
                inventory = inventory[inventory["signal"] != 0]
                inventory = inventory.dropna()
                signal_name = filename.replace(".csv","")
                inventory.to_csv(signal_path+signal_name+".csv",index=False)
