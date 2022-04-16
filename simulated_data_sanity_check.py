import pandas as pd
import numpy as np
import os


root_path=os.pahth.join("Users","chenziheng","Desktop","AlphaTrader")
folder_path="Data"

prices=pd.read_csv(os.path.join(folder_path,"prices.csv"),index_col=0)
signals={1:[],5:[],10:[],30:[]}
for key in signals:
    signals[key]=pd.read_csv(os.path.join(folder_path,str(key)+"_minutes_signals.csv"), index_col=0)
# print(prices)
# print(signals)

for i in range(1,10000,100):
    price=prices.iloc[i,:]

    for signal_length in [1,5,10,30]:
            return_series=np.array(price.pct_change(signal_length).shift(-signal_length).fillna(0))
            signal_series=np.array(signals[signal_length].iloc[i,:])  
            IC = np.corrcoef(return_series, signal_series)
            print("sample number: ", str(i), "signal_length: ", signal_length, "IC:", IC[0,1])


