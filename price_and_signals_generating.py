import os
import numpy as np
import pandas as pd
np.random.seed(0)

root_path=os.pahth.join("/Users","chenziheng","Desktop","AlphaTrader")
folder_path="Data"

if not os.path.exists(os.path.join(root_path, folder_path)):
    os.makedirs(os.path.join(root_path, folder_path))

# Generate Simulated Price Series Using Geometric Brownian Motion
# mu is set to zero, the only way to make profit is out of volatility
# daily_volatility is 3%, 390 minutes in a trading day, from 9:30 to 16:00
# minute volatility is 3%/sqrt(390)= 0.0015
S_0=100
T=390
mu=0
sigma=0.0015
sample_number=10000
# Generate Simulated Alpha Signals Having Setted Information Coefficient(IC)
# IC: Correlation Coefficient with Future Return
# I use r_1+r_2+...+r_n to approximate (1+r_1)(1+r_2)...(1+r_n)-1 since r_1, r_2...r_n are small
# r_1 = sigma * N_1(0,1), r_2 = sigma * N_2(0,1)...r_n = sigma * N_n(0,1) 
# N_1...N_n are independent standard normal variables 
# r_1+...r_n= sigma * (N_1(0,1)+...+N_n(0,1)) = sigma * N(0,n) = sigma * sqrt(n) * N(0,1)
# n_minutes_signals = sigma * sqrt(n) * (IC * N(0,1) + sqrt(1-IC^2) * N_noise(0,1))= IC * (r_1+...+r_n)+sigma * sqrt((1-IC^2)*n) * N_noise(0,1)
# N_noise are different for every signal
# n_minute siganls has no signal since last n minutes, contains noise only



prices=[]
returns=[]
signals={1:[],5:[],10:[],30:[]}
signals_IC={1:0.5,5:0.5,10:0.5,30:0.5}

for i in range(sample_number):
    if i%1000==0:
        print("Current Process:", i)
    standard_normal_series = np.random.normal(loc=0, scale=1, size=T)  
    ret=sigma*(standard_normal_series)
    price = np.concatenate(([S_0],S_0*(1+ret).cumprod()))
    prices.append(price)
    returns.append(ret)
    
    for signal_length in [1,5,10,30]:
        return_series=np.array(pd.Series(price).pct_change(signal_length).shift(-signal_length).fillna(0))
        noise_normal_series= np.random.normal(loc=0, scale=1, size=T+1) 
        signals_series=signals_IC[signal_length]*return_series+sigma*(signal_length*(1-signals_IC[signal_length]**2))**(1/2)*noise_normal_series
        signals[signal_length].append(signals_series)
        
prices=pd.DataFrame(prices)
for key in signals:
    signals[key]=pd.DataFrame(signals[key])
    signals[key].to_csv(os.path.join(root_path, folder_path ,str(key)+"_minutes_signals.csv"))


prices.to_csv(os.path.join(root_path, folder_path, "prices.csv"))



