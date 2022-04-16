import os
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt

from environment import Tzero_Environment
from agent_ppo import Proximal_Policy_Optimization


folder_path="Data"
prices=pd.read_csv(os.path.join(folder_path,"prices.csv"),index_col=0)
signals={1:[],5:[],10:[],30:[]}
for key in signals:
    signals[key]=pd.read_csv(os.path.join(folder_path,str(key)+"_minutes_signals.csv"), index_col=0)

print("Data Loaded Suceesfully")

TOTAL_EPISODE_NUMBER=200000
SAMPLE_LENGTH=391
# How many states are there in one sample
DELAY=0
# Delay in minutes, if cannot be executed immediately

trial_num=0
Output_folder="AlphaTrader_"+str(trial_num)

if not os.path.exists(Output_folder):
    os.makedirs(Output_folder)

PnL_fig="PnL_fig"
Entropy_fig="Entropy_fig"
Critic_loss_fig="Critic_loss_fig"

parameter_dic={

"Batch_size":2048, 

"Mini_learn_epoch":10,

"Discount_rate":0.8,

"Whether_use_critic":False,

"Whether_use_rolling_mean_as_critic":False,

"Critic_train_batch_size":10000,

"Critic_test_batch_size":3000,

"Feature_number":47,
# 46 signals and 1 position

"Action_number":41,

"Actor_entropy_multiplier":1,

"Actor_entropy_multiplier_decay":0.9,

"Actor_minimum_entropy_multiplier":0.005,

"Clipping_up_ratio":0.2,

"Clipping_down_ratio":0.2,

"Actor_learning_rate":0.01,

"Critic_learning_rate":0.01,

"Critic_training_epoch":30,

"Save_model":True,

"Save_model_path":Output_folder,

"Load_model":False,

"Load_model_path":None
}

print("Total_episode_number: ", TOTAL_EPISODE_NUMBER)
print("Sample_length: ", SAMPLE_LENGTH)
print("Trading_delay_minutes: ", DELAY)
print(parameter_dic)


# Hyperparameters Setting
####################################################################################

rl_environment= Tzero_Environment(signals, prices, DELAY, SAMPLE_LENGTH)
rl_agent=Proximal_Policy_Optimization(parameter_dic)

current_episode_number=0

PnL_list=[]
entropy_list=[]
critic_loss_list=[]
current_episode_number=0

while current_episode_number<TOTAL_EPISODE_NUMBER:
    
    current_episode_number+=1    
    whether_terminal=False
    rl_environment.reset_environment()
    entropy_record=[]

    if current_episode_number>=20000 and current_episode_number%5000==0:
        rl_agent.rolling_mean_as_critic=pd.Series(PnL_list[-15000:]).mean()
        
    
    if current_episode_number%1000==0:
        rl_agent.update_entropy_mutiplier()
        print("current episode:",current_episode_number,"entropy_multiplier: ", rl_agent.entropy_multiplier)

    if current_episode_number>=5000 and current_episode_number%2000==0:
        plt.plot(pd.Series(PnL_list).rolling(4000).mean(), color='black', lw=0.2)     
        plt.savefig(os.path.join(Output_folder,PnL_fig))
        plt.clf()


        plt.plot(pd.Series(entropy_list).rolling(4000).mean(), color='black', lw=0.2)     
        plt.savefig(os.path.join(Output_folder,Entropy_fig))
        plt.clf()

        
        if rl_agent.whether_use_critic==True:

            plt.plot(pd.Series(rl_agent.critic_train_loss_list[20:]), color='black', lw=0.2)   
            plt.plot(pd.Series(rl_agent.critic_test_loss_list[20:]), color='red', lw=0.2)  
            plt.savefig(os.path.join(Output_folder,Critic_loss_fig))
            plt.clf()

    if current_episode_number%1000==0 and rl_agent.whether_save_model==True:
        rl_agent.save_model()
        print("episode:", current_episode_number, "Model saved successfully")

    while True:
        current_state=rl_environment.get_now_state()
        if rl_environment.now_state_num==rl_environment.sample_length-2:
            whether_terminal=True
        # when we get to the last second state our episode ends

        current_action, entropy=rl_agent.choose_action(current_state, whether_terminal)   
                
        reward, next_state=rl_environment.accept_action_and_give_reward_and_next_state(current_action, whether_terminal)
        # print("current state:",current_state)
        # print("current action", current_action) 
        # print("reward:",reward, "next state", next_state)
        if entropy!=None:
            rl_agent.reward_history.append(reward)
            entropy_record.append(entropy)

        if whether_terminal==True:
            terminal_reward=reward
            rl_agent.terminal_reward.append(terminal_reward)
            PnL_list.append(rl_environment.realized_PnL)
            entropy_list.append(np.array(entropy_record).mean())
            if current_episode_number%100==0:
                print("current episode number: ", current_episode_number, "PnL: ", rl_environment.realized_PnL)
            break
    if current_episode_number>0 and current_episode_number%rl_agent.batch_size==0:
        print("current episode: ", current_episode_number, "last batch pnl: ", np.array(PnL_list[-rl_agent.batch_size:]).mean())
        rl_agent.learn()

if rl_agent.whether_save_model==True:
    rl_agent.save_model()
    print("episode:", current_episode_number, "Model saved successfully")