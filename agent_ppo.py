import os
import copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class Actor_NN(nn.Module):
    def __init__(self, feature_number, action_number):
        super(Actor_NN, self).__init__()

        self.linear1 = nn.Linear(feature_number, 40)
        torch.nn.init.orthogonal_(self.linear1.weight)
        self.linear2 = nn.Linear(40, 40)
        torch.nn.init.orthogonal_(self.linear2.weight)
        self.linear3 = nn.Linear(40, 20)
        torch.nn.init.orthogonal_(self.linear3.weight)
        self.linear4 = nn.Linear(20, 10)
        torch.nn.init.orthogonal_(self.linear4.weight)
        self.linear5 = nn.Linear(10, action_number)
        torch.nn.init.orthogonal_(self.linear5.weight)
     

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        action_scores = self.linear5(x)
        action_probs = F.softmax(action_scores,dim=0)
        return action_probs

class Critic_NN(nn.Module):
    def __init__(self, feature_number):
        super(Critic_NN, self).__init__()

        self.linear1 = nn.Linear(feature_number, 20)
        torch.nn.init.kaiming_normal_(self.linear1.weight)
        self.linear2 = nn.Linear(20, 10)
        torch.nn.init.kaiming_normal_(self.linear2.weight)
        self.linear3 = nn.Linear(10, 10)
        torch.nn.init.kaiming_normal_(self.linear3.weight)
        self.linear4 = nn.Linear(10, 1)
        torch.nn.init.kaiming_normal_(self.linear4.weight)

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        state_values = self.linear4(x)
        return state_values

def get_entropy(probs):
    return -torch.sum(torch.mul(torch.log(probs+1e-6),probs))


class Proximal_Policy_Optimization:

    def __init__(self,para_dic):
        
        # action 0,1...19,20,21...39,40 correspons to trade/change to position [-100,-95...-,5,0,5,...95,100]
        
        self.action_number=para_dic["Action_number"]
        self.feature_number=para_dic["Feature_number"]
        self.discount_rate=para_dic["Discount_rate"]
        self.entropy_multiplier=para_dic["Actor_entropy_multiplier"]
        self.entropy_decay=para_dic["Actor_entropy_multiplier_decay"]
        self.minimum_entropy_multiplier=para_dic["Actor_minimum_entropy_multiplier"]
        self.actor_learning_rate=para_dic["Actor_learning_rate"] 
        self.rolling_mean_as_critic=0
        self.whether_use_critic=para_dic["Whether_use_critic"]
        self.whether_rolling_mean_as_critic=para_dic["Whether_use_rolling_mean_as_critic"]
        self.critic_learning_rate=para_dic["Critic_learning_rate"] 
        self.up_clipping_ratio = para_dic["Clipping_up_ratio"]
        self.down_clipping_ratio=para_dic["Clipping_down_ratio"]
        self.whether_save_model=para_dic["Save_model"]
        self.whether_load_model=para_dic["Load_model"]
        self.save_model_path=para_dic["Save_model_path"]
        self.load_model_path=para_dic["Load_model_path"]
        self.batch_size=para_dic["Batch_size"]
        self.mini_learn_epoch=para_dic["Mini_learn_epoch"]
        self.critic_train_batch_size=para_dic["Critic_train_batch_size"]
        self.critic_test_batch_size=para_dic["Critic_test_batch_size"]
        self.critic_training_epoch=para_dic["Critic_training_epoch"]
        self.action_space=np.array(range(self.action_number))

        if self.whether_load_model==True:
            self.load_model()
            print("Model loaded successfully")
        else:
            self.actor_net=Actor_NN(self.feature_number, self.action_number)
            self.critic_net= Critic_NN(self.feature_number)


        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.critic_learning_rate)

        self.critic_loss_func=nn.MSELoss()
        
        self.reward_history = []
        self.state_history=[]
        self.action_history=[]
        self.critic_train_loss_list=[]
        self.critic_test_loss_list=[]

        self.first_episode=True
        self.epoch_length_counter=0

        self.epoch_length_history=[]
        self.terminal_reward=[]

    def save_model(self):
        torch.save(self.actor_net,os.path.join(self.save_model_path,"Actor_Net"))
        torch.save(self.critic_net,os.path.join(self.save_model_path,"Critic_Net"))

    def load_model(self):
        self.actor_net=torch.load(os.path.join(self.load_model_path,"Actor_Net"))
        self.critic_net=torch.load(os.path.join(self.load_model_path,"Critic_Net"))

    def get_action_space(self, state):
        
        current_position = state[-1]
        
        if current_position >= 0:
            buy_action_space = list(range(21,21+int((100-current_position)/5)))
            hold_action_space=[20]
            sell_action_space= list(range(0,19))

        else:
            buy_action_space= list(range(21, 40))
            hold_action_space=[20]
            sell_action_space= list(range(19,19+int((-100-current_position)/5),-1))

        action_space=buy_action_space+hold_action_space+sell_action_space
        action_space=list(set(action_space))
        action_space.sort()
        
        return action_space

    def update_entropy_mutiplier(self):
        if self.entropy_multiplier>self.minimum_entropy_multiplier:
            self.entropy_multiplier*=self.entropy_decay

    def choose_action(self, state, whether_terminal):

        original_state=copy.deepcopy(state)

        if whether_terminal:
            action_position = -state[-1]
            if action_position==0:
                action=20
            elif action_position>0:
                action=int(action_position/5+20)
            else:
                action=int(action_position/5+20) 
            entropy=None

            self.epoch_length_history.append(self.epoch_length_counter)
            self.epoch_length_counter=0
            
        else:
            self.epoch_length_counter+=1

            action_space=self.get_action_space(state)

            # print("action_space:", action_space)
    
            state = torch.from_numpy(state).float()
            probs = self.actor_net(state)
                     
            m = Categorical(probs)
            action = m.sample()  
            action=action.item()

            self.state_history.append(original_state)
            self.action_history.append(action)

            # I need to map illegal action to legal action space
            if action in action_space: pass
            elif action<action_space[0]:
                while action not in action_space:
                    action+=1
            elif action>action_space[-1]:
                while action not in action_space:
                    action-=1
            entropy=get_entropy(probs).item()

        return action, entropy

    def learn(self):
        
        rewards = []
        batch_size=self.batch_size
        assert len(self.terminal_reward)==self.batch_size

        for i in range(batch_size):
            R = self.terminal_reward[i]
            local_saver=[]
            if i==0:
                for r in self.reward_history[self.epoch_length_history[i]-1::-1]:   
                    R = r + R * self.discount_rate
                    local_saver.insert(0,R)
            else:    
                for r in self.reward_history[sum(self.epoch_length_history[0:i+1])-1:sum(self.epoch_length_history[0:i])-1:-1]:
                    R = r + R * self.discount_rate
                    local_saver.insert(0,R)
            rewards=rewards+local_saver

        assert len(rewards)==len(self.state_history)==len(self.action_history)
        print("total sample_number:", len(self.state_history))

        self.state_history=np.array(self.state_history).astype(np.float32)

        if self.whether_use_critic==True:

            if self.first_episode==True: 
                advantage_values=torch.tensor(rewards,requires_grad=False)
            else: 
                X_estimate=np.array(self.state_history).astype(np.float32)
                critic_estimated_state_value=torch.tensor([self.critic_net(torch.tensor(i)) for i in X_estimate]).reshape(-1,1)
                advantage_values=torch.tensor(rewards).reshape(-1,1)-critic_estimated_state_value           

            X_critic_train=self.state_history[:int(len(self.state_history)*0.7)]
            Y_critic_train=np.array(rewards).astype(np.float32)[:int(len(self.state_history)*0.7)]

            X_critic_test=self.state_history[int(len(self.state_history)*0.7):]
            Y_critic_test=np.array(rewards).astype(np.float32)[int(len(self.state_history)*0.7):]


            for i in range(self.critic_training_epoch): 
                if i%10==0: print("critic_training_epoch", i)

                train_batch_indice = torch.randint(len(X_critic_train), size=(min(self.critic_train_batch_size,len(X_critic_train)),))
                X_critic_train_batch=X_critic_train[train_batch_indice]
                Y_critic_train_batch=Y_critic_train[train_batch_indice]

                test_batch_indice = torch.randint(len(X_critic_test), size=(min(self.critic_test_batch_size,len(X_critic_test)),))
                X_critic_test_batch=X_critic_test[test_batch_indice]
                Y_critic_test_batch=Y_critic_test[test_batch_indice]

                critic_estimating_results=[self.critic_net(torch.tensor(item)) for item in X_critic_train_batch]
                critic_loss_list=[(item-Y_critic_train_batch[num])**2 for num,item  in enumerate(critic_estimating_results)]
                critic_loss=sum(critic_loss_list)/len(critic_loss_list)
                self.critic_optimizer.zero_grad()
                critic_external_grad = torch.tensor([1.])
                critic_loss.backward(critic_external_grad,retain_graph=True)
                self.critic_optimizer.step()

                with torch.no_grad():
                    critic_estimating_results=[self.critic_net(torch.tensor(item)) for item in X_critic_train_batch]
                    critic_loss_list=[(item-Y_critic_train_batch[num])**2 for num,item  in enumerate(critic_estimating_results)]
                    critic_loss=sum(critic_loss_list)/len(critic_loss_list)
                    self.critic_train_loss_list.append(critic_loss)
                    critic_test_estimating_results=[self.critic_net(torch.tensor(item)) for item in X_critic_test_batch]
                    critic_test_loss_list=[(item-Y_critic_test_batch[num])**2 for num,item  in enumerate(critic_test_estimating_results)]
                    critic_test_loss=sum(critic_test_loss_list)/len(critic_test_loss_list)
                    self.critic_test_loss_list.append(critic_test_loss)
            print("critic_train_loss:", critic_loss, "critic_test_loss:",critic_test_loss)
        
        elif self.whether_rolling_mean_as_critic==True:
            advantage_values=torch.tensor(rewards).reshape(-1,1)
            advantage_values-=self.rolling_mean_as_critic

        else:
            advantage_values=torch.tensor(rewards).reshape(-1,1)

        if len(self.state_history)>=30000:
            self.state_history=self.state_history[:30000]
            self.action_history=self.action_history[:30000]
        print("actor_learn_sample_number:", len(self.state_history))


        # generate old policy probabilities
        with torch.no_grad():
            old_policy_probs=[self.actor_net(torch.tensor(i)) for i in self.state_history]

        for mini_learn_epoch in range(self.mini_learn_epoch):
            print("mini_actor_learn_epoch", mini_learn_epoch)
        
            new_policy_probs=[self.actor_net(torch.tensor(state)) for state in self.state_history]
            entropy_list=[get_entropy(i) for i in new_policy_probs]
            entropy=sum(entropy_list)/len(entropy_list)

            surrogate_list=[]
            probs_ratio_list=[]
    
            for num, action in enumerate(self.action_history):
                probs_ratio=torch.div(new_policy_probs[num][action], old_policy_probs[num][action])
                probs_ratio_list.append(probs_ratio)
                surrogate1 = 1000*advantage_values[num]* probs_ratio
                surrogate2 = 1000*advantage_values[num]* probs_ratio.clamp(1 - self.down_clipping_ratio, 1 + self.up_clipping_ratio)
                surrogate_list.append(-torch.min(surrogate1, surrogate2))
 
            # print("probs_ratio_list", probs_ratio_list)
            actor_loss = (sum(surrogate_list)/len(surrogate_list)-self.entropy_multiplier * entropy).reshape(1,)

            print("expected_advantage_value_part", -sum(surrogate_list)/len(surrogate_list))
            print("entropy_part", self.entropy_multiplier * entropy)

            self.actor_optimizer.zero_grad()
            actor_external_grad = torch.tensor([1.])
            actor_loss.backward(actor_external_grad, retain_graph=True)
            self.actor_optimizer.step()

        # print("actor_loss",torch.mean(obj_surrogate).reshape(1,))

        
        self.action_history=[]
        self.reward_history=[]
        self.state_history=[]
        self.epoch_length_history=[]
        self.terminal_reward=[]

        self.first_episode=False

        
            
            
           