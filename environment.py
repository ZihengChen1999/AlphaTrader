import numpy as np

class Tzero_Environment:
    def __init__(self, signals, prices, delay, sample_length, variance_penalty):

        self.signals=signals
        self.prices=prices
        self.total_sample_number=prices.shape[0]    
        self.sample_number=0
        self.delay=delay
        self.sample_length=sample_length
        self.variance_penalty=variance_penalty

        print("Total sample number: ", self.total_sample_number)

        
    def reset_environment(self,random=True):

        if random==True: self.sample_number=np.random.randint(0, self.total_sample_number-1)
        if self.sample_number>=self.total_sample_number: self.sample_number=0

        self.sample_signals=[self.signals[1].iloc[self.sample_number,:], self.signals[5].iloc[self.sample_number,:], self.signals[10].iloc[self.sample_number,:], self.signals[30].iloc[self.sample_number,:]]
        self.sample_prices=self.prices.iloc[self.sample_number,:]
        self.now_state_num=self.delay
        self.position=0
        self.cum_buy_position=0
        self.cum_sell_position=0
        self.realized_PnL=0
        # PnL are represented in percentage
        self.buy_price=0
        self.sell_price=0
        self.hold_cost=0
        self.sample_number+=1

    def get_now_state(self):

        signals=[]
        for index, signal_length in enumerate([1,5,10,30]):
            if self.now_state_num<signal_length-1:
                signal=self.sample_signals[index].values[:self.now_state_num+1]
                signal=np.concatenate((np.zeros(signal_length-1-self.now_state_num), signal))

            else:
                signal=self.sample_signals[index].values[self.now_state_num+1-signal_length:self.now_state_num+1]

            signals.append(signal)

        now_state=np.concatenate((np.concatenate(tuple(signals)),np.array(self.position).reshape(1,)))

        return now_state
   
    def accept_action_and_give_reward_and_next_state(self, action, whether_terminal):
        # action 0,1...19,20,21...39,40 correspons to trade/change to position [-100,-95...-,5,0,5,...95,100]
       
        if action<=19:
            action_position=-5*(20-action)
        elif action==20:
            action_position=0
        else:
            action_position=5*(action-20)

        self.position+=action_position

        if action_position>0:
            self.buy_price  = (self.buy_price * self.cum_buy_position + action_position * self.sample_prices.values[self.now_state_num+1])/(self.cum_buy_position+action_position)
        elif action_position<0:
            self.sell_price = (self.sell_price * self.cum_sell_position - action_position * self.sample_prices.values[self.now_state_num+1])/(self.cum_sell_position-action_position)

        if action_position>=0:
            self.cum_buy_position+=action_position
        else:
            self.cum_sell_position+=abs(action_position)      

        if whether_terminal==True:

            assert self.position==0
            assert self.cum_buy_position==self.cum_sell_position
            # I need to check cum_buy==cum_sell 
            # I need to check whether self.cum_buy_position>0

            if self.cum_buy_position>0:
                self.realized_PnL = self.cum_buy_position * ((self.sell_price-self.buy_price)/self.buy_price)
            
        last_period_return = self.sample_prices.values[self.now_state_num+1]/self.sample_prices.values[self.now_state_num]-1
        
        # if last_period_return> 0.02:last_period_return= 0.02
        # if last_period_return<-0.02:last_period_return=-0.02
        # To eliminate the outliers
    
        reward = last_period_return * self.position
        reward = reward - self.variance_penalty*reward**2

        self.now_state_num+=1
        next_state=self.get_now_state()

        return reward, next_state
    