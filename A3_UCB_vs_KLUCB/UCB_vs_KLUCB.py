#Multi-armed bandits
#The aim is to compare UCB to KL-UCB quantitatively

#Importing necessary modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kl_div
from scipy.optimize import bisect
from scipy.interpolate import make_interp_spline
from scipy.stats import ttest_ind
import pandas as pd

#Creating a Bandit class
class Bernoulli_bandit():
    def __init__(self, prob):
        self.prob=prob

    #Pulling the arm of the bandit will generate a stochastic reward
    def pull(self):
        if np.random.random()<self.prob:
            return 1
        else:
            return 0

#UCB algorithm
class UCB():
    def __init__(self):
        self.exist=0

    #Re-initialising the algorithm between runs
    def reset(self, n_arms):
        #Number of pulls per arm
        self.pulls=np.zeros(n_arms)
        #Rewards obtained per arm
        self.rewards=np.zeros(n_arms)

    #Choosing which arm to pull based on UCB decision rule
    def choose_arm(self):
        n_arms=len(self.pulls)

        #For each arm, if an arm has never been pulled, pull it
        for n in range(n_arms):
            if self.pulls[n]==0:
                #Pull it
                return n

        #Compute the upper bound as per UCB formula
        t=np.sum(self.pulls)
        upper_bound=np.sqrt((1/self.pulls)*np.log(1+t*(np.log(t)**2)))
        #Add the upper bound to the previous rewards
        values=upper_bound+self.rewards

        #Select highest value (optimistic strategy)
        return np.argmax(values)

    #Once the selected arm has been pulled, the stats of the model are updated
    def update_stats(self, chosen_arm, reward):
        #Updating number of pulls
        self.pulls[chosen_arm]+=1
        n_pulls=self.pulls[chosen_arm]
        current_mean=self.rewards[chosen_arm]
        #Updating mean reward for that ar
        self.rewards[chosen_arm]=current_mean+(reward-current_mean)/n_pulls

class KL_UCB():
    #Initialising the algorithm which has three characteristics: rewards per arm, pulls per arm, number of arms
    def __init__(self):
        self.exist=0

    def reset(self, n_arms):
        self.pulls=np.zeros(n_arms)
        self.rewards=np.zeros(n_arms)

    #Choosing which arm to pull based on UCB
    def choose_arm(self):
        n_arms=len(self.pulls)

        #For each arm, if an arm has never been pulled, choose it
        for n in range(n_arms):
            if self.pulls[n]==0:
                #Pull it
                return n

        #Compute
        #values=np.zeros(n_arms)
        def f(x, y, t, N):
            return kl_div(y,x)-np.log(1+t*(np.log(t)**2))/N

        mus=self.rewards

        values=np.zeros(n_arms)

        t=self.pulls.sum()

        for i, mu_hat in enumerate(mus):
            values[i]=bisect(f, mu_hat, 1000, args=(mu_hat, t, self.pulls[i]))

        return np.argmax(values+mus)

    def update_stats(self, chosen_arm, reward):
        self.pulls[chosen_arm]+=1
        n_pulls=self.pulls[chosen_arm]
        current_mean=self.rewards[chosen_arm]
        self.rewards[chosen_arm]=current_mean+(reward-current_mean)/n_pulls

#Evaluating an algorithm in the multi-armed bandit context
def evaluation(model, arms, n_runs, length):
    #Initialising decision and reward arrays
    selected_arms_data=np.zeros((n_runs, length))
    reward_data=np.zeros((n_runs, length))

    #For a number of runs
    for run in range(n_runs):
        #Resetting the model at the beginning of the run
        model.reset(len(arms))
        #For a number of pulls (per run)
        for t in range(length):
            #The model chooses an action
            action=model.choose_arm()
            #The action (arm selected) is saved
            selected_arm=arms[action]
            selected_arms_data[run, t]=action
            #The arm is pulled
            reward=selected_arm.pull()
            #The statisticss of the model are updated
            model.update_stats(action, reward)
            reward_data[run,t]=reward

    return selected_arms_data, reward_data

#In[]:
#Testing the algorithms
model=KL_UCB() #UCB()
arm_1=Bernoulli_bandit(0.1)
arm_2=Bernoulli_bandit(0.9)
arms=[arm_1, arm_2]

=evaluation(model, arms, 1, 10000)

#In[]:
#This section computes the expected regret for two arms:  Ber(0.5) and Ber(0.5+delta)
deltas=np.linspace(-0.5,0.5,num=51)
regrets=[]
steps=10000
runs=30
model_name="KL_UCB" #"UCB"

#For each delta
for delta in deltas:
    #Initialise the arms of the bandit
    arm_1=Bernoulli_bandit(0.5)
    arm_2=Bernoulli_bandit(0.5+delta)
    arms=[arm_1, arm_2]

    #Initialise the model to evaluate
    if model_name=="UCB":
        model=UCB()
    elif model_name=="KL_UCB":
        model=KL_UCB()

    selected_bandits_data, reward_data=evaluation(model, arms, runs, steps)

    #Calculate the mean reward per pull of the best arm
    max_mean=max(0.5, 0.5+delta)
    #Calculate the total expected reward of the best arms over the run
    max_target=max_mean*steps

    #Calculate regret
    regret=max_target-reward_data.sum(axis=1)
    #Compute mean regret over the runs and store
    regrets.append(regret)
    print("Finished arm: {}, mean regret:{}".format(delta+0.5, regret.mean()))

df_regret=pd.DataFrame(dict(zip(deltas,regrets)))
#df_regret.rename(columns=dict(zip(df_regret.columns,deltas)), inplace=True)

#Export the regret to csv
df_regret.to_csv("Regret_{}.csv".format(model_name))

#In[]:
#Plotting regret graphs

KL_UCB_regret=pd.read_csv("Regret_KL_UCB.csv")
UCB_regret=pd.read_csv("Regret_UCB.csv")


KL_UCB_regret.drop("Unnamed: 0", axis=1, inplace=True)
UCB_regret.drop("Unnamed: 0", axis=1, inplace=True)

x, p=ttest_ind(KL_UCB_regret, UCB_regret, equal_var=False)

colors=np.repeat("r",len(p))

colors[x<=0]="g"

x_axis=np.round(np.array(KL_UCB_regret.columns).astype("float"), 2)

plt.figure()
plt.bar(x_axis, height=p, width=0.02, color=colors)
plt.ylim((0,0.1))
plt.plot(x_axis, np.ones(len(p))*0.025, color="red")
plt.title("T-test results (null hypothesis: regret_UCB=<regret_KL_UCB)")
plt.ylabel("P-value")
plt.xlabel("Delta")
plt.savefig("ttest_regret_graph.png")
plt.show()


plt.figure()

plt.plot(x_axis, KL_UCB_regret.mean(), label="KL_UCB")
plt.plot(x_axis, UCB_regret.mean(), label="UCB", )
plt.legend()
plt.ylabel("Regret")
plt.xlabel("Deltas")
plt.title("Mean regret over 30 runs")

plt.savefig("mean_regret_graph.png")
plt.show()
