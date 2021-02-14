import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNet(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(obs_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, n_actions)

    def forward(self, state):
        # ====================================================
        x=self.relu(self.fc1(state))
        x=self.relu(self.fc2(x))
        Q=self.fc3(x)
        # ====================================================
        return Q

    def select_greedyaction(self, state):
        with torch.no_grad():
            # ====================================================
            Q=q_net.forward(state)
            action_index=torch.argmax(Q)
            # ====================================================
        return action_index.item()

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, sample):
        """Saves a transition.
            sample is a tuple (state, next_state, action, reward, done)
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = sample
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch_size = min(len(self.memory), batch_size)
        samples = random.sample(self.memory, batch_size)
        return map(np.asarray, zip(*samples))

    def __len__(self):
        return len(self.memory)

def eval_dqn(env, qnet, n_sim=5):
    """
    Monte Carlo evaluation of DQN agent
    """
    rewards = np.zeros(n_sim)
    copy_env = deepcopy(env) # Important!
    # Loop over number of simulations
    for sim in range(n_sim):
        state = copy_env.reset()
        done = False
        while not done:
            tensor_state = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = qnet.select_greedyaction(tensor_state)
            next_state, reward, done, _ = copy_env.step(action)
            # update sum of rewards
            rewards[sim] += reward
            state = next_state
    return rewards

# Discount factor
GAMMA = 0.99
EVAL_EVERY = 2

# Batch size
BATCH_SIZE = 256
# Capacity of the replay buffer
BUFFER_CAPACITY = 30000
# Update target net every ... episodes
UPDATE_TARGET_EVERY = 20

# Initial value of epsilon
EPSILON_START = 1.0
# Parameter to decrease epsilon
DECREASE_EPSILON = 200
# Minimum value of epislon
EPSILON_MIN = 0.05

# Number of training episodes
N_EPISODES = 250

# Learning rate
LEARNING_RATE = 0.0001

results=pd.DataFrame()

for i in range(20):
    env = gym.make('CartPole-v0')

    # initialize replay buffer
    replay_buffer = ReplayBuffer(BUFFER_CAPACITY)

    # create network and target network
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    # ====================================================
    # YOUR IMPLEMENTATION HERE
    q_net=QNet(obs_size, n_actions)
    target_net=QNet(obs_size, n_actions)

    # ====================================================

    # objective and optimizer
    optimizer = optim.Adam(params=q_net.parameters(), lr=LEARNING_RATE)

    # Algorithm
    state = env.reset()
    epsilon = EPSILON_START
    ep = 0
    total_time = 0
    learn_steps = 0
    episode_reward = 0

    episode_rewards = np.zeros((N_EPISODES, 3))
    while ep < N_EPISODES:
        # ====================================================
        # YOUR IMPLEMENTATION HERE
        # sample epsilon-greedy action
        p = random.random()
        if p < epsilon:
            action = env.action_space.sample()
        else:
            tensor_state = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = q_net.select_greedyaction(tensor_state)
        # ====================================================

        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        total_time += 1

        # ====================================================
        # YOUR IMPLEMENTATION HERE
        sample=(state, next_state, action, reward, done)
        replay_buffer.push(sample)
        # ====================================================


        if len(replay_buffer) > BATCH_SIZE:
            learn_steps += 1
            # UPDATE MODEL
            # ====================================================
            # YOUR IMPLEMENTATION HERE
            # get batch
            sample=replay_buffer.sample(BATCH_SIZE)
            batch_state, batch_next_state, batch_action, batch_reward, batch_done = sample
            # ====================================================


            batch_state = torch.FloatTensor(batch_state).to(device)
            batch_next_state = torch.FloatTensor(batch_next_state).to(device)
            batch_action = torch.FloatTensor(batch_action).unsqueeze(1).to(device)
            batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)
            batch_done = torch.FloatTensor(batch_done).unsqueeze(1).to(device)

            with torch.no_grad():
                # ====================================================
                # YOUR IMPLEMENTATION HERE
                # build target (recall that we conseder the Q function
                # in the next state only if not terminal, ie. done != 1)
                # (1- done) * value_next
                next_Q = torch.max(target_net.forward(batch_next_state), dim=1)[0].unsqueeze(1)
                targets=(1-batch_done)*(next_Q * GAMMA) + batch_reward
                # ====================================================

            values = q_net(batch_state).gather(1, batch_action.long())

            # ====================================================
            # YOUR IMPLEMENTATION HERE
            # compute loss and update model (loss and optimizer)
            loss = nn.MSELoss()
            output=loss(values, targets)
            optimizer.zero_grad()
            output.backward()
            optimizer.step()
            # ====================================================

            if epsilon > EPSILON_MIN:
                epsilon -= (EPSILON_START - EPSILON_MIN) / DECREASE_EPSILON

        # ====================================================
        # YOUR IMPLEMENTATION HERE
        # update target network
        if learn_steps % UPDATE_TARGET_EVERY  == 0:
            target_net.load_state_dict(q_net.state_dict())
        # ====================================================

        state = next_state
        if done:
            mean_rewards = -1
            if (ep+1) % EVAL_EVERY == 0:
                # evaluate current policy
                rewards = eval_dqn(env, q_net)
                mean_rewards = np.mean(rewards)
                print("episode =", ep, ", reward = ", np.round(np.mean(rewards),2), ", obs_rew = ", episode_reward)
                # if np.mean(rewards) >= REWARD_THRESHOLD:
                #     break

            episode_rewards[ep] = [total_time, episode_reward, mean_rewards]
            state = env.reset()
            ep += 1
            episode_reward = 0

    results["Run_%s_learning" % (i)]=episode_rewards[:,1]
    results["Run_%s_testing" % (i)]=episode_rewards[:,2]
    results.to_csv("DQN_results.csv")

###################################################################
# VISUALIZATION
###################################################################
for episode in range(3):
    done = False
    state = env.reset()
    env.render()
    while not done:
        tensor_state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = q_net.select_greedyaction(tensor_state)
        state, reward, done, info = env.step(action)
        env.render()

plt.figure()
plt.title('Performance over learning')
plt.plot(episode_rewards[:,0], episode_rewards[:,1])
plt.xlabel('time steps')
plt.ylabel('total reward')

plt.figure()
plt.title('Performance on Test Env')
xv = np.arange(EVAL_EVERY-1, N_EPISODES+1, EVAL_EVERY)
plt.plot(episode_rewards[xv, 0], episode_rewards[xv, 2], ':o')
plt.xlabel('time steps')
plt.ylabel('expected total reward (greedy policy)')
plt.show()
