# https://github.com/keep9oing/DRQN-Pytorch-CartPole-v1
# https://ropiens.tistory.com/80
# https://github.com/chingyaoc/pytorch-REINFORCE/tree/master
# https://blog.naver.com/songblue61/221853600720
# While input training, x.shape: ([1, 300, 4]), hidden[0].shape: ([1, 1, 32]), hidden[1].shape: ([1, 1, 32])

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from mfrl_lib.lib import *

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class Pinet(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(Pinet, self).__init__()
        self.hidden_space = 32
        self.lstm = nn.LSTM(n_observations, self.hidden_space, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_space, self.hidden_space)
        self.actor = nn.Linear(self.hidden_space, n_actions)
        self.critic = nn.Linear(self.hidden_space, 1)
        
        self.init_weights()
        
    def init_weights(self):
        for layer in [self.fc1, self.actor, self.critic]:
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(layer.bias, 0)

        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                n = param.size(0)
                start, end = n//4, n//2
                param.data[start:end].fill_(1.)

    def pi(self, x, hidden):
        x, lstm_hidden = self.lstm(x, hidden)
        x = F.relu(self.fc1(x))
        x = self.actor(x)
        prob = F.softmax(x, dim=2)
        return prob, lstm_hidden
    
    def v(self, x, hidden):
        x, lstm_hidden = self.lstm(x, hidden)
        x = F.relu(self.fc1(x))
        v = self.critic(x)
        return v

    def sample_action(self, obs, h, c):
        prob, (h_new, c_new) = self.pi(obs, (h, c))
        m = Categorical(prob)
        action = m.sample()
        return prob, h_new, c_new, m.log_prob(action), m.entropy(), self.v(obs, (h, c))

class Agent:
    def __init__(self, topology, n_obs=N_OBSERVATIONS, n_act=N_ACTIONS):
        self.gamma = GAMMA
        self.topology = topology
        self.env = MFRLFullEnv(self)
        self.pinet = Pinet(n_obs, n_act).to(device)
        self.optimizer = optim.Adam(self.pinet.parameters(), lr=LEARNING_RATE)
        self.data = []
        
    def get_adjacent_ids(self):
        return np.where(self.topology.adjacency_matrix[self.id] == 1)[0]
    
    def get_adjacent_num(self):
        return len(self.get_adjacent_ids())
    
    def put_data(self, item):
        self.data.append(item)
        
    def get_age(self):
        return self.env.age.copy()

    def train(self):
        s, a, r, s_prime, done_mask = self.make_batch()
        h = (torch.zeros([1, 1, self.pinet.hidden_space], dtype=torch.float).to(device),
             torch.zeros([1, 1, self.pinet.hidden_space], dtype=torch.float).to(device))
        
        v_prime = self.pinet.v(s_prime, h).squeeze(1).detach()
        td_target = r + self.gamma * v_prime * done_mask
        v_s = self.pinet.v(s, h).squeeze(1)
        delta = td_target - v_s
        advantage = delta.detach()

        pi, _ = self.pinet.pi(s, h)
        pi_a = pi.squeeze(0).gather(1, a.squeeze(1))
        loss = -torch.log(pi_a) * advantage.unsqueeze(1) + F.smooth_l1_loss(v_s, td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(self.pinet.parameters(), MAX_GRAD_NORM)
        self.optimizer.step()
        self.data = []

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s = torch.tensor(s_lst, dtype=torch.float).to(device)
        a = torch.tensor(a_lst).to(device)
        r = torch.tensor(r_lst).to(device)
        s_prime = torch.tensor(s_prime_lst, dtype=torch.float).to(device)
        done_mask = torch.tensor(done_lst, dtype=torch.float).to(device)
        
        s = s.view([1, len(self.data), -1])
        s_prime = s_prime.view([1, len(self.data), -1])
        return s, a, r, s_prime, done_mask


if __name__ == "__main__":
    # Summarywriter setting
    timestamp = FIXED_TIMESTAMP
    output_path = 'outputs'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    writer = SummaryWriter(output_path + "/" + "RA2Cfull" + "_" + topo_string + "_" + timestamp)

    # Parameter overwriting
    n_obs = 2*node_n+1
    n_act = N_ACTIONS**node_n
    # MAX_EPISODES = 10
    
    # Make agents
    agent = Agent(topology, n_obs=n_obs, n_act=n_act)

    reward_data = []

    for n_epi in tqdm(range(MAX_EPISODES), desc="Episodes", position=0, leave=True):
        episode_utility = 0.0
        observation = agent.env.reset(seed=GLOBAL_SEED)[0]
        h = torch.zeros(1, 1, 32).to(device)
        c = torch.zeros(1, 1, 32).to(device)
        done = False
        
        for t in tqdm(range(MAX_STEPS), desc="  Steps", position=1, leave=False):
            actions = []
            max_aoi = []
            
            prob, h, c, log_prob, entropy, value = agent.pinet.sample_action(
                torch.from_numpy(observation.astype('float32')).unsqueeze(0).to(device), h, c)
            m = Categorical(prob)
            a = m.sample().item()
            actions.append(a)
            
            next_observation, reward, done, _, _ = agent.env.step(actions)
            curr_age = agent.get_age()
            observation = next_observation
            episode_utility += reward
            reward_data.append({'episode': n_epi, 'step': t, 'reward': reward, 'action': decimal_to_binary_array([a], node_n), 'age': curr_age})
            agent.put_data((observation, actions, reward, next_observation, done))
                
            if done:
                break
        
        agent.train()
        
        writer.add_scalar('Avg. Rewards per episodes', episode_utility, n_epi)

        if n_epi % print_interval == 0:
            print(f"# of episode :{n_epi}, avg reward : {episode_utility:.1f}")

    # Save rewards to DataFrame and CSV
    reward_df = pd.DataFrame(reward_data)
    action_cols = reward_df['action'].apply(pd.Series)
    action_cols.columns = [f'action_{i}' for i in range(node_n)]
    age_cols = reward_df['age'].apply(pd.Series)
    age_cols.columns = [f'age_{i}' for i in range(node_n)]
    result_df = pd.concat([reward_df.drop(['action', 'age'], axis=1), action_cols, age_cols], axis=1)
    result_df.to_csv(f'RA2Cfull_{topo_string}_{timestamp}.csv', index=False)
    writer.close()

    # Save models
    model_path = 'models'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    save_model(agent.pinet, f'{model_path}/RA2Cfull_{topo_string}_{timestamp}.pth')
