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
    def __init__(self, topology, id):
        self.gamma = GAMMA
        self.topology = topology
        if id >= topology.n:
            raise ValueError("id must be less than n.")
        self.id = id
        self.env = MFRLEnv(self)
        self.pinet = Pinet(N_OBSERVATIONS, N_ACTIONS).to(device)
        self.optimizer = optim.Adam(self.pinet.parameters(), lr=LEARNING_RATE)
        self.data = []
        
    def get_adjacent_ids(self):
        return np.where(self.topology.adjacency_matrix[self.id] == 1)[0]
    
    def get_adjacent_num(self):
        return len(self.get_adjacent_ids())
    
    def put_data(self, item):
        self.data.append(item)

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
        pi_a = pi.squeeze(0).gather(1, a)
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
    writer = SummaryWriter(output_path + "/" + "RA2C" + "_" + topo_string + "_" + timestamp)

    # Parameter overwriting
    # MAX_EPISODES = 10

    # Make agents
    agents = [Agent(topology, i) for i in range(node_n)]

    reward_data = []

    for n_epi in tqdm(range(MAX_EPISODES), desc="Episodes", position=0, leave=True):
        episode_utility = 0.0
        observation = [agent.env.reset()[0] for agent in agents]
        h = [torch.zeros(1, 1, 32).to(device) for _ in range(node_n)]
        c = [torch.zeros(1, 1, 32).to(device) for _ in range(node_n)]
        done = [False] * node_n
        
        for t in tqdm(range(MAX_STEPS), desc="  Steps", position=1, leave=False):
            actions = []
            max_aoi = []
            for agent_id, agent in enumerate(agents):
                prob, h[agent_id], c[agent_id], log_prob, entropy, value = agent.pinet.sample_action(
                    torch.from_numpy(observation[agent_id].astype('float32')).unsqueeze(0).to(device), 
                    h[agent_id], 
                    c[agent_id]
                )
                m = Categorical(prob)
                a = m.sample().item()
                actions.append(a)
                reward_data.append({'episode': n_epi, 'step': t, 'agent_id': agent_id, 'prob of 1': prob[0, 0, 1].item()})
            
            for agent in agents:
                agent.env.set_all_actions(actions)
                max_aoi.append(agent.env.get_maxaoi())
            
            for agent_id, agent in enumerate(agents):
                agent.env.set_max_aoi(max_aoi)
                next_observation, reward, done[agent_id], _, _ = agent.env.step(actions[agent_id])
                observation[agent_id] = next_observation
                episode_utility += reward
                agent.put_data((observation[agent_id], actions[agent_id], reward, next_observation, done[agent_id]))
                
            if all(done):
                break
                
        for agent in agents:
            agent.train()
        episode_utility /= node_n
        writer.add_scalar('Avg. Rewards per episodes', episode_utility, n_epi)

        if n_epi % print_interval == 0:
            print(f"# of episode :{n_epi}, avg reward : {episode_utility:.1f}")

    # Save rewards to DataFrame and CSV
    reward_df = pd.DataFrame(reward_data)
    df_pivot = reward_df.pivot_table(index=['episode', 'step'], columns='agent_id', values='prob of 1').reset_index()
    df_pivot.columns = ['episode', 'step'] + [f'agent_{col}' for col in df_pivot.columns[2:]]
    df_pivot.to_csv(f'RA2C_{topo_string}_{timestamp}.csv', index=False)
    writer.close()

    # Save models
    model_path = 'models'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    for i, agent in enumerate(agents):
        save_model(agent.pinet, f'{model_path}/RA2C_{topo_string}_agent_{i}_{timestamp}.pth')
