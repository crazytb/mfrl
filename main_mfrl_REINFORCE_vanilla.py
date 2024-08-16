# https://github.com/keep9oing/DRQN-Pytorch-CartPole-v1
# https://ropiens.tistory.com/80
# https://github.com/chingyaoc/pytorch-REINFORCE/tree/master

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
        self.fc1 = nn.Linear(n_observations, self.hidden_space)
        self.fc2 = nn.Linear(self.hidden_space, self.hidden_space)
        self.fc3 = nn.Linear(self.hidden_space, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=2)
        return x

    def sample_action(self, obs):
        obs = obs.unsqueeze(0)
        out = self.forward(obs)
        m = Categorical(out)
        action = m.sample()
        return action.item()
    
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
        R = 0
        loss = []
        returns = []
        self.optimizer.zero_grad()
        for r, log_prob in reversed(self.data):
            R = r + self.gamma * R
            returns.insert(0, R)
            loss.append(-log_prob * R)
        
        policy_loss = torch.stack(loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        self.data = []

if __name__ == "__main__":

    # Summarywriter setting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = 'outputs'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    writer = SummaryWriter(output_path + "/" + "REINFORCE_vanilla" + "_" + timestamp)

    # # Make topology
    # topology = Topology(4, "dumbbell")
    # topology.show_adjacency_matrix()
    # node_n = topology.n

    # Make agents
    agents = [Agent(topology, i) for i in range(node_n)]

    # DataFrame to store rewards
    # reward_data = []

    for n_epi in tqdm(range(MAX_EPISODES), desc="Episodes", position=0, leave=True):
        episode_utility = 0.0
        observation = [agent.env.reset()[0] for agent in agents]
        done = [False] * node_n

        for t in tqdm(range(MAX_STEPS), desc="  Steps", position=1, leave=False):
            actions = []
            probs = [0]*node_n
            for agent_id, agent in enumerate(agents):
                probs[agent_id] = agent.pinet(
                    torch.from_numpy(observation[agent_id].astype('float32')).unsqueeze(0).to(device)
                    )
                m = Categorical(probs[agent_id])
                a = m.sample()
                actions.append(a.item())
                reward_data.append({'episode': n_epi, 'step': t, 'agent_id': agent_id, 'prob of 1': probs[agent_id][0, 0, -1].item()})
            
            for agent in agents:
                agent.env.set_all_actions(actions)
            
            for agent_id, agent in enumerate(agents):
                next_observation, reward, done[agent_id], _, _ = agent.env.step(actions[agent_id])
                # agent.put_data((reward, probs[agent_id][0, 0, actions[agent_id]]))
                observation[agent_id] = next_observation
                episode_utility += reward
                agent.put_data((reward, probs[agent_id][0, 0, actions[agent_id]]))
            
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
    df_pivot.to_csv(f'REINFORCE_vanilla_{timestamp}.csv', index=False)
    writer.close()

    # Save models
    model_path = 'models'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    for i, agent in enumerate(agents):
        save_model(agent.pinet, f'{model_path}/REINFORCE_vanilla_agent_{i}_{timestamp}.pth')