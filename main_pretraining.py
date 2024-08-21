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
        self.lstm = nn.LSTM(self.hidden_space, self.hidden_space, batch_first=True)
        self.actor = nn.Linear(self.hidden_space, n_actions)
        self.critic = nn.Linear(self.hidden_space, 1)
        
        self.init_weights()
        
    def init_weights(self):
        # fc1 layer (ReLU activation)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0)

        # LSTM layer
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                # Set forget gate bias to 1
                n = param.size(0)
                start, end = n//4, n//2
                param.data[start:end].fill_(1.)

        # Actor layer
        nn.init.kaiming_normal_(self.actor.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.actor.bias, 0)

        # Critic layer
        nn.init.kaiming_normal_(self.critic.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.critic.bias, 0)

    def forward(self, x, h, c):
        x = F.relu(self.fc1(x))
        x, (h_new, c_new) = self.lstm(x, (h, c))
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value, h_new, c_new

    def sample_action(self, obs, h, c):
        policy, value, h, c = self.forward(obs, h, c)
        probs = F.softmax(policy, dim=2)
        m = Categorical(probs)
        action = m.sample()
        return probs, h, c, m.log_prob(action), m.entropy(), value

    def init_hidden_state(self):
        return (torch.zeros([1, 1, self.hidden_space], device=device), 
                torch.zeros([1, 1, self.hidden_space], device=device))
    
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
        actor_loss = []
        critic_loss = []
        entropy_term = 0
        self.optimizer.zero_grad()
        for r, value, log_prob, entropy in reversed(self.data):
            R = r + self.gamma * R
            advantage = R - value.item()
            actor_loss.append(-log_prob * advantage)
            critic_loss.append(F.smooth_l1_loss(value, torch.tensor([[[R]]]).to(device)))
            entropy_term += entropy
        
        actor_loss = torch.stack(actor_loss).sum()
        critic_loss = torch.stack(critic_loss).sum()
        entropy_term = entropy_term.mean()
        
        loss = actor_loss + CRITIC_COEFF*critic_loss - ENTROPY_COEFF*entropy_term
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.pinet.parameters(), MAX_GRAD_NORM)
        
        self.optimizer.step()
        self.data = []

if __name__ == "__main__":
    # Summarywriter setting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = 'outputs'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    writer = SummaryWriter(output_path + "/" + "RA2C" + "_" + timestamp)

    # Make agents
    agents = [Agent(topology, i) for i in range(node_n)]

    # DataFrame to store rewards
    # reward_data = []

    for n_epi in tqdm(range(MAX_EPISODES), desc="Episodes", position=0, leave=True):
        episode_utility = 0.0
        observation = [agent.env.reset()[0] for agent in agents]
        h = [torch.zeros(1, 1, 32).to(device) for _ in range(node_n)]
        c = [torch.zeros(1, 1, 32).to(device) for _ in range(node_n)]
        done = [False] * node_n
        
        for t in tqdm(range(MAX_STEPS), desc="  Steps", position=1, leave=False):
            actions = []
            log_probs = []
            entropies = []
            values = []
            probs = [0]*node_n

            for agent_id, agent in enumerate(agents):
                probs[agent_id], h[agent_id], c[agent_id], log_prob, entropy, value = agent.pinet.sample_action(
                    torch.from_numpy(observation[agent_id].astype('float32')).unsqueeze(0).to(device), 
                    h[agent_id], 
                    c[agent_id]
                )
                m = Categorical(probs[agent_id])
                a = m.sample()
                actions.append(a.item())
                log_probs.append(log_prob)
                entropies.append(entropy)
                values.append(value)
                reward_data.append({'episode': n_epi, 'step': t, 'agent_id': agent_id, 'prob of 1': value.item()})
            
            for agent in agents:
                agent.env.set_all_actions(actions)
            
            for agent_id, agent in enumerate(agents):
                next_observation, reward, done[agent_id], _, _ = agent.env.step(actions[agent_id])
                # agent.put_data((reward, probs[agent_id][0, 0, actions[agent_id]]))
                observation[agent_id] = next_observation
                episode_utility += reward
                agent.put_data((reward, values[agent_id], log_probs[agent_id], entropies[agent_id]))
                
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
    df_pivot.to_csv(f'RA2C_{timestamp}.csv', index=False)
    writer.close()

    # Save models
    model_path = 'models'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    for i, agent in enumerate(agents):
        save_model(agent.pinet, f'{model_path}/RA2C_agent_{i}_{timestamp}.pth')