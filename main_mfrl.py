# https://github.com/keep9oing/DRQN-Pytorch-CartPole-v1
# https://ropiens.tistory.com/80
# https://github.com/chingyaoc/pytorch-REINFORCE/tree/master

import sys
import os
import csv
from typing import Dict
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import matplotlib.pyplot as plt
import random
import collections 
import itertools


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Device: {device}")


class Topology():
    def __init__(self, n, model="random", density=1):
        self.n = n
        self.model = model
        self.density = density
        self.adjacency_matrix = self.make_adjacency_matrix()
        
    def make_adjacency_matrix(self) -> np.ndarray:
        if self.density < 0 or self.density > 1:
            raise ValueError("Density must be between 0 and 1.")

        n_edges = int(self.n * (self.n - 1) / 2 * self.density)
        adjacency_matrix = np.zeros((self.n, self.n))

        if self.model == "dumbbell":
            adjacency_matrix[0, self.n-1] = 1
            adjacency_matrix[self.n-1, 0] = 1
            for i in range(1, self.n//2):
                adjacency_matrix[0, i] = 1
                adjacency_matrix[i, 0] = 1
            for i in range(self.n//2+1, self.n):
                adjacency_matrix[i-1, self.n-1] = 1
                adjacency_matrix[self.n-1, i-1] = 1
        elif self.model == "linear":
            for i in range(1, self.n):
                adjacency_matrix[i-1, i] = 1
                adjacency_matrix[i, i-1] = 1
        elif self.model == "random":
            for i in range(1, self.n):
                adjacency_matrix[i-1, i] = 1
                adjacency_matrix[i, i-1] = 1
                n_edges -= 1
            if n_edges <= 0:
                return adjacency_matrix
            else:
                arr = [1]*n_edges + [0]*((self.n-1)*(self.n-2)//2 - n_edges)
                np.random.shuffle(arr)
                for i in range(0, self.n):
                    for j in range(i+2, self.n):
                        adjacency_matrix[i, j] = arr.pop()
                        adjacency_matrix[j, i] = adjacency_matrix[i, j]
        else:
            raise ValueError("Model must be dumbbell, linear, or random.")
        return adjacency_matrix

    def show_adjacency_matrix(self):
        print(self.adjacency_matrix)
        
    def get_density(self):
        return np.sum(self.adjacency_matrix) / (self.n * (self.n - 1))
    
    def save_graph_with_labels(self, path):
        rows, cols = np.where(self.adjacency_matrix == 1)
        edges = zip(rows.tolist(), cols.tolist())
        G = nx.Graph()
        G.add_edges_from(edges)
        pos = nx.kamada_kawai_layout(G)
        nx.draw_networkx(G, pos=pos, with_labels=True)
        plt.savefig(path + '/adj_graph.png')


class MFRLEnv(gym.Env):
    actions = np.array([])
    def __init__(self, agent):
        self.id = agent.id
        self.all_num = agent.topology.n
        self.adj_num = agent.get_adjacent_num()
        self.adj_ids = agent.get_adjacent_ids()
        self.adj_obs = {adj_id: [0, 0] for adj_id in self.adj_ids}
        self.counter = 0
        self.age = 0

        self.observation_space = spaces.Box(low=0, high=1, shape=(1, 2))
        self.action_space = spaces.Discrete(2)
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        observation = np.array([[0.5, 0.5]])
        # observation = [[0.5, 0.5]]
        info = {}
        MFRLEnv.actions = np.zeros(self.all_num)
        self.counter = 0
        self.age = 0
        return observation, info
    
    def gather_actions(self, action):
        MFRLEnv.actions = np.array(action)
        
    def calculate_meanfield(self):  
        if self.idle_check():
            return np.array([[0.0, 1.0]])
        else:
            return np.array([self.adj_num*(2**self.adj_num)*np.array([0.5, 0.5]) - self.adj_num*np.array([1, 0])])/(self.adj_num*(2**self.adj_num)-self.adj_num)

    def idle_check(self):
        if all(MFRLEnv.actions[self.adj_ids] == 0):
            return True
        else:
            return False
        
    def get_adjacent_nodes(self, *args):
        if len(args) > 0:
            return np.where(topology.adjacency_matrix[args[0]] == 1)[0]
        else:
            return np.where(topology.adjacency_matrix[self.id] == 1)[0]
    
    def step(self, action):
        observation = self.calculate_meanfield()
        self.age += 1/MAX_STEPS
        if action == 1:
            adjacent_nodes = self.get_adjacent_nodes()
            for j in adjacent_nodes:
                js_adjacent_nodes = self.get_adjacent_nodes(j)
                js_adjacent_nodes_except_ind = js_adjacent_nodes[js_adjacent_nodes != self.id]
                if (np.all(MFRLEnv.actions[js_adjacent_nodes_except_ind] == 0)
                    and MFRLEnv.actions[j] == 0):
                    # reward = np.log2(self.age+1)
                    reward = np.tanh(5*self.age)
                    self.age = 0
                    break
                else:
                    reward = 0
        else:
            reward = 0
        self.counter += 1
        terminated = False
        info = {}
        if self.counter == MAX_STEPS:
            terminated = True
        return observation, reward, terminated, False, info


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=BUFFER_LIMIT)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n, updates="sequential"):
        if updates == "random":
            mini_batch = random.sample(self.buffer, n)
        elif updates == "sequential":
            # epi_len is the number of entries that has done_mask = 0
            epi_idx = random.randint(0, n_epi-1)
            mini_batch = list(itertools.islice(self.buffer, epi_idx, epi_idx+n))

        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])
        return (torch.tensor(np.stack(s_lst), dtype=torch.float32).to(device),
                torch.tensor(a_lst).to(device),
                torch.tensor(r_lst).to(device),
                torch.tensor(np.stack(s_prime_lst), dtype=torch.float32).to(device),
                torch.tensor(done_mask_lst).to(device))
    
    def size(self):
        return len(self.buffer)
    
    # Export the buffer to csv file into path
    def export_buffer(self, path, suffix=''):
        df = pd.DataFrame(self.buffer, columns=['s', 'a', 'r', 's_prime', 'done_mask'])
        df.to_csv(path + f'/{timestamp}_{suffix}.csv', index=False)
    

class Qnet(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(Qnet, self).__init__()
        self.hidden_space = 128
        self.fc1 = nn.Linear(n_observations, self.hidden_space)
        # self.fc2 = nn.Linear(128, 128)
        self.fc2 = nn.LSTM(self.hidden_space, self.hidden_space, batch_first=True)
        self.fc3 = nn.Linear(self.hidden_space, n_actions)

    def forward(self, x, h, c):
        x = F.relu(self.fc1(x))
        # h = h.view(h.size(0), -1)
        # c = c.view(c.size(0), -1)
        x, (h_new, c_new) = self.fc2(x, (h, c))
        x = self.fc3(x)
        return x, h_new, c_new

    def sample_action(self, obs, h, c, epsilon):
        obs = obs.unsqueeze(0)
        out, h, c = self.forward(obs, h, c)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1), h, c
        else: 
            return out.argmax().item(), h, c
        
    def init_hidden_state(self, batch_size, training=None):
        assert training is not None, "training step parameter should be determined"
        if training is True:
            return (torch.zeros([1, batch_size, self.hidden_space]), 
                    torch.zeros([1, batch_size, self.hidden_space]))
        else:
            return (torch.zeros([1, 1, self.hidden_space]), 
                    torch.zeros([1, 1, self.hidden_space]))

        
    # def init_hidden_state(self, batch_size=300):
    #     return torch.zeros(batch_size, 1, 128).to(device), torch.zeros(batch_size, 1, 128).to(device)

class EpisodeMemory():
    """Episode memory for recurrent agent"""

    def __init__(self, random_update=False, 
                       max_epi_num=100, max_epi_len=500,
                       batch_size=1,
                       lookup_step=None):
        self.random_update = random_update # if False, sequential update
        self.max_epi_num = max_epi_num
        self.max_epi_len = max_epi_len
        self.batch_size = batch_size
        self.lookup_step = lookup_step

        if (random_update is False) and (self.batch_size > 1):
            sys.exit('..')

        self.memory = collections.deque(maxlen=self.max_epi_num)

    def put(self, episode):
        self.memory.append(episode)

    def sample(self):
        sampled_buffer = []

        ##################### RANDOM UPDATE ############################
        if self.random_update: # Random upodate
            sampled_episodes = random.sample(self.memory, self.batch_size)
            
            check_flag = True # check if every sample data to train is larger than batch size
            min_step = self.max_epi_len

            for episode in sampled_episodes:
                min_step = min(min_step, len(episode)) # get minimum step from sampled episodes

            for episode in sampled_episodes:
                if min_step > self.lookup_step: # sample buffer with lookup_step size
                    idx = np.random.randint(0, len(episode)-self.lookup_step+1)
                    sample = episode.sample(random_update=self.random_update, lookup_step=self.lookup_step, idx=idx)
                    sampled_buffer.append(sample)
                else:
                    idx = np.random.randint(0, len(episode)-min_step+1) # sample buffer with minstep size
                    sample = episode.sample(random_update=self.random_update, lookup_step=min_step, idx=idx)
                    sampled_buffer.append(sample)

        ##################### SEQUENTIAL UPDATE ############################           
        else: # Sequential update
            idx = np.random.randint(0, len(self.memory))
            sampled_buffer.append(self.memory[idx].sample(random_update=self.random_update))

        return sampled_buffer, len(sampled_buffer[0]['obs']) # buffers, sequence_length

    def export(self, path):
        for i, episode in enumerate(self.memory):
            episode.replaybuffer.export_buffer(path, suffix=f'episode_{i}')
    
    def __len__(self):
        return len(self.memory)


class EpisodeBuffer:
    """A simple numpy replay buffer."""

    def __init__(self):
        self.obs = []
        self.action = []
        self.reward = []
        self.next_obs = []
        self.done = []

    def put(self, transition):
        self.obs.append(transition[0])
        self.action.append(transition[1])
        self.reward.append(transition[2])
        self.next_obs.append(transition[3])
        self.done.append(transition[4])

    def sample(self, random_update=False, lookup_step=None, idx=None) -> Dict[str, np.ndarray]:
        obs = np.array(self.obs)
        action = np.array(self.action)
        reward = np.array(self.reward)
        next_obs = np.array(self.next_obs)
        done = np.array(self.done)

        if random_update is True:
            obs = obs[idx:idx+lookup_step]
            action = action[idx:idx+lookup_step]
            reward = reward[idx:idx+lookup_step]
            next_obs = next_obs[idx:idx+lookup_step]
            done = done[idx:idx+lookup_step]

        return dict(obs=obs,
                    acts=action,
                    rews=reward,
                    next_obs=next_obs,
                    done=done)

    def __len__(self) -> int:
        return len(self.obs)


class Agent:
    episode_memory = EpisodeMemory(random_update=False,
                                   max_epi_num=100, 
                                   max_epi_len=500,
                                   batch_size=1,
                                   lookup_step=5)
    def __init__(self, topology, id):
        self.gamma = GAMMA
        self.topology = topology
        if id >= topology.n:
            raise ValueError("id must be less than n.")
        else:
            self.id = id
        self.env = MFRLEnv(self)
        self.replaybuffer = EpisodeBuffer()
        # self.episode_memory = EpisodeMemory(random_update=False,
        #                                     max_epi_num=100, 
        #                                     max_epi_len=500,
        #                                     batch_size=1,
        #                                     lookup_step=5)
        self.qnet = Qnet(N_OBSERVATIONS, N_ACTIONS).to(device)
        self.targetnet = Qnet(N_OBSERVATIONS, N_ACTIONS).to(device)
        self.targetnet.load_state_dict(self.qnet.state_dict())
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=0.0005)
        
    def get_adjacent_ids(self):
        return np.where(self.topology.adjacency_matrix[self.id] == 1)[0]
    
    def get_adjacent_num(self):
        return len(self.get_adjacent_ids())
    
    def train(self):
        samples, seq_len = self.episode_memory.sample()

        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []

        batch_size = 1
        
        for i in range(batch_size):
            observations.append(samples[i]["obs"])
            actions.append(samples[i]["acts"])
            rewards.append(samples[i]["rews"])
            next_observations.append(samples[i]["next_obs"])
            dones.append(samples[i]["done"])

        observations = np.array(observations)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_observations = np.array(next_observations)
        dones = np.array(dones)

        observations = torch.FloatTensor(observations.reshape(batch_size,seq_len,-1)).to(device)
        actions = torch.LongTensor(actions.reshape(batch_size,seq_len,-1)).to(device)
        rewards = torch.FloatTensor(rewards.reshape(batch_size,seq_len,-1)).to(device)
        next_observations = torch.FloatTensor(next_observations.reshape(batch_size,seq_len,-1)).to(device)
        dones = torch.FloatTensor(dones.reshape(batch_size,seq_len,-1)).to(device)

        h_target, c_target = self.targetnet.init_hidden_state(batch_size=batch_size, training=True)

        q_target, _, _ = self.targetnet(next_observations, h_target.to(device), c_target.to(device))

        q_target_max = q_target.max(2)[0].view(batch_size,seq_len,-1).detach()
        targets = rewards + self.gamma*q_target_max*dones

        h, c = self.qnet.init_hidden_state(batch_size=batch_size, training=True)
        q_out, _, _ = self.qnet(observations, h.to(device), c.to(device))
        q_a = q_out.gather(2, actions)

        # Multiply Importance Sampling weights to loss        
        loss = F.smooth_l1_loss(q_a, targets)
        
        # Update Network
        max_grad_norm=1.0
        self.optimizer.zero_grad()
        loss.backward()
        # Add gradient clipping here
        clip_grad_norm_(self.qnet.parameters(), max_grad_norm)
        self.optimizer.step()


def append_to_csv(filename, id, obs, action, reward, next_obs, done):
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header if file is empty
        if csvfile.tell() == 0:
            writer.writerow(['agent_id', 'obs', 'action', 'reward', 'next_obs', 'done'])
        
        # Write data row
        writer.writerow([id] + obs.tolist() + [action, reward] + next_obs.tolist() + [done])
    

# Summarywriter setting
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = 'outputs'
if not os.path.exists(output_path):
    os.makedirs(output_path)
writer = SummaryWriter(output_path + f"/{timestamp}")

# Make topology
# topology = Topology(12, "dumbbell", 0.5)
topology = Topology(4, "random", 1)
topology.show_adjacency_matrix()
node_n = topology.n
N_OBSERVATIONS = 2
N_ACTIONS = 2

# Hyperparameters
MAX_STEPS = 300
BUFFER_LIMIT  = 10000
GAMMA = 0.98

# Make agents
agents = [Agent(topology, i) for i in range(node_n)]
# check_env(agents[0].env)

MAX_EPISODES = 2
epsilon = 0.1
print_interval = 10

# DataFrame to store rewards
reward_data = []

for n_epi in tqdm(range(MAX_EPISODES), desc="Episodes", position=0, leave=True):
    epsilon = max(0.01, 0.08 - 0.01*(n_epi/100)) # Linear annealing from 8% to 1%
    episode_utility = 0.0  # Initialize episode score
    observation = [agent.env.reset()[0] for agent in agents]
    done = False
    next_observation = [0]*node_n
    reward = [0]*node_n
    action = [0]*node_n
    h = [torch.zeros(1, 1, 128).to(device) for _ in range(node_n)]
    c = [torch.zeros(1, 1, 128).to(device) for _ in range(node_n)]
    
    for t in tqdm(range(MAX_STEPS), desc="   Steps", position=1, leave=False):
        for agent_id, agent in enumerate(agents):
            action[agent_id], h[agent_id], c[agent_id] = agent.qnet.sample_action(torch.from_numpy(observation[agent_id]).float().to(device), 
                                                                                  h[agent_id], 
                                                                                  c[agent_id], 
                                                                                  epsilon)
        for agent in agents:
            agent.env.gather_actions(action)
        for agent_id, agent in enumerate(agents):
            next_observation[agent_id], reward[agent_id], done, _, _ = agent.env.step(action[agent_id])
            done_mask = 0.0 if done else 1.0
            agent.replaybuffer.put((observation[agent_id], 
                                    action[agent_id], 
                                    reward[agent_id], 
                                    next_observation[agent_id], 
                                    done_mask))
            # Append to CSV file
            append_to_csv(f'transition_data.csv', 
                          agent_id,
                          observation[agent_id], 
                          action[agent_id], 
                          reward[agent_id], 
                          next_observation[agent_id], 
                          done_mask)
            observation[agent_id] = next_observation[agent_id]
            episode_utility += reward[agent_id]
            
            # Append reward data
            reward_data.append({'episode': n_epi, 'step': t, 'agent_id': agent_id, 'reward': reward[agent_id]})

            # if agent.replaybuffer.size() >= BUFFER_LIMIT:
            # if n_epi >= 3:

        for agent in agents:
            agent.episode_memory.put(agent.replaybuffer)
    
    for agent in agents:
        agent.train()
    
    writer.add_scalar('Rewards per episodes', episode_utility, n_epi)

    if n_epi % print_interval == 0 and n_epi != 0:
        for i in range(node_n):
            agents[i].targetnet.load_state_dict(agents[i].qnet.state_dict())
        print(f"# of episode :{n_epi}, avg reward : {episode_utility:.1f}, buffer size : {agent.replaybuffer.size()}, epsilon : {epsilon*100:.1f}%")

# Export settings
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save rewards to DataFrame and CSV
reward_df = pd.DataFrame(reward_data)
# Pivot the dataframe to have columns for each agent's reward at each step of each episode
df_pivot = reward_df.pivot_table(index=['episode', 'step'], columns='agent_id', values='reward').reset_index()
# Rename the columns appropriately
df_pivot.columns = ['episode', 'step'] + [f'agent_{col}' for col in df_pivot.columns[2:]]
# Save the pivoted dataframe to a new CSV file
df_pivot.to_csv(f'{timestamp}_agent_rewards.csv', index=False)

# Export the episode memory to a CSV file
# rep_path = 'episode_memory'
# if not os.path.exists(rep_path):
#     os.makedirs(rep_path)
# for i in range(node_n):
#     agents[i].episode_memory.export(rep_path, suffix=f'agent_{i}')