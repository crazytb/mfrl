import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
# from mfrl_lib.environment import *
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import matplotlib.pyplot as plt
import random
import collections 

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
        """Make adjacency matrix of a clique network.
        Args:
            n (int): Number of nodes.
            density (float): Density of the clique network.

        Returns:
            np.ndarray: Adjacency matrix.
        """
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
            # If the density of the current adjacency matrix is over density, return it.
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

        self.observation_space = spaces.Box(low=0, high=1, shape=(1, 2))
        self.action_space = spaces.Discrete(2)
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        observation = np.array([0.5, 0.5])
        info = {}
        # Set MFRLEnv.actions to np.zeros with length of the number of agents
        MFRLEnv.actions = np.zeros(self.all_num)
        return observation, info
    
    def gather_actions(self, action):
        MFRLEnv.actions = np.append(MFRLEnv.actions, action)
        
    def calculate_meanfield(self):  
        # Return the meanfield observation
        if self.idle_check():
            return np.array([0, 1])
        else:
            return np.array(
                (self.adj_num*(2**self.adj_num)*np.array([0.5, 0.5]) - self.adj_num*np.array([1, 0])) 
                / (self.adj_num*(2**self.adj_num)-self.adj_num))

    def idle_check(self):
        # Check if all the adjacent agents are idle, based on MFRLEnv.actions
        if all(MFRLEnv.actions[self.adj_ids] == 0):
            return True
        else:
            return False
    
    def step(self, action):
        observation = self.calculate_meanfield()
        if action == 1 and self.idle_check():
            reward = 1
        else:
            reward = 0
        terminated = False
        self.counter += 1
        info = {}
        if self.counter == MAX_COUNTER:
            terminated = True
        return observation, reward, terminated, False, info

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=BUFFER_LIMIT)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])
        return torch.tensor(np.stack(s_lst), dtype=torch.float).cuda(), torch.tensor(a_lst).cuda(), torch.tensor(r_lst).cuda(), torch.tensor(np.stack(s_prime_lst), dtype=torch.float).cuda(), torch.tensor(done_mask_lst).cuda()
    
    def size(self):
        return len(self.buffer)
    
class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else: 
            return out.argmax().item()


class Agent:
    def __init__(self, topology, id):
        self.topology = topology
        if id >= topology.n:
            raise ValueError("id must be less than n.")
        else:
            self.id = id
        self.env = MFRLEnv(self)
        self.replaybuffer = ReplayBuffer()
        self.qnet = Qnet().to(device)
         
    def get_adjacent_ids(self):
        return np.where(self.topology.adjacency_matrix[self.id] == 1)[0]
    
    def get_adjacent_num(self):
        return len(self.get_adjacent_ids())
    
    def train(self, q_target, optimizer):
        for _ in range(10):
            s, a, r, s_prime, done_mask = self.replaybuffer.sample(BATCH_SIZE)

            q_out = self.qnet(s)
            q_a = q_out.gather(1, a)
            max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
            target = r + GAMMA * max_q_prime * done_mask
            loss = F.smooth_l1_loss(q_a, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    

# Make topology
topology = Topology(12, "dumbbell", 0.5)
topology.show_adjacency_matrix()

# Hyperparameters
MAX_COUNTER = 300
BUFFER_LIMIT  = 50000
BATCH_SIZE    = 32
GAMMA = 0.98

# Make agents
agents = [Agent(topology, i) for i in range(topology.n)]

MAX_EPISODES = 1000
epsilon = 0.1
print_interval = 20
score = 0.0
q_target = Qnet().to(device)
q_target.load_state_dict(agents[0].qnet.state_dict())
optimizer = optim.Adam(agents[0].qnet.parameters(), lr=0.0005)

episode_rewards = []  # List to store average rewards per episode

for n_epi in range(MAX_EPISODES):
    epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) # Linear annealing from 8% to 1%
    episode_score = 0.0  # Initialize episode score

    for agent in agents:
        observation, _ = agent.env.reset()
        done = False
    for agent in agents:
            action = agent.qnet.sample_action(torch.from_numpy(observation).float().to(device), epsilon)
            agent.env.gather_actions(action)
    for agent in agents:
        while not done:    
            next_observation, reward, done, _, info = agent.env.step(action)
            done_mask = 0.0 if done else 1.0
            agent.replaybuffer.put((observation, action, reward, next_observation, done_mask))
            observation = next_observation
            episode_score += reward

            if agent.replaybuffer.size() > 2000:
                agent.train(q_target, optimizer)

    episode_rewards.append(episode_score / topology.n)  # Calculate average reward for this episode

    if n_epi % print_interval == 0 and n_epi != 0:
        q_target.load_state_dict(agents[0].qnet.state_dict())
        avg_reward = np.mean(episode_rewards[-print_interval:])  # Calculate average reward over print_interval
        print(f"# of episode :{n_epi}, avg reward : {avg_reward:.1f}, buffer size : {agent.replaybuffer.size()}, epsilon : {epsilon*100:.1f}%")
        score = 0.0