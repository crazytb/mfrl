import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import matplotlib.pyplot as plt
import os
from datetime import datetime

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


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
    def __init__(self, agent):
        self.id = agent.id
        self.all_num = agent.topology.n
        self.adj_num = agent.get_adjacent_num()
        self.adj_ids = agent.get_adjacent_ids()
        self.adj_obs = {adj_id: [0, 0] for adj_id in self.adj_ids}
        self.counter = 0
        self.age = 0
        self.max_aoi = 0
        self.all_actions = np.zeros(self.all_num)
        self.topology = agent.topology

        self.observation_space = spaces.Box(low=0, high=1, shape=(1, 4))
        self.action_space = spaces.Discrete(2)
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.counter = 0
        self.age = 0
        self.max_aoi = 0
        observation = np.array([[0.5, 0.5, self.age, self.counter/MAX_STEPS]])
        info = {}
        self.all_actions = np.zeros(self.all_num)
        return observation, info
    
    def set_all_actions(self, actions):
        self.all_actions = np.array(actions)
        
    def get_maxaoi(self):
        return self.max_aoi
    
    def set_max_aoi(self, max_aoi):
        self.max_aoi_set = max_aoi
        
    def calculate_meanfield(self):  
        if self.idle_check():
            return np.array([[1.0, 0.0]])
        else:
            return np.array([self.adj_num*(2**self.adj_num)*np.array([0.5, 0.5]) - self.adj_num*np.array([1.0, 0.0])])/(self.adj_num*(2**self.adj_num)-self.adj_num)

    def idle_check(self):
        if all(self.all_actions[self.adj_ids] == 0):
            return True
        else:
            return False
        
    def get_adjacent_nodes(self, *args):
        if len(args) > 0:
            return np.where(self.topology.adjacency_matrix[args[0]] == 1)[0]
        else:
            return np.where(self.topology.adjacency_matrix[self.id] == 1)[0]
    
    def step(self, action):
        self.counter += 1
        self.age += 1/MAX_STEPS
        observation = self.calculate_meanfield()
        observation = np.append(observation, [self.age, self.counter/MAX_STEPS])
        observation = np.array([observation])
        if action == 1:
            adjacent_nodes = self.get_adjacent_nodes()
            for j in adjacent_nodes:
                js_adjacent_nodes = self.get_adjacent_nodes(j)
                js_adjacent_nodes_except_ind = js_adjacent_nodes[js_adjacent_nodes != self.id]
                if (np.all(self.all_actions[js_adjacent_nodes_except_ind] == 0)
                    and self.all_actions[j] == 0):
                    self.age = 0
                    break
                else:
                    pass
            reward = -1*ENERGY_COEFF
        else:
            reward = 0
        # Save maximum AoI value during the episode
        self.max_aoi = max(self.age, self.max_aoi)
        terminated = False
        info = {}
        if self.counter == MAX_STEPS:
            terminated = True
            reward += (1-self.max_aoi)*MAX_STEPS
            # reward -= MAX_STEPS*np.max(self.max_aoi_set)
        return observation, reward, terminated, False, info
    
def save_model(model, path='default.pth'):
        torch.save(model.state_dict(), path)


class MFRLFullEnv(gym.Env):
    def __init__(self, agent):
        self.n = agent.topology.n
        self.topology = agent.topology
        self.observation_space = spaces.Box(low=0, high=1, shape=(1, self.n+2))
        self.action_space = spaces.Discrete(2**self.n)
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.counter = 0
        self.age = np.zeros(self.n)
        self.max_aoi = np.zeros(self.n)
        observation = np.concatenate((np.array([0.5]*self.n), self.age, np.array([self.counter/MAX_STEPS])))
        observation = np.array([observation])
        info = {}
        self.all_actions = np.zeros(self.n)
        return observation, info
    
    def set_all_actions(self, actions):
        self.all_actions = np.array(actions)
        
    def get_maxaoi(self):
        return self.max_aoi
    
    def set_max_aoi(self, max_aoi):
        self.max_aoi_set = max_aoi
        
    def idle_check(self):
        if all(self.all_actions[self.adj_ids] == 0):
            return True
        else:
            return False
        
    def get_adjacent_nodes(self, *args):
        if len(args) > 0:
            return np.where(self.topology.adjacency_matrix[args[0]] == 1)[0]
        else:
            return np.where(self.topology.adjacency_matrix[self.id] == 1)[0]
        
    def step(self, action):
        self.counter += 1
        self.age += 1/MAX_STEPS
        action_arr = decimal_to_binary_array(action, self.n)
        reward = 0
        for ind, action in enumerate(action_arr):
            if action == 1:
                adjacent_nodes = self.get_adjacent_nodes(ind)
                for j in adjacent_nodes:
                    js_adjacent_nodes = self.get_adjacent_nodes(j)
                    js_adjacent_nodes_except_ind = js_adjacent_nodes[js_adjacent_nodes != ind]
                    if (np.all(action_arr[js_adjacent_nodes_except_ind] == 0)
                        and action_arr[j] == 0):
                        self.age[ind] = 0
                        break
                    else:
                        pass
                reward += -1*ENERGY_COEFF
            else:
                pass
        # Save maximum AoI value during the episode
        self.max_aoi = np.maximum(self.age, self.max_aoi)
        terminated = False
        info = {}
        if self.counter == MAX_STEPS:
            terminated = True
            reward += sum((1-self.max_aoi)*MAX_STEPS)
            
        observation = np.concatenate((action_arr, self.age, np.array([self.counter/MAX_STEPS])))
        observation = np.array([observation])
        reward = reward/self.n
        return observation, reward, terminated, False, info
    
def decimal_to_binary_array(decimal, n):
    # Convert decimal to binary string, remove '0b' prefix
    binary_str = bin(decimal[0])[2:]
    # Pad with zeros if necessary
    binary_str = binary_str.zfill(n)
    # Convert to numpy array of integers
    binary_array = np.array([int(x) for x in binary_str])
    # Ensure the array has length n by truncating if necessary
    return binary_array[-n:]

def save_model(model, path='default.pth'):
        torch.save(model.state_dict(), path)


# Hyperparameters
MAX_STEPS = 300
MAX_EPISODES = 200
GAMMA = 0.98
LEARNING_RATE = 0.0001
N_OBSERVATIONS = 4
N_ACTIONS = 2
print_interval = 10
ENERGY_COEFF = 0.1
ENTROPY_COEFF = 0.01
CRITIC_COEFF = 0.5
MAX_GRAD_NORM = 0.5

    
# Make topology
node_n = 10
method = "linear"
topology = Topology(n=node_n, model=method, density=1)
topo_string = f"{method}_{node_n}"

# Make timestamp
def get_fixed_timestamp():
    timestamp_file = 'fixed_timestamp.txt'
    if os.path.exists(timestamp_file):
        with open(timestamp_file, 'r') as f:
            return f.read().strip()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(timestamp_file, 'w') as f:
            f.write(timestamp)
        return timestamp

FIXED_TIMESTAMP = get_fixed_timestamp()

# DataFrame to store rewards
reward_data = []