import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

GAMMA = 0.99

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
    def __init__(self, agent, topology, max_steps=300):
        self.id = agent.id
        self.all_num = agent.topology.n
        self.adj_num = agent.get_adjacent_num()
        self.adj_ids = agent.get_adjacent_ids()
        self.topology = topology
        self.max_steps = max_steps
        self.counter = 0
        self.age = 0
        # Observation: [no tx prob, tx prob, age]
        self.observation_space = spaces.Box(low=0, high=1, shape=(1, 3))
        self.action_space = spaces.Discrete(2)
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.counter = 0
        self.age = 0
        observation = np.array([[0.5, 0.5, self.age]])
        info = {}
        MFRLEnv.actions = np.zeros(self.all_num)
        return observation, info
    
    def gather_actions(self, action):
        MFRLEnv.actions = np.array(action)
        
    def calculate_meanfield(self):  
        if self.idle_check():
            return np.array([[1.0, 0.0]])
        else:
            return np.array([self.adj_num*(2**self.adj_num)*np.array([0.5, 0.5]) - self.adj_num*np.array([1.0, 0.0])])/(self.adj_num*(2**self.adj_num)-self.adj_num)

    def idle_check(self):
        if all(MFRLEnv.actions[self.adj_ids] == 0):
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
        self.age += 1/self.max_steps
        observation = self.calculate_meanfield()
        observation = np.append(observation, self.counter/self.max_steps)
        observation = np.array([observation])
        if action == 1:
            adjacent_nodes = self.get_adjacent_nodes()
            for j in adjacent_nodes:
                js_adjacent_nodes = self.get_adjacent_nodes(j)
                js_adjacent_nodes_except_ind = js_adjacent_nodes[js_adjacent_nodes != self.id]
                if (np.all(MFRLEnv.actions[js_adjacent_nodes_except_ind] == 0)
                    and MFRLEnv.actions[j] == 0):
                    reward = np.tanh(5*self.age)
                    self.age = 0
                    break
                else:
                    reward = 0
        else:
            reward = 0
        
        terminated = False
        info = {}
        if self.counter == self.max_steps:
            terminated = True
        return observation, reward, terminated, False, info

class Pinet(nn.Module):
    def __init__(self, n_observations, n_actions, n_hidden=32):
        super(Pinet, self).__init__()
        self.fc1 = nn.Linear(n_observations, n_hidden)
        self.fc2 = nn.LSTM(n_hidden, n_hidden, batch_first=True)
        self.fc3 = nn.Linear(n_hidden, n_actions)

    def forward(self, x, h, c):
        x = F.relu(self.fc1(x))
        h = h.view(h.size(0), -1)
        c = c.view(c.size(0), -1)
        x, (h_new, c_new) = self.fc2(x, (h, c))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x, h_new, c_new

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
    def forward(self, src):
        return self.transformer_encoder(src)

class Pinet_transformer(nn.Module):
    def __init__(self, n_observations, n_actions, d_model=64, nhead=4, num_layers=2):
        super(Pinet_transformer, self).__init__()
        self.fc1 = nn.Linear(n_observations, d_model)
        self.transformer = TransformerEncoder(d_model, nhead, num_layers)
        self.fc2 = nn.Linear(d_model, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(1)  # Add sequence dimension
        x = self.transformer(x)
        x = x.squeeze(1)  # Remove sequence dimension
        x = self.fc2(x)
        x = F.softmax(x, dim=-1)
        return x

class Agent:
    def __init__(self, topology, id, n_observations, n_actions, gamma, max_steps, model="Pinet"):
        self.topology = topology
        if id >= topology.n:
            raise ValueError("id must be less than n.")
        else:
            self.id = id
        self.env = MFRLEnv(self, topology, max_steps)
        self.data = []
        # self.pinet = Pinet(n_observations, n_actions).to(device)
        self.pinet = Pinet_transformer(n_observations, n_actions).to(device)
        self.optimizer = optim.Adam(self.pinet.parameters(), lr=0.0005)
        
    def get_adjacent_ids(self):
        return np.where(self.topology.adjacency_matrix[self.id] == 1)[0]
    
    def get_adjacent_num(self):
        return len(self.get_adjacent_ids())
    
    def put_data(self, item):
        self.data.append(item)
    
    def train(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + GAMMA * R
            loss = -torch.log(prob) * R
            loss.backward()
        self.optimizer.step()
        self.data = []