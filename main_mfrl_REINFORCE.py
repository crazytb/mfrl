# https://github.com/keep9oing/DRQN-Pytorch-CartPole-v1
# https://ropiens.tistory.com/80
# https://github.com/chingyaoc/pytorch-REINFORCE/tree/master

import os
import numpy as np
import pandas as pd
# from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
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
# print(f"Device: {device}")

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

        self.observation_space = spaces.Box(low=0, high=1, shape=(1, 3))
        self.action_space = spaces.Discrete(2)
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.counter = 0
        self.age = 0
        observation = np.array([[0.5, 0.5, self.age]])
        # observation = [[0.5, 0.5]]
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
            return np.where(topology.adjacency_matrix[args[0]] == 1)[0]
        else:
            return np.where(topology.adjacency_matrix[self.id] == 1)[0]
    
    def step(self, action):
        self.counter += 1
        self.age += 1/MAX_STEPS
        observation = self.calculate_meanfield()
        observation = np.append(observation, self.counter/MAX_STEPS)
        observation = np.array([observation])
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
        
        terminated = False
        info = {}
        if self.counter == MAX_STEPS:
            terminated = True
        return observation, reward, terminated, False, info

class Pinet(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(Pinet, self).__init__()
        self.fc1 = nn.Linear(n_observations, N_HIDDEN)
        self.fc2 = nn.LSTM(N_HIDDEN, N_HIDDEN, batch_first=True)
        self.fc3 = nn.Linear(N_HIDDEN, n_actions)

    def forward(self, x, h, c):
        x = F.relu(self.fc1(x))
        h = h.view(h.size(0), -1)
        c = c.view(c.size(0), -1)
        x, (h_new, c_new) = self.fc2(x, (h, c))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x, h_new, c_new
    
class Agent:
    def __init__(self, topology, id):
        self.topology = topology
        if id >= topology.n:
            raise ValueError("id must be less than n.")
        else:
            self.id = id
        self.env = MFRLEnv(self)
        self.data = []
        self.pinet = Pinet(N_OBSERVATIONS, N_ACTIONS).to(device)
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

# Summarywriter setting
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = 'outputs'
if not os.path.exists(output_path):
    os.makedirs(output_path)
writer = SummaryWriter(output_path + f"/{timestamp}")

# Make topology
topology = Topology(8, "dumbbell")
# topology = Topology(10, "random", 1)
topology.show_adjacency_matrix()
node_n = topology.n
N_OBSERVATIONS = 3
N_HIDDEN = 8
N_ACTIONS = 2

# Hyperparameters
MAX_STEPS = 300
MAX_EPISODES = 50
print_interval = 10
GAMMA = 0.98

# Make agents
agents = [Agent(topology, i) for i in range(node_n)]

# DataFrame to store rewards
reward_data = []

# for n_epi in tqdm(range(MAX_EPISODES), desc="Episodes", position=0, leave=True):
for n_epi in range(MAX_EPISODES):
    episode_utility = 0.0  # Initialize episode score
    observation = [agent.env.reset()[0] for agent in agents]
    done = False
    next_observation = [0]*node_n
    reward = [0]*node_n
    action = [0]*node_n
    probs = []
    h = [torch.zeros(1, 1, N_HIDDEN).to(device) for _ in range(node_n)]
    c = [torch.zeros(1, 1, N_HIDDEN).to(device) for _ in range(node_n)]
    
    # for t in tqdm(range(MAX_STEPS), desc="   Steps", position=1, leave=False):
    for t in range(MAX_STEPS):
        for agent_id, agent in enumerate(agents):
            prob, h[agent_id], c[agent_id] = agent.pinet(torch.from_numpy(observation[agent_id]).float().to(device), 
                                                         h[agent_id], 
                                                         c[agent_id])
            probs.append(prob)
            m = torch.distributions.Categorical(prob)
            action[agent_id] = m.sample().item()

        for agent in agents:
            agent.env.gather_actions(action)
        for agent_id, agent in enumerate(agents):
            next_observation[agent_id], reward[agent_id], done, _, _ = agent.env.step(action[agent_id])
            done_mask = 0.0 if done else 1.0

            observation[agent_id] = next_observation[agent_id]
            episode_utility += reward[agent_id]
            
            # Append reward data
            reward_data.append({'episode': n_epi, 'step': t, 'agent_id': agent_id, 'reward': reward[agent_id]})
        
    agent.train()
    writer.add_scalar('Rewards per episodes', episode_utility, n_epi)
    # if n_epi % print_interval == 0 and n_epi != 0:
        # for i in range(node_n):
        #     agents[i].targetnet.load_state_dict(agents[i].pinet.state_dict())
    print(f"# of episode :{n_epi}/{MAX_EPISODES}, reward : {episode_utility}")

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