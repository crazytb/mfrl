import os
import pandas as pd
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from mfrl_lib.lib import *

# if torch.cuda.is_available():
#     device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")

# class Topology():
#     def __init__(self, n, model="random", density=1):
#         self.n = n
#         self.model = model
#         self.density = density
#         self.adjacency_matrix = self.make_adjacency_matrix()
        
#     def make_adjacency_matrix(self) -> np.ndarray:
#         if self.density < 0 or self.density > 1:
#             raise ValueError("Density must be between 0 and 1.")

#         n_edges = int(self.n * (self.n - 1) / 2 * self.density)
#         adjacency_matrix = np.zeros((self.n, self.n))

#         if self.model == "dumbbell":
#             adjacency_matrix[0, self.n-1] = 1
#             adjacency_matrix[self.n-1, 0] = 1
#             for i in range(1, self.n//2):
#                 adjacency_matrix[0, i] = 1
#                 adjacency_matrix[i, 0] = 1
#             for i in range(self.n//2+1, self.n):
#                 adjacency_matrix[i-1, self.n-1] = 1
#                 adjacency_matrix[self.n-1, i-1] = 1
#         elif self.model == "linear":
#             for i in range(1, self.n):
#                 adjacency_matrix[i-1, i] = 1
#                 adjacency_matrix[i, i-1] = 1
#         elif self.model == "random":
#             for i in range(1, self.n):
#                 adjacency_matrix[i-1, i] = 1
#                 adjacency_matrix[i, i-1] = 1
#                 n_edges -= 1
#             if n_edges <= 0:
#                 return adjacency_matrix
#             else:
#                 arr = [1]*n_edges + [0]*((self.n-1)*(self.n-2)//2 - n_edges)
#                 np.random.shuffle(arr)
#                 for i in range(0, self.n):
#                     for j in range(i+2, self.n):
#                         adjacency_matrix[i, j] = arr.pop()
#                         adjacency_matrix[j, i] = adjacency_matrix[i, j]
#         else:
#             raise ValueError("Model must be dumbbell, linear, or random.")
#         return adjacency_matrix

#     def show_adjacency_matrix(self):
#         print(self.adjacency_matrix)
        
#     def get_density(self):
#         return np.sum(self.adjacency_matrix) / (self.n * (self.n - 1))
    
#     def save_graph_with_labels(self, path):
#         rows, cols = np.where(self.adjacency_matrix == 1)
#         edges = zip(rows.tolist(), cols.tolist())
#         G = nx.Graph()
#         G.add_edges_from(edges)
#         pos = nx.kamada_kawai_layout(G)
#         nx.draw_networkx(G, pos=pos, with_labels=True)
#         plt.savefig(path + '/adj_graph.png')

# class MFRLEnv(gym.Env):
#     actions = np.array([])
#     def __init__(self, agent):
#         self.id = agent.id
#         self.all_num = agent.topology.n
#         self.adj_num = agent.get_adjacent_num()
#         self.adj_ids = agent.get_adjacent_ids()
#         self.adj_obs = {adj_id: [0, 0] for adj_id in self.adj_ids}
#         self.counter = 0
#         self.age = 0

#         self.observation_space = spaces.Box(low=0, high=1, shape=(1, 2))
#         self.action_space = spaces.Discrete(2)
        
#     def reset(self, seed=None):
#         super().reset(seed=seed)
#         observation = np.array([[0.5, 0.5]])
#         info = {}
#         MFRLEnv.actions = np.zeros(self.all_num)
#         self.counter = 0
#         self.age = 0
#         return observation, info
    
#     def gather_actions(self, action):
#         MFRLEnv.actions = np.array(action)
        
#     def calculate_meanfield(self):  
#         if self.idle_check():
#             return np.array([[1.0, 0.0]])
#         else:
#             return np.array([self.adj_num*(2**self.adj_num)*np.array([0.5, 0.5]) - self.adj_num*np.array([1.0, 0.0])])/(self.adj_num*(2**self.adj_num)-self.adj_num)

#     def idle_check(self):
#         if all(MFRLEnv.actions[self.adj_ids] == 0):
#             return True
#         else:
#             return False
        
#     def get_adjacent_nodes(self, *args):
#         if len(args) > 0:
#             return np.where(topology.adjacency_matrix[args[0]] == 1)[0]
#         else:
#             return np.where(topology.adjacency_matrix[self.id] == 1)[0]
    
#     def step(self, action):
#         observation = self.calculate_meanfield()
#         self.age += 1/MAX_STEPS
#         if action == 1:
#             adjacent_nodes = self.get_adjacent_nodes()
#             for j in adjacent_nodes:
#                 js_adjacent_nodes = self.get_adjacent_nodes(j)
#                 js_adjacent_nodes_except_ind = js_adjacent_nodes[js_adjacent_nodes != self.id]
#                 if (np.all(MFRLEnv.actions[js_adjacent_nodes_except_ind] == 0)
#                     and MFRLEnv.actions[j] == 0):
#                     reward = np.tanh(5*self.age)
#                     self.age = 0
#                     break
#                 else:
#                     reward = 0
#         else:
#             reward = 0
#         self.counter += 1
#         terminated = False
#         info = {}
#         if self.counter == MAX_STEPS:
#             terminated = True
#         return observation, reward, terminated, False, info

# class TransformerEncoder(nn.Module):
#     def __init__(self, d_model, nhead, num_layers):
#         super(TransformerEncoder, self).__init__()
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
#     def forward(self, src):
#         return self.transformer_encoder(src)

# class Pinet(nn.Module):
#     def __init__(self, n_observations, n_actions, d_model=64, nhead=4, num_layers=2):
#         super(Pinet, self).__init__()
#         self.fc1 = nn.Linear(n_observations, d_model)
#         self.transformer = TransformerEncoder(d_model, nhead, num_layers)
#         self.fc2 = nn.Linear(d_model, n_actions)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = x.unsqueeze(1)  # Add sequence dimension
#         x = self.transformer(x)
#         x = x.squeeze(1)  # Remove sequence dimension
#         x = self.fc2(x)
#         x = F.softmax(x, dim=-1)
#         return x

# class Agent:
#     def __init__(self, topology, id):
#         self.topology = topology
#         if id >= topology.n:
#             raise ValueError("id must be less than n.")
#         else:
#             self.id = id
#         self.env = MFRLEnv(self)
#         self.data = []
#         self.pinet = Pinet(N_OBSERVATIONS, N_ACTIONS).to(device)
#         self.optimizer = optim.Adam(self.pinet.parameters(), lr=0.0005)
        
#     def get_adjacent_ids(self):
#         return np.where(self.topology.adjacency_matrix[self.id] == 1)[0]
    
#     def get_adjacent_num(self):
#         return len(self.get_adjacent_ids())
    
#     def put_data(self, item):
#         self.data.append(item)
    
#     def train(self):
#         R = 0
#         self.optimizer.zero_grad()
#         for r, prob in self.data[::-1]:
#             R = r + GAMMA * R
#             loss = -torch.log(prob) * R
#             loss.backward()
#         self.optimizer.step()
#         self.data = []

# Summarywriter setting
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = 'outputs'
if not os.path.exists(output_path):
    os.makedirs(output_path)
writer = SummaryWriter(output_path + f"/{timestamp}")

# Make topology
topology = Topology(10, "dumbbell")
topology.show_adjacency_matrix()
node_n = topology.n
N_OBSERVATIONS = 3
N_ACTIONS = 2

# Hyperparameters
MAX_STEPS = 300
MAX_EPISODES = 50
print_interval = 10
GAMMA = 0.98

# Make agents
agents = [Agent(topology, i, N_OBSERVATIONS, N_ACTIONS, GAMMA, MAX_STEPS) for i in range(node_n)]

# DataFrame to store rewards
reward_data = []

for n_epi in range(MAX_EPISODES):
    episode_utility = 0.0
    observation = [agent.env.reset()[0] for agent in agents]
    done = False
    next_observation = [0]*node_n
    reward = [0]*node_n
    action = [0]*node_n
    probs = np.zeros((MAX_STEPS, node_n, N_ACTIONS))
    
    for t in range(MAX_STEPS):
        for agent_id, agent in enumerate(agents):
            prob = agent.pinet(torch.from_numpy(observation[agent_id]).float().to(device))
            probs[t, agent_id] = prob.cpu().detach().numpy()
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

    # Utility per episode
    writer.add_scalar('Rewards per episodes', episode_utility, n_epi)
    print(f"# of episode :{n_epi}/{MAX_EPISODES}, reward : {episode_utility}")
    # Tx probabilities per episode
    np.set_printoptions(precision=3)
    print(f"Tx probabilities: {probs[-1, :, -1]}")

# Close the writer
writer.close()

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