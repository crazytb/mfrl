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
        observation = np.array([0.5, 0.5])
        info = {}
        MFRLEnv.actions = np.zeros(self.all_num)
        self.counter = 0
        self.age = 0
        return observation, info
    
    def gather_actions(self, action):
        MFRLEnv.actions = np.array(action)
        
    def calculate_meanfield(self):  
        if self.idle_check():
            return np.array([0.0, 1.0])
        else:
            return np.array(self.adj_num*(2**self.adj_num)*np.array([0.5, 0.5]) - self.adj_num*np.array([1, 0]))/(self.adj_num*(2**self.adj_num)-self.adj_num)

    def idle_check(self):
        if all(MFRLEnv.actions[self.adj_ids] == 0):
            return True
        else:
            return False
    
    def step(self, action):
        observation = self.calculate_meanfield()
        self.age += 1/MAX_STEPS
        if action == 1 and self.idle_check():
            reward = self.age
            self.age = 0
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
        return torch.tensor(np.stack(s_lst), dtype=torch.float).to(device), torch.tensor(a_lst).to(device), torch.tensor(r_lst).to(device), torch.tensor(np.stack(s_prime_lst), dtype=torch.float).to(device), torch.tensor(done_mask_lst).to(device)
    
    def size(self):
        return len(self.buffer)
    
    # Export the buffer to csv file into path
    def export_buffer(self, path, suffix=''):
        df = pd.DataFrame(self.buffer, columns=['s', 'a', 'r', 's_prime', 'done_mask'])
        df.to_csv(path + f'/{timestamp}_{suffix}.csv', index=False)
    
class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        # self.fc2 = nn.Linear(128, 128)
        self.fc2 = nn.LSTM(128, 128, batch_first=True)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x, h, c):
        x = F.relu(self.fc1(x))
        h = h.view(h.size(0), -1)
        c = c.view(c.size(0), -1)
        x, (h, c) = self.fc2(x, (h, c))
        x = self.fc3(x)
        return x, (h.unsqueeze(0), c.unsqueeze(0))
      
    def sample_action(self, obs, h, c, epsilon):
        obs = obs.unsqueeze(0)
        out, (h, c) = self.forward(obs, h, c)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1), (h, c)
        else: 
            return out.argmax().item(), (h, c)


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
    
# Summarywriter setting
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = 'outputs'
if not os.path.exists(output_path):
    os.makedirs(output_path)
writer = SummaryWriter(output_path + f"/{timestamp}")

# Make topology
topology = Topology(10, "dumbbell", 0.5)
topology.show_adjacency_matrix()
node_n = topology.n

# Hyperparameters
MAX_STEPS = 300
BUFFER_LIMIT  = 10000
BATCH_SIZE    = 32
GAMMA = 0.98

# Make agents
agents = [Agent(topology, i) for i in range(node_n)]

MAX_EPISODES = 500
epsilon = 0.1
print_interval = 10
score = 0.0
q_target = [Qnet().to(device) for _ in range(node_n)]
for i in range(node_n):
    q_target[i].load_state_dict(agents[i].qnet.state_dict())
optimizer = [optim.Adam(agents[i].qnet.parameters(), lr=0.0005) for i in range(node_n)]

# DataFrame to store rewards
reward_data = []

for n_epi in tqdm(range(MAX_EPISODES), desc="Episodes", position=0, leave=True):
    epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) # Linear annealing from 8% to 1%
    episode_score = 0.0  # Initialize episode score
    observation = [agent.env.reset()[0] for agent in agents]
    done = False
    next_observation = [0]*node_n
    reward = [0]*node_n
    action = [0]*node_n
    h = [torch.zeros(1, 1, 128).to(device) for _ in range(node_n)]
    c = [torch.zeros(1, 1, 128).to(device) for _ in range(node_n)]
    
    for t in tqdm(range(MAX_STEPS), desc="   Steps", position=1, leave=False):
        for agent_id, agent in enumerate(agents):
            action[agent_id], (h[agent_id], c[agent_id]) = agent.qnet.sample_action(torch.from_numpy(observation[agent_id]).float().to(device), 
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
            observation[agent_id] = next_observation[agent_id]
            episode_score += reward[agent_id]
            
            # Append reward data
            reward_data.append({'episode': n_epi, 'step': t, 'agent_id': agent_id, 'reward': reward[agent_id]})

            if agent.replaybuffer.size() > BUFFER_LIMIT:
                agent.train(q_target[agent_id], optimizer[agent_id])

    writer.add_scalar('Rewards per episodes', episode_score, n_epi)

    if n_epi % print_interval == 0 and n_epi != 0:
        for i in range(node_n):
            q_target[i].load_state_dict(agents[i].qnet.state_dict())
        avg_reward = episode_score / (node_n * MAX_STEPS)
        print(f"# of episode :{n_epi}, avg reward : {avg_reward:.1f}, buffer size : {agent.replaybuffer.size()}, epsilon : {epsilon*100:.1f}%")

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

# Export the replay buffer to a CSV file
rep_path = 'replay_buffers'
if not os.path.exists(rep_path):
    os.makedirs(rep_path)
for i in range(node_n):
    agents[i].replaybuffer.export_buffer(path=rep_path, suffix=f'{i}')