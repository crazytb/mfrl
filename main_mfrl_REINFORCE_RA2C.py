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
    def __init__(self, agent):
        self.id = agent.id
        self.all_num = agent.topology.n
        self.adj_num = agent.get_adjacent_num()
        self.adj_ids = agent.get_adjacent_ids()
        self.adj_obs = {adj_id: [0, 0] for adj_id in self.adj_ids}
        self.counter = 0
        self.age = 0
        self.all_actions = np.zeros(self.all_num)
        self.topology = agent.topology

        self.observation_space = spaces.Box(low=0, high=1, shape=(1, 3))
        self.action_space = spaces.Discrete(2)
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.counter = 0
        self.age = 0
        observation = np.array([[0.5, 0.5, self.age]])
        info = {}
        self.all_actions = np.zeros(self.all_num)
        return observation, info
    
    def set_all_actions(self, actions):
        self.all_actions = np.array(actions)
        
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
        observation = np.append(observation, self.age)
        observation = np.array([observation])
        if action == 1:
            adjacent_nodes = self.get_adjacent_nodes()
            for j in adjacent_nodes:
                js_adjacent_nodes = self.get_adjacent_nodes(j)
                js_adjacent_nodes_except_ind = js_adjacent_nodes[js_adjacent_nodes != self.id]
                if (np.all(self.all_actions[js_adjacent_nodes_except_ind] == 0)
                    and self.all_actions[j] == 0):
                    reward = 1
                    self.age = 0
                    break
                else:
                    reward = -1
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
        self.hidden_space = 32
        self.fc1 = nn.Linear(n_observations, self.hidden_space)
        self.lstm = nn.LSTM(self.hidden_space, self.hidden_space, batch_first=True)
        self.actor = nn.Linear(self.hidden_space, n_actions)
        self.critic = nn.Linear(self.hidden_space, 1)

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
        return action, m.log_prob(action), m.entropy(), value, h, c

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

# Hyperparameters
MAX_STEPS = 300
MAX_EPISODES = 100
GAMMA = 0.98
LEARNING_RATE = 0.0001
N_OBSERVATIONS = 3
N_ACTIONS = 2
print_interval = 10
ENERGY_COEFF = 1/MAX_STEPS
ENTROPY_COEFF = 0.01
CRITIC_COEFF = 0.5
MAX_GRAD_NORM = 0.5

# Summarywriter setting
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = 'outputs'
if not os.path.exists(output_path):
    os.makedirs(output_path)
writer = SummaryWriter(output_path + "/" + "RA2C" + "_" + timestamp)

# Make topology
topology = Topology(4, "dumbbell")
topology.show_adjacency_matrix()
node_n = topology.n

# Make agents
agents = [Agent(topology, i) for i in range(node_n)]

# DataFrame to store rewards
reward_data = []

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

        for agent_id, agent in enumerate(agents):
            action, log_prob, entropy, value, h[agent_id], c[agent_id] = agent.pinet.sample_action(
                torch.from_numpy(observation[agent_id].astype('float32')).unsqueeze(0).to(device), 
                h[agent_id], 
                c[agent_id]
            )
            actions.append(action.item())
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
    writer.add_scalar('Rewards per episodes per agent', episode_utility, n_epi)

    if n_epi % print_interval == 0:
        print(f"# of episode :{n_epi}, avg reward : {episode_utility:.1f}")

# Save rewards to DataFrame and CSV
reward_df = pd.DataFrame(reward_data)
df_pivot = reward_df.pivot_table(index=['episode', 'step'], columns='agent_id', values='prob of 1').reset_index()
df_pivot.columns = ['episode', 'step'] + [f'agent_{col}' for col in df_pivot.columns[2:]]
df_pivot.to_csv(f'RA2C_{timestamp}.csv', index=False)