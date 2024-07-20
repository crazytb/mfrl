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


class PPONet(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(PPONet, self).__init__()
        self.fc1 = nn.Linear(n_observations, 128)
        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.actor = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, x, h, c):
        x = F.relu(self.fc1(x))
        x, (h_new, c_new) = self.lstm(x, (h, c))
        policy = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return policy, value, h_new, c_new


class PPOAgent:
    def __init__(self, topology, id):
        self.topology = topology
        self.id = id
        self.env = MFRLEnv(self)
        self.ppo_net = PPONet(N_OBSERVATIONS, N_ACTIONS).to(device)
        self.optimizer = optim.Adam(self.ppo_net.parameters(), lr=3e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)

        # Pre-allocate tensors
        self.states_tensor = torch.zeros((MAX_STEPS, 1, N_OBSERVATIONS), dtype=torch.float32, device=device)
        self.actions_tensor = torch.zeros(MAX_STEPS, dtype=torch.long, device=device)
        self.old_log_probs_tensor = torch.zeros(MAX_STEPS, dtype=torch.float32, device=device)
        self.returns_tensor = torch.zeros(MAX_STEPS, dtype=torch.float32, device=device)
        self.advantages_tensor = torch.zeros(MAX_STEPS, dtype=torch.float32, device=device)

    def get_adjacent_ids(self):
        return np.where(self.topology.adjacency_matrix[self.id] == 1)[0]
    
    def get_adjacent_num(self):
        return len(self.get_adjacent_ids())
    
    def get_action(self, state, h, c):
        with torch.no_grad():
            policy, value, h_new, c_new = self.ppo_net(state.unsqueeze(0).to(device), h, c)
        dist = torch.distributions.Categorical(policy.squeeze(0))
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item(), h_new, c_new

    def update(self, states, actions, old_log_probs, returns, advantages):
        states_np = np.array(states)

        # Ensure states_np has the correct shape
        if states_np.ndim == 2:
            states_np = states_np[:, np.newaxis, :]
        elif states_np.ndim == 3:
            pass  # Already in the correct shape
        else:
            raise ValueError(f"Unexpected shape of states: {states_np.shape}")

        # Use pre-allocated tensors and torch.as_tensor()
        self.states_tensor[:len(states)] = torch.as_tensor(states_np, dtype=torch.float32, device=device)
        self.actions_tensor[:len(actions)] = torch.as_tensor(actions, dtype=torch.long, device=device)
        self.old_log_probs_tensor[:len(old_log_probs)] = torch.as_tensor(old_log_probs, dtype=torch.float32, device=device)
        self.returns_tensor[:len(returns)] = torch.as_tensor(returns, dtype=torch.float32, device=device)
        self.advantages_tensor[:len(advantages)] = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        
        actual_length = len(states)
        
        for _ in range(PPO_EPOCHS):
            policy, values, _, _ = self.ppo_net(
                self.states_tensor[:actual_length], 
                torch.zeros(1, actual_length, 128, device=device), 
                torch.zeros(1, actual_length, 128, device=device)
            )
            values = values.squeeze(-1)
            dist = torch.distributions.Categorical(policy.squeeze(1))
            new_log_probs = dist.log_prob(self.actions_tensor[:actual_length])
            
            ratio = torch.exp(new_log_probs - self.old_log_probs_tensor[:actual_length])
            surr1 = ratio * self.advantages_tensor[:actual_length]
            surr2 = torch.clamp(ratio, 1-CLIP_EPSILON, 1+CLIP_EPSILON) * self.advantages_tensor[:actual_length]
            actor_loss = -torch.min(surr1, surr2).mean()
            
            critic_loss = F.mse_loss(values.squeeze(1), self.returns_tensor[:actual_length])
            
            entropy = dist.entropy().mean()
            
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.ppo_net.parameters(), max_norm=0.5)
            self.optimizer.step()

        self.scheduler.step()

        return loss.item(), actor_loss.item(), critic_loss.item(), entropy.item()


def compute_gae(rewards, values, next_value, done, gamma, lam):
    advantages = []
    last_advantage = 0
    last_value = next_value

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * last_value * (1 - done[t]) - values[t]
        advantage = delta + gamma * lam * (1 - done[t]) * last_advantage
        advantages.insert(0, advantage)
        last_advantage = advantage
        last_value = values[t]

    returns = np.array(advantages) + values
    advantages = np.array(advantages)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages, returns


# Summarywriter setting
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = 'outputs'
if not os.path.exists(output_path):
    os.makedirs(output_path)
writer = SummaryWriter(output_path + f"/{timestamp}")

# Make topology
topology = Topology(12, "dumbbell", 0.5)
# topology = Topology(10, "random", 1)
topology.show_adjacency_matrix()
node_n = topology.n
N_OBSERVATIONS = 2
N_ACTIONS = 2

# Hyperparameters
MAX_STEPS = 300
MAX_EPISODES = 50
GAMMA = 0.98
LAM = 0.95
CLIP_EPSILON = 0.2
PPO_EPOCHS = 10
print_interval = 10

# Make agents
agents = [PPOAgent(topology, i) for i in range(node_n)]

# DataFrame to store rewards
reward_data = []

for n_epi in tqdm(range(MAX_EPISODES), desc="Episodes", position=0, leave=True):
    episode_utility = 0.0
    observations = [agent.env.reset()[0] for agent in agents]
    h = [torch.zeros(1, 1, 128).to(device) for _ in range(node_n)]
    c = [torch.zeros(1, 1, 128).to(device) for _ in range(node_n)]
    
    episode_data = [[] for _ in range(node_n)]
    
    for t in tqdm(range(MAX_STEPS), desc="   Steps", position=1, leave=False):
        actions = []
        log_probs = []
        values = []
        
        for agent_id, agent in enumerate(agents):
            action, log_prob, value, h[agent_id], c[agent_id] = agent.get_action(torch.from_numpy(observations[agent_id]).float(), h[agent_id], c[agent_id])
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)

        for agent in agents:
            agent.env.gather_actions(actions)
        
        next_observations = []
        rewards = []
        dones = []
        
        for agent_id, agent in enumerate(agents):
            next_obs, reward, done, _, _ = agent.env.step(actions[agent_id])
            next_observations.append(next_obs)
            rewards.append(reward)
            dones.append(done)
            episode_utility += reward
            
            episode_data[agent_id].append((observations[agent_id], actions[agent_id], rewards[agent_id], log_probs[agent_id], values[agent_id], dones[agent_id]))
            
            # Append reward data
            reward_data.append({'episode': n_epi, 'step': t, 'agent_id': agent_id, 'reward': reward})
        
        observations = next_observations
        
        if all(dones):
            break
    
    # Process episode data and update agents
    for agent_id, agent in enumerate(agents):
        states, actions, rewards, old_log_probs, values, dones = zip(*episode_data[agent_id])
        
        # Normalize rewards
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
        
        next_value = 0  # Assuming episode ends
        advantages, returns = compute_gae(rewards, values, next_value, dones, GAMMA, LAM)
        
        loss, actor_loss, critic_loss, entropy = agent.update(states, actions, old_log_probs, returns, advantages)
        
        # # Log metrics
        # writer.add_scalar(f'Agent_{agent_id}/Total_Loss', loss, n_epi)
        # writer.add_scalar(f'Agent_{agent_id}/Actor_Loss', actor_loss, n_epi)
        # writer.add_scalar(f'Agent_{agent_id}/Critic_Loss', critic_loss, n_epi)
        # writer.add_scalar(f'Agent_{agent_id}/Entropy', entropy, n_epi)
    
    writer.add_scalar('Rewards per episodes', episode_utility, n_epi)
    
    if n_epi % print_interval == 0:
        print(f"# of episode :{n_epi}/{MAX_EPISODES}, reward : {episode_utility}")

# Export settings
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save rewards to DataFrame and CSV
reward_df = pd.DataFrame(reward_data)
df_pivot = reward_df.pivot_table(index=['episode', 'step'], columns='agent_id', values='reward').reset_index()
df_pivot.columns = ['episode', 'step'] + [f'agent_{col}' for col in df_pivot.columns[2:]]
df_pivot.to_csv(f'{timestamp}_agent_rewards.csv', index=False)

# Save graph
topology.save_graph_with_labels(output_path)