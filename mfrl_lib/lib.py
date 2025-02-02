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
import random

# Add this function to set global seed
def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set a global seed
GLOBAL_SEED = 42
set_global_seed(GLOBAL_SEED)

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
        elif self.model == "random" or self.model == "fullmesh":
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
        super().__init__()
        self.agent = agent
        self.arrival_rate = agent.arrival_rate
        self.id = agent.id
        self.all_num = agent.topology.n
        # self.adj_num = agent.get_adjacent_num()
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
        
    def set_all_arrival_rates(self, arrival_rate):
        self.all_arrival_rates = arrival_rate
        
    def get_maxaoi(self):
        return self.max_aoi
    
    def set_max_aoi(self, max_aoi):
        self.max_aoi_set = max_aoi
        
    def calculate_meanfield(self):  
        if self.idle_check():
            return np.array([[1.0, 0.0]])
        else:
            adj_num = self.agent.get_adjacent_num()
            return np.array([adj_num*(2**adj_num)*np.array([0.5, 0.5]) - adj_num*np.array([1.0, 0.0])])/(adj_num*(2**adj_num)-adj_num)
            # return np.array([self.adj_num*(2**self.adj_num)*np.array([0.5, 0.5]) - self.adj_num*np.array([1.0, 0.0])])/(self.adj_num*(2**self.adj_num)-self.adj_num)

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
        if (self.all_actions[self.id] == 1):
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
        # Check termination condition
        terminated = self.counter == MAX_STEPS
        info = {}
        
        if terminated:
            reward += (1-self.max_aoi)*MAX_STEPS
        return observation, reward, terminated, False, info
    
def get_env_ages(agents):
    """
    Get all environment ages as a numpy array.
    """
    return np.array([agent.env.age for agent in agents])

def save_model(model, path='default.pth'):
        torch.save(model.state_dict(), path)


class MFRLFullEnv(gym.Env):
    def __init__(self, agent):
        super().__init__()
        self.n = agent.topology.n
        self.topology = agent.topology
        self.arrival_rate = agent.arrival_rate
        self.counter = 0
        self.age = np.zeros(self.n)
        self.max_aoi = np.zeros(self.n)
        self.states = np.zeros((self.n, 3))  # States for each device: [idle, success, collision]
        self.states[:, 0] = 1  # Initialize all devices to idle state
        
        # Define observation space
        self.observation_space = spaces.Dict({
            "devices": spaces.Tuple([
                spaces.Dict({
                    "state": spaces.MultiBinary(3),  # One-hot encoded {idle:0, success:1, collision:2}
                    "age": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
                }) for _ in range(self.n)
            ]),
            "counter": spaces.Discrete(MAX_STEPS)
        })

        # Define action space as MultiBinary for direct binary actions
        self.action_space = spaces.MultiBinary(self.n)
        
        # For storing last action and observation (useful for debugging)
        self.last_action = None
        self.last_observation = None
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset counters and states
        self.counter = 0
        self.age = np.zeros(self.n)
        self.max_aoi = np.zeros(self.n)
        self.states = np.zeros((self.n, 3))
        self.states[:, 0] = 1  # All devices start in idle state
        
        # Create observation
        observation = self._create_observation()
        self.last_observation = observation
        
        info = {}
        return observation, info
    
    def _create_observation(self):
        """Helper method to create observation dictionary."""
        return {
            "devices": tuple({
                "state": self.states[i],
                "age": np.array([self.age[i]], dtype=np.float32)
            } for i in range(self.n)),
            "counter": self.counter
        }
    
    def _validate_action(self, action):
        """Validate the input action."""
        if not isinstance(action, (np.ndarray, list)):
            raise ValueError(f"Action must be numpy array or list, got {type(action)}")
        if len(action) != self.n:
            raise ValueError(f"Action length must be {self.n}, got {len(action)}")
        if not np.all(np.logical_or(action == 0, action == 1)):
            raise ValueError("Actions must be binary (0 or 1)")
        return np.array(action)
        
    def step(self, action):
        """Execute one step in the environment."""
        # Validate and store action
        action = self._validate_action(action)
        self.last_action = action
        
        # Increment counter and age
        self.counter += 1
        self.age += 1 / MAX_STEPS
        
        # Track transmitting devices and calculate energy cost
        transmitting_devices = []
        for ind, act in enumerate(action):
            if act == 1 and self.arrival_rate[ind] > np.random.rand():
                transmitting_devices.append(ind)
        
        # Calculate energy cost
        energy_reward = -1 * len(transmitting_devices) * ENERGY_COEFF
        
        # Update states based on transmission outcomes
        new_states = np.zeros((self.n, 3))
        for ind in range(self.n):
            if ind in transmitting_devices:
                if len(transmitting_devices) == 1:
                    # Successful transmission
                    new_states[ind, 1] = 1  # Success state
                    self.age[ind] = 0  # Reset age
                else:
                    # Collision
                    new_states[ind, 2] = 1  # Collision state
            else:
                # Idle
                new_states[ind, 0] = 1
        
        self.states = new_states
        self.max_aoi = np.maximum(self.age, self.max_aoi)
        
        # Create observation
        observation = self._create_observation()
        self.last_observation = observation
        
        # Check termination
        terminated = self.counter >= MAX_STEPS
        
        # Calculate reward
        if terminated:
            # Add AoI reward at episode end
            aoi_reward = np.sum((1 - self.max_aoi)) * MAX_STEPS
            total_reward = energy_reward + aoi_reward
        else:
            total_reward = energy_reward
            
        return observation, total_reward, terminated, False, {}
    
    def get_maxaoi(self):
        """Return the maximum AoI values."""
        return self.max_aoi.copy()
    
    def render(self):
        """Optional: Implement visualization of the environment state."""
        pass
        
    def close(self):
        """Perform any necessary cleanup."""
        pass
    
def decimal_to_binary_array(decimal, n):
    # Convert decimal to binary string, remove '0b' prefix
    binary_str = bin(decimal)[2:]
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
ENERGY_COEFF = 1
ENTROPY_COEFF = 0.01
CRITIC_COEFF = 0.5
MAX_GRAD_NORM = 0.5

    
# Make topology
node_n = 8
method = "fullmesh"
if method == "fullmesh":
    density = 1
topology = Topology(n=node_n, model=method, density=density)
topo_string = f"{method}_{node_n}"
arrival_rate = np.linspace(0, 1, node_n+2).tolist()[1:-1]

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

