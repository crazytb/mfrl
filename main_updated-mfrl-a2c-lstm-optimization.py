import optuna
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from tqdm import tqdm
from mfrl_lib.lib import Topology, MFRLEnv

# Constants
MAX_EPISODES = 10
MAX_STEPS = 300
GAMMA = 0.98
N_OBSERVATIONS = 4
N_ACTIONS = 2
NODE_N = 8
METHOD = "dumbbell"
ENERGY_COEFF = 0.1
MAX_GRAD_NORM = 0.5

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

class OptimizedPinet(nn.Module):
    def __init__(self, n_observations, n_actions, hidden_size, num_lstm_layers, dropout):
        super(OptimizedPinet, self).__init__()
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.fc1 = nn.Linear(n_observations, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_lstm_layers, batch_first=True, dropout=dropout)
        self.actor = nn.Linear(hidden_size, n_actions)
        self.critic = nn.Linear(hidden_size, 1)
        
        self.init_weights()
    
    def init_weights(self):
        for layer in [self.fc1, self.actor, self.critic]:
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(layer.bias, 0)

        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                n = param.size(0)
                start, end = n//4, n//2
                param.data[start:end].fill_(1.)

    def pi(self, x, hidden):
        x = F.relu(self.fc1(x))
        x, lstm_hidden = self.lstm(x, hidden)
        x = self.actor(x)
        prob = F.softmax(x, dim=2)
        return prob, lstm_hidden
    
    def v(self, x, hidden):
        x = F.relu(self.fc1(x))
        x, lstm_hidden = self.lstm(x, hidden)
        v = self.critic(x)
        return v

    def sample_action(self, obs, hidden):
        prob, hidden_new = self.pi(obs, hidden)
        m = Categorical(prob)
        action = m.sample()
        return prob, hidden_new, m.log_prob(action), m.entropy(), self.v(obs, hidden)

class Agent:
    def __init__(self, topology, id, pinet_class, learning_rate, hidden_size, num_lstm_layers, dropout):
        self.gamma = GAMMA
        self.topology = topology
        if id >= topology.n:
            raise ValueError("id must be less than n.")
        self.id = id
        self.env = MFRLEnv(self)
        self.pinet = pinet_class(N_OBSERVATIONS, N_ACTIONS, hidden_size, num_lstm_layers, dropout).to(device)
        self.optimizer = optim.Adam(self.pinet.parameters(), lr=learning_rate)
        self.data = []
        
    def get_adjacent_ids(self):
        return np.where(self.topology.adjacency_matrix[self.id] == 1)[0]
    
    def get_adjacent_num(self):
        return len(self.get_adjacent_ids())
    
    def put_data(self, item):
        self.data.append(item)

    def train(self):
        s, a, r, s_prime, done_mask = self.make_batch()
        hidden = self.init_hidden(1)
        
        v_prime = self.pinet.v(s_prime, hidden).squeeze(1).detach()
        td_target = r + self.gamma * v_prime * done_mask
        v_s = self.pinet.v(s, hidden).squeeze(1)
        delta = td_target - v_s
        advantage = delta.detach()

        pi, _ = self.pinet.pi(s, hidden)
        pi_a = pi.squeeze(0).gather(1, a)
        loss = -torch.log(pi_a) * advantage.unsqueeze(1) + F.smooth_l1_loss(v_s, td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(self.pinet.parameters(), MAX_GRAD_NORM)
        self.optimizer.step()
        self.data = []

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s = torch.tensor(s_lst, dtype=torch.float).to(device)
        a = torch.tensor(a_lst).to(device)
        r = torch.tensor(r_lst).to(device)
        s_prime = torch.tensor(s_prime_lst, dtype=torch.float).to(device)
        done_mask = torch.tensor(done_lst, dtype=torch.float).to(device)
        
        s = s.view([1, len(self.data), -1])
        s_prime = s_prime.view([1, len(self.data), -1])
        return s, a, r, s_prime, done_mask

    def init_hidden(self, batch_size):
        return (torch.zeros(self.pinet.num_lstm_layers, batch_size, self.pinet.hidden_size).to(device),
                torch.zeros(self.pinet.num_lstm_layers, batch_size, self.pinet.hidden_size).to(device))

def create_agent(trial, topology, id):
    # Hyperparameters to optimize
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    hidden_size = trial.suggest_int('hidden_size', 16, 128)
    num_lstm_layers = trial.suggest_int('num_lstm_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    
    return Agent(topology, id, OptimizedPinet, learning_rate, hidden_size, num_lstm_layers, dropout)

def objective(trial):
    # Create topology
    topology = Topology(NODE_N, METHOD)
    
    # Create agents
    agents = [create_agent(trial, topology, i) for i in range(NODE_N)]
    
    # Training loop
    total_utility = 0
    for n_epi in tqdm(range(MAX_EPISODES), desc="Episodes", leave=False):
        episode_utility = 0.0
        observation = [agent.env.reset()[0] for agent in agents]
        hidden = [agent.init_hidden(batch_size=1) for agent in agents]
        done = [False] * NODE_N
        
        for t in range(MAX_STEPS):
            actions = []
            max_aoi = []
            for agent_id, agent in enumerate(agents):
                prob, hidden[agent_id], log_prob, entropy, value = agent.pinet.sample_action(
                    torch.from_numpy(observation[agent_id].astype('float32')).unsqueeze(0).to(device), 
                    hidden[agent_id]
                )
                m = Categorical(prob)
                a = m.sample().item()
                actions.append(a)
            
            for agent in agents:
                agent.env.set_all_actions(actions)
                max_aoi.append(agent.env.get_maxaoi())
            
            for agent_id, agent in enumerate(agents):
                agent.env.set_max_aoi(max_aoi)
                next_observation, reward, done[agent_id], _, _ = agent.env.step(actions[agent_id])
                observation[agent_id] = next_observation
                episode_utility += reward
                agent.put_data((observation[agent_id], actions[agent_id], reward, next_observation, done[agent_id]))
                
            if all(done):
                break
                
        for agent in agents:
            agent.train()
        episode_utility /= NODE_N
        total_utility += episode_utility

        # Report intermediate objective value
        trial.report(total_utility / (n_epi + 1), n_epi)

        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.TrialPruned()

    return total_utility / MAX_EPISODES

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=5, show_progress_bar=True)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Save the best hyperparameters
    best_params = study.best_params
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'best_hyperparameters_{timestamp}.txt', 'w') as f:
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")

    # Optionally, you can plot the optimization history
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(f"optimization_history_{timestamp}.png")
    except:
        print("Could not generate optimization history plot. Make sure you have plotly installed.")