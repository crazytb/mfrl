import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

#Hyperparameters
learning_rate = 0.0002
gamma         = 0.98

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

device = torch.device("cpu")

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)
        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim = 0)
        return x
      
    def put_data(self, item):
        self.data.append(item)
        
    def train_net(self):
        R = 0
        cum_loss = []
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + gamma * R
            loss = -torch.log(prob) * R # Optimizer는 최소화를 하므로 -를 붙여준다.
            cum_loss.insert(0, loss)
            # loss.backward()
        loss = torch.stack(cum_loss).sum()
        loss.backward()
        self.optimizer.step()
        self.data = []

    
env = gym.make('CartPole-v1')
pi = Policy().to(device)
score = 0.0
print_interval = 20

for n_epi in range(10000):
    s, _ = env.reset()
    done = False
    
    while not done: # CartPole-v1 forced to terminates at 500 step.
        prob = pi(torch.from_numpy(s).float().to(device))
        m = Categorical(prob)
        a = m.sample()
        s_prime, r, done, _, _ = env.step(a.item())
        pi.put_data((r, prob[a]))
        s = s_prime
        score += r
        
    pi.train_net()
    
    if n_epi % print_interval == 0 and n_epi != 0:
        print(f"# of episode :{n_epi}, avg score : {score/print_interval}")
        score = 0.0

env.close()