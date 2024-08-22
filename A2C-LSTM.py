import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
T_horizon = 20

class A2C(nn.Module):
    def __init__(self):
        super(A2C, self).__init__()
        self.data = []
        
        self.fc1 = nn.Linear(4, 64)
        self.lstm = nn.LSTM(64, 32)
        self.fc_pi = nn.Linear(32, 2)
        self.fc_v = nn.Linear(32, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, hidden):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, 64)
        x, lstm_hidden = self.lstm(x, hidden)
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=2)
        return prob, lstm_hidden
    
    def v(self, x, hidden):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, 64)
        x, lstm_hidden = self.lstm(x, hidden)
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
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
            
        s = torch.tensor(s_lst, dtype=torch.float)
        a = torch.tensor(a_lst)
        r = torch.tensor(r_lst)
        s_prime = torch.tensor(s_prime_lst, dtype=torch.float)
        done_mask = torch.tensor(done_lst, dtype=torch.float)
        
        self.data = []
        return s, a, r, s_prime, done_mask
        
    def train_net(self):
        s, a, r, s_prime, done_mask = self.make_batch()
        h = (torch.zeros([1, 1, 32], dtype=torch.float).to(device), torch.zeros([1, 1, 32], dtype=torch.float).to(device))
        s, s_prime, a, r, done_mask = s.to(device), s_prime.to(device), a.to(device), r.to(device), done_mask.to(device)
        
        # Calculate advantages
        v_prime = self.v(s_prime, h).squeeze(1).detach()
        td_target = r + gamma * v_prime * done_mask
        v_s = self.v(s, h).squeeze(1)
        delta = td_target - v_s
        advantage = delta.detach()

        pi, _ = self.pi(s, h)
        pi_a = pi.squeeze(1).gather(1, a)
        loss = -torch.log(pi_a) * advantage.unsqueeze(1) + F.smooth_l1_loss(v_s, td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

def main():
    env = gym.make('CartPole-v1')
    model = A2C()
    model.to(device)
    score = 0.0
    print_interval = 20
    
    for n_epi in range(1000):
        h_out = (torch.zeros([1, 1, 32], dtype=torch.float).to(device), torch.zeros([1, 1, 32], dtype=torch.float).to(device))
        s, _ = env.reset()
        done = False
        
        while not done:
            for t in range(T_horizon):
                h_in = h_out
                prob, h_out = model.pi(torch.from_numpy(s.astype('float32')).unsqueeze(0).to(device), h_in)
                prob = prob.view(-1)
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, truncated, info = env.step(a)

                model.put_data((s, a, r/100.0, s_prime, done))
                s = s_prime

                score += r
                if done:
                    break
                    
            model.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()