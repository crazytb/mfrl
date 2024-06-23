from random import shuffle
from matplotlib import pyplot as plt
from collections import deque
import numpy as np
import torch
from mfrl_lib.func import *

# Main
# Parameters
num_states = 2
num_actions = 2
size = (10, 10)
J = np.prod(size)
hid_layer = 10
layers = [(num_actions,hid_layer),
          (hid_layer,num_actions)]
params = gen_params(1,num_states*hid_layer+hid_layer*num_actions)
grid = init_grid(size=size)
grid_ = grid.clone()
grid__ = grid.clone()
print(grid.sum())

# Training
epochs = 100
lr = 0.0001
num_iter = 3
replay_size = 50
replay = deque(maxlen=replay_size)
batch_size = 10
gamma = 0.9
losses = [[] for i in range(J)]

for i in range(epochs):
    act_means = torch.zeros((J, 2))
    q_next = torch.zeros(J)
    
    for m in range(num_iter):
        for j in range(J):
            action_mean = mean_action(grid_, j).detach()
            act_means[j] = action_mean.clone()
            qvals = qfunc(action_mean.detach(), params[0], layers=layers)
            action = softmax_policy(qvals.detach(), temp=0.5)
            grid_[get_coords(grid_, j)] = action
            q_next[j] = torch.max(qvals).detach()
    
    grid_.data = grid_.data
    grid.data = grid_.data
    actions = torch.stack([get_substate(a.item()) for a in grid.flatten()])
    rewards = torch.stack([get_reward_2d(actions[j], act_means[j]) for j in range(J)])
    # Collects an experience and adds to the experience replay buffer
    exp = (actions, rewards, act_means, q_next)
    replay.append(exp)
    shuffle(replay)

    # Once the experience replay buffer has more experiences than the batch size parameter, starts training
    if len(replay) > batch_size:
        ids = np.random.randint(low=0, high=len(replay), size=batch_size)
        exps = [replay[idx] for idx in ids]
        for j in range(J):
            jacts = torch.stack([ex[0][j] for ex in exps]).detach()
            jrewards = torch.stack([ex[1][j] for ex in exps]).detach()
            jmeans = torch.stack([ex[2][j] for ex in exps]).detach()
            vs = torch.stack([ex[3][j] for ex in exps]).detach()
            qvals = torch.stack([
                qfunc(jmeans[h].detach(), params[0], layers=layers)
                for h in range(batch_size)
            ])
            target = qvals.clone().detach()
            target[:, torch.argmax(jacts, dim=1)] = jrewards + gamma * vs
            loss = torch.sum(torch.pow(qvals - target.detach(), 2))
            losses[j].append(loss.item())
            loss.backward()
            with torch.no_grad():
                params[0] = params[0] - lr * params[0].grad
            params[0].requires_grad = True
    if 'loss' in locals():
        print(f"Epoch: {i}/{epochs}, Grid sum: {grid.sum()}, Loss: {loss.item()}")
    else:
        print(f"Epoch: {i}/{epochs}, Grid sum: {grid.sum()}")