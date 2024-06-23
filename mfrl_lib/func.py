import torch
import numpy as np
from collections import deque


def init_grid(size=(10,)):
    grid = torch.randn(*size)
    grid[grid > 0] = 1
    grid[grid <= 0] = 0
    grid = grid.byte()
    return grid

def gen_params(N, size):
    ret = []
    for i in range(N):
        vec = torch.randn(size)/10.
        vec.requires_grad = True
        ret.append(vec)
    return ret

def qfunc(s, theta, layers=[(4, 20), (20, 2)], afn=torch.tanh):
    l1n = layers[0]
    l1s = np.prod(l1n)
    theta_1 = theta[0:l1s].reshape(l1n)
    l2n = layers[1]
    l2s = np.prod(l2n)
    theta_2 = theta[l1s:l1s+l2s].reshape(l2n)
    bias = torch.ones((1, theta_1.shape[1]))
    l1 = torch.nn.functional.elu(s@theta_1 + bias)
    l2 = afn(l1@theta_2)
    return l2.flatten()

def softmax_policy(qvals, temp=0.9):
    soft = torch.exp(qvals / temp)
    soft /= torch.sum(soft)
    action = torch.multinomial(soft, 1)
    return action

def get_coords(grid, j):
    x = int(np.floor(j / grid.shape[0]))
    y = int(j - x * grid.shape[0])
    return x, y

def get_reward_2d(action, action_mean):
    r = (action*(action_mean - action/2)).sum()/action.sum()
    return torch.tanh(5*r)

def get_substate(b):
    s = torch.zeros(2)
    if b > 0:
        s[1] = 1
    else:
        s[0] = 1
    return s

def mean_action(grid, j):
    x, y = get_coords(grid, j)
    action_mean = torch.zeros(2)
    for i in [-1, 0, 1]:
        for k in [-1, 0, 1]:
            if i == k == 0:
                continue
            x_, y_ = x + i, y + k
            x_ = x_ if x_ >= 0 else grid.shape[0] - 1
            y_ = y_ if y_ >= 0 else grid.shape[1] - 1
            x_ = x_ if x_ < grid.shape[0] else 0
            y_ = y_ if y_ < grid.shape[1] else 0
            cur_n = grid[x_, y_]
            s = get_substate(cur_n)
            action_mean += s
            action_mean /= action_mean.sum()
    return action_mean