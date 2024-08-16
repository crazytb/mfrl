import main_mfrl_REINFORCE_vanilla as Vanilla
import main_mfrl_REINFORCE_recurrent as Recurrent
import main_mfrl_REINFORCE_RA2C as RA2C
from mfrl_lib.lib import *

import torch
from torch.distributions import Categorical
import pandas as pd
from tqdm import tqdm

# if GPU is to be used
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

topology.show_adjacency_matrix()

agents = [RA2C.Agent(topology, i) for i in range(node_n)]

# Load the trained models
for i in range(topology.n):
    agents[i].pinet.load_state_dict(torch.load(f"models/RA2C_agent_{i}.pth", map_location=device))

simmode = "RA2C"
states = [torch.from_numpy(agent.env.reset()[0].astype('float32')).unsqueeze(0).to(device) for agent in agents]
probs = [None]*node_n

def test_model(agents, states, probs, simmode):
    df = pd.DataFrame()
    for n_epi in tqdm(range(20)):
        reward_sum = 0
        h = [torch.zeros(1, 1, 32).to(device) for _ in range(node_n)]
        c = [torch.zeros(1, 1, 32).to(device) for _ in range(node_n)]
        for t in range(MAX_STEPS):
            # Calculate the action for each agent
            actions = []
            next_states = []
            for i in range(node_n):
                with torch.no_grad():
                    probs[i], _, _, _, h[i], c[i] = agents[i].pinet.sample_action(states[i], h[i], c[i])
                    action = Categorical(probs[i]).sample().item()
                    actions.append(action)
            for i in range(node_n):
                next_state, reward_inst, _, _, _ = agents[i].env.step(actions[i])
                next_states.append(torch.from_numpy(next_state.astype('float32')).unsqueeze(0).to(device))
                reward_sum += reward_inst
            states = next_states
            df_index = pd.DataFrame(data=[[n_epi, t]], columns=['episode', 'epoch'])
            df_data = pd.DataFrame(data=[[probs[i][0, 0, -1].item() for i in range(node_n)]], columns=[f'agent_{i}' for i in range(node_n)])
            df_reward = pd.DataFrame(data=[[reward_sum]], columns=['reward'])
            df1 = pd.concat([df_index, df_data, df_reward], axis=1)
            df = pd.concat([df, df1])
    filename = "test_log_" + simmode + ".csv"
    df.to_csv(filename)
    return df