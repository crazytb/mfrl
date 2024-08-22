import main_mfrl_REINFORCE_vanilla as Vanilla
import main_mfrl_REINFORCE_recurrent as Recurrent
import main_mfrl_REINFORCE_RA2C as RA2C
import main_mfrl_REINFORCE_A2C as A2C
from mfrl_lib.lib import *

import torch
from torch.distributions import Categorical
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import glob

def load_model(filepath, device):
    return torch.load(filepath, map_location=device)

def average_models(model_files, device='cpu'):
    # Load the first model to get the structure
    base_model = load_model(model_files[0], device)
    
    # Initialize a dict to store the sum of parameters
    averaged_params = {}
    for name, param in base_model.items():
        averaged_params[name] = torch.zeros_like(param)

    # Sum up the parameters from all models
    for file in model_files:
        model = load_model(file, device)
        for name, param in model.items():
            averaged_params[name] += param

    # Divide by the number of models to get the average
    for name in averaged_params:
        averaged_params[name] /= len(model_files)

    return averaged_params

def fuse_ra2c_models(model_pattern, output_file, device='cpu'):
    # Get all model files matching the pattern
    model_files = glob.glob(model_pattern)
    
    if not model_files:
        raise ValueError(f"No model files found matching the pattern: {model_pattern}")

    print(f"Found {len(model_files)} model files.")

    # Average the models
    averaged_model = average_models(model_files, device)

    # Save the averaged model
    torch.save(averaged_model, output_file)
    print(f"Fused model saved to {output_file}")

# if GPU is to be used
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

topology.show_adjacency_matrix()

# suffix = "20240819_022826"
suffix = None
def test_model(simmode=None, max_episodes=20, max_steps=300):
    # Create the agents
    if simmode == "RA2C" or simmode == "RA2C_fed":
        agents = [RA2C.Agent(topology, i) for i in range(node_n)]
    elif simmode == "A2C":
        agents = [A2C.Agent(topology, i) for i in range(node_n)]
    elif simmode == "recurrent":
        agents = [Recurrent.Agent(topology, i) for i in range(node_n)]
    elif simmode == "vanilla" or simmode == "fixedprob":
        agents = [Vanilla.Agent(topology, i) for i in range(node_n)]
    else:
        raise ValueError("Invalid simmode.")
    # Load the trained models
    for i in range(topology.n):
        if simmode == "RA2C":
            agents[i].pinet.load_state_dict(torch.load(f"models/RA2C_agent_{i}_20240822_185310.pth", map_location=device))  # 20240819_022826
        elif simmode == "RA2C_fed":
            model_pattern = f"models/RA2C_agent_*_20240822_185310.pth"
            output_file = f"models/RA2C_fed_20240822_185310.pth"
            fuse_ra2c_models(model_pattern, output_file, device)
            agents[i].pinet.load_state_dict(torch.load(output_file, map_location=device))
        elif simmode == "A2C":
            agents[i].pinet.load_state_dict(torch.load(f"models/A2C_agent_{i}_20240822_201033.pth", map_location=device))
        elif simmode == "recurrent":
            agents[i].pinet.load_state_dict(torch.load(f"models/REINFORCE_DRQN_agent_{i}_20240822_185311.pth", map_location=device))
        elif simmode == "vanilla" or simmode == "fixedprob":
            agents[i].pinet.load_state_dict(torch.load(f"models/REINFORCE_vanilla_agent_{i}_20240822_185313.pth", map_location=device))
    
    total_reward = 0
    df = pd.DataFrame()
    states = [torch.from_numpy(agent.env.reset()[0].astype('float32')).unsqueeze(0).to(device) for agent in agents]
    probs = [None]*node_n
    for n_epi in tqdm(range(max_episodes)):
        reward_sum = 0
        aoi_all = [0]*node_n
        h = [torch.zeros(1, 1, 32).to(device) for _ in range(node_n)]
        c = [torch.zeros(1, 1, 32).to(device) for _ in range(node_n)]
        for t in range(max_steps):
            # Calculate the action for each agent
            actions = []
            next_states = []
            for i in range(node_n):
                with torch.no_grad():
                    if simmode == "RA2C" or simmode == "RA2C_fed":
                        probs[i], h[i], c[i], _, _, _ = agents[i].pinet.sample_action(states[i], h[i], c[i])
                    elif simmode == "A2C":
                        probs[i], _, _, _ = agents[i].pinet.sample_action(states[i])
                    elif simmode == "recurrent":
                        probs[i], h[i], c[i], _, _ = agents[i].pinet.sample_action(states[i], h[i], c[i])
                    elif simmode == "vanilla":
                        probs[i] = agents[i].pinet.sample_action(states[i])
                    elif simmode == "fixedprob":
                        num_adjacent = sum(topology.adjacency_matrix[i])
                        txprob = 1/(num_adjacent+1)
                        probs[i] = torch.tensor([[[1-txprob, txprob]]]).to(device)
                    action = Categorical(probs[i]).sample().item()
                    actions.append(action)
            for i in range(node_n):
                agents[i].env.set_all_actions(actions)
                next_state, reward_inst, _, _, _ = agents[i].env.step(actions[i])
                next_states.append(torch.from_numpy(next_state.astype('float32')).unsqueeze(0).to(device))
                if reward_inst > 0:
                    aoi_all[i] = 0
                else:
                    aoi_all[i] += 1/MAX_STEPS
                reward_sum += reward_inst
            states = next_states
            df_index = pd.DataFrame(data=[[n_epi, t]], columns=['episode', 'epoch'])
            df_aoi = pd.DataFrame(data=[aoi_all], columns=[f'aoi_{node}' for node in range(node_n)])
            df_action = pd.DataFrame(data=[actions], columns=[f'action_{node}' for node in range(node_n)])
            df_reward = pd.DataFrame(data=[[reward_sum]], columns=['reward'])
            df1 = pd.concat([df_index, df_aoi, df_action, df_reward], axis=1)
            df = pd.concat([df, df1])
        total_reward += reward_sum
    average_reward = total_reward/max_episodes
    print(f"Average reward for {simmode}: {average_reward:.4f}")
    return df, average_reward

for mode in ["RA2C", "A2C", "recurrent", "vanilla", "fixedprob"]:
    df, avg_reward = test_model(simmode=mode, max_episodes=20, max_steps=300)
    filename = "test_log_" + mode + "_" + "final" + ".csv"
    df.to_csv(filename)
    print(f"Results for {mode} saved to {filename}")
    print(f"Final average reward for {mode}: {avg_reward:.4f}\n")