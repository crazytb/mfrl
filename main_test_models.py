import main_mfrl_REINFORCE_vanilla as Vanilla
import main_mfrl_REINFORCE_recurrent as Recurrent
import main_mfrl_REINFORCE_RA2C as RA2C
from mfrl_lib.lib import *

import torch
from torch.distributions import Categorical
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


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
    if simmode == "RA2C":
        agents = [RA2C.Agent(topology, i) for i in range(node_n)]
    elif simmode == "recurrent":
        agents = [Recurrent.Agent(topology, i) for i in range(node_n)]
    elif simmode == "vanilla" or simmode == "fixedprob":
        agents = [Vanilla.Agent(topology, i) for i in range(node_n)]
    # Load the trained models
    for i in range(topology.n):
        if simmode == "RA2C":
            # agents[i].pinet.load_state_dict(torch.load(f"models/RA2C_agent_{i}.pth", map_location=device))
            agents[i].pinet.load_state_dict(torch.load(f"models/RA2C_agent_{i}_20240819_022826.pth", map_location=device))
        elif simmode == "recurrent":
            agents[i].pinet.load_state_dict(torch.load(f"models/REINFORCE_DRQN_agent_{i}.pth", map_location=device))
        elif simmode == "vanilla" or simmode == "fixedprob":
            agents[i].pinet.load_state_dict(torch.load(f"models/REINFORCE_vanilla_agent_{i}.pth", map_location=device))
        
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
                    if simmode == "RA2C":
                        probs[i], h[i], c[i], _, _, _ = agents[i].pinet.sample_action(states[i], h[i], c[i])
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
    return df

# Set the simulation mode
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for mode in ["RA2C", "recurrent", "vanilla", "fixedprob"]:
    df = test_model(simmode=mode, max_episodes=10, max_steps=300)
    filename = "test_log_" + mode + "_" + timestamp + ".csv"
    df.to_csv(filename)
    
    
# Set the style for the plot
sns.set_theme(style="whitegrid")
sns.set_palette("husl")

# Function to load and process data
def load_data(filename):
    df = pd.read_csv(filename)
    aoi_columns = [col for col in df.columns if col.startswith('aoi_')]
    df['mean_aoi'] = df[aoi_columns].mean(axis=1)
    df['smoothed_mean_aoi'] = df.groupby('episode')['mean_aoi'].transform(lambda x: x.rolling(window=20, min_periods=1).mean())
    return df

# Load data for each mode
modes = ["RA2C", "recurrent", "vanilla", "fixedprob"]
dataframes = {}

for mode in modes:
    filename = f"test_log_{mode}_{timestamp}.csv"
    dataframes[mode] = load_data(filename)

# Create the plot
plt.figure(figsize=(12, 6))

for mode in modes:
    df = dataframes[mode]
    plt.plot(df['epoch'], df['smoothed_mean_aoi'], label=mode.capitalize())

plt.xlabel('Time Step')
plt.ylabel('Mean AoI')
plt.title('Comparison of Mean AoI Trends')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Add overall mean AoI for each mode
for i, mode in enumerate(modes):
    mean_aoi = dataframes[mode]['mean_aoi'].mean()
    plt.text(0.02, 0.95 - i*0.05, f'{mode.capitalize()} Overall Mean AoI: {mean_aoi:.4f}', 
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

plt.tight_layout()

# Save the figure
plt.savefig(f'aoi_trend_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Figure saved as aoi_trend_comparison_{timestamp}.png")