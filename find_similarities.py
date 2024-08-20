import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
import numpy as np

def weight_similarity(model1, model2):
    """Calculate cosine similarity between flattened model weights."""
    weights1 = torch.cat([p.view(-1) for p in model1.parameters()])
    weights2 = torch.cat([p.view(-1) for p in model2.parameters()])
    return F.cosine_similarity(weights1.unsqueeze(0), weights2.unsqueeze(0)).item()

def output_similarity(model1, model2, data):
    """Calculate Pearson correlation between model outputs."""
    input_data, h, c = data
    with torch.no_grad():
        policy1, value1, _, _ = model1(input_data, h, c)
        policy2, value2, _, _ = model2(input_data, h, c)
        
        policy_sim = pearsonr(policy1.flatten().cpu().numpy(), policy2.flatten().cpu().numpy())[0]
        value_diff = torch.abs(value1 - value2).item()
    
    return policy_sim, value_diff

def activation_similarity(model1, model2, data):
    """Compare activations of each layer."""
    input_data, h, c = data
    similarities = []
    
    def get_activation(name):
        def hook(module, input, output):
            if isinstance(output, tuple):  # LSTM output
                activation[name] = (output[0].detach(), output[1][0].detach(), output[1][1].detach())
            else:  # Linear layer output
                activation[name] = output.detach()
        return hook

    activation = {}
    hooks = []
    
    # Register hooks for both models
    for name, module in model1.named_modules():
        if isinstance(module, (nn.Linear, nn.LSTM)):
            hooks.append(module.register_forward_hook(get_activation(f'model1_{name}')))

    for name, module in model2.named_modules():
        if isinstance(module, (nn.Linear, nn.LSTM)):
            hooks.append(module.register_forward_hook(get_activation(f'model2_{name}')))

    # Forward pass
    with torch.no_grad():
        model1(input_data, h, c)
        model2(input_data, h, c)

    # Compare activations
    for (name1, module1), (name2, module2) in zip(model1.named_modules(), model2.named_modules()):
        if isinstance(module1, (nn.Linear, nn.LSTM)) and isinstance(module2, (nn.Linear, nn.LSTM)):
            act1 = activation[f'model1_{name1}']
            act2 = activation[f'model2_{name2}']
            
            if isinstance(module1, nn.LSTM):
                # For LSTM, compare output, hidden state, and cell state
                out_sim = pearsonr(act1[0].flatten().cpu().numpy(), act2[0].flatten().cpu().numpy())[0]
                h_sim = pearsonr(act1[1].flatten().cpu().numpy(), act2[1].flatten().cpu().numpy())[0]
                c_sim = pearsonr(act1[2].flatten().cpu().numpy(), act2[2].flatten().cpu().numpy())[0]
                similarities.append((f"{name1}_output", out_sim))
                similarities.append((f"{name1}_hidden", h_sim))
                similarities.append((f"{name1}_cell", c_sim))
            else:
                # For Linear layers
                if act1.numel() > 1:  # More than one element (actor output)
                    sim = pearsonr(act1.flatten().cpu().numpy(), act2.flatten().cpu().numpy())[0]
                else:  # Single element (critic output)
                    sim = abs(act1.item() - act2.item())  # Absolute difference for single values
                similarities.append((name1, sim))

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return similarities


def compare_models(model1, model2, input_data):
    """Compare two models using multiple similarity metrics."""
    weight_sim = weight_similarity(model1, model2)
    output_sim = output_similarity(model1, model2, input_data)
    activation_sims = activation_similarity(model1, model2, input_data)
    
    print(f"Weight Similarity: {weight_sim:.4f}")
    print(f"Policy Similarity: {output_sim[0]:.4f}, Value Difference: {output_sim[1]:.4f}")
    print("Activation Similarities:")
    for name, sim in activation_sims:
        print(f"  {name}: {sim:.4f}")


import main_mfrl_REINFORCE_RA2C as RA2C
from mfrl_lib.lib import *
import torch

agents = [RA2C.Agent(topology, i) for i in range(node_n)]
for i in range(node_n):
    agents[i].pinet.load_state_dict(torch.load(f"models/RA2C_agent_{i}_20240819_022826.pth", map_location=device))
    
    
compare_models(agents[0].pinet, agents[1].pinet, ((torch.Tensor([[[0.5, 0.5, 0.0]]])).to(device),
                                                   torch.zeros(1, 1, 32).to(device),
                                                   torch.zeros(1, 1, 32).to(device)))