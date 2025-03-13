import torch
import matplotlib.pyplot as plt
import os
import numpy as np
# import re
from collections import defaultdict

def plot_weight_distributions(checkpoint_path, output_dir='weight_distributions'):
    """
    Plot weight distributions of convolutional layers from a PyTorch checkpoint
    
    Args:
        checkpoint_path: Path to the PyTorch checkpoint (.tar file)
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    
    # Get the model state dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint  # Assume the checkpoint is just the state dict
    
    # Group weight parameters by layer type
    layer_weights = defaultdict(list)
    
    for name, param in state_dict.items():
        if 'weight' in name:
            # Skip non-tensor parameters
            if not isinstance(param, torch.Tensor):
                continue
                
            # Extract layer type from parameter name
            if '.' in name:
                layer_type = name.split('.')[0]
            else:
                layer_type = 'other'
                
            layer_weights[layer_type].append((name, param.cpu().detach().numpy()))
    
    checkpoint_name = os.path.basename(checkpoint_path).replace('.tar', '')
    
    # Plot overall weight distribution
    all_weights = []
    for layer_list in layer_weights.values():
        for _, weights in layer_list:
            all_weights.append(weights.flatten())
    
    if all_weights:
        all_weights_combined = np.concatenate(all_weights)
        
        plt.figure(figsize=(10, 6))
        plt.hist(all_weights_combined, bins=100, alpha=0.7)
        plt.title(f"All Weights Distribution\nMean: {all_weights_combined.mean():.6f}, Std: {all_weights_combined.std():.6f}")
        plt.grid(True)
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, f"{checkpoint_name}_all_weights.png"))
        plt.close()
    
    # Plot distributions by layer type
    for layer_type, weights_list in layer_weights.items():
        if not weights_list:
            continue
            
        # Plot combined distribution for this layer type
        combined_weights = np.concatenate([w[1].flatten() for w in weights_list])
        
        plt.figure(figsize=(10, 6))
        plt.hist(combined_weights, bins=100, alpha=0.7)
        plt.title(f"{layer_type} Weights\nMean: {combined_weights.mean():.6f}, Std: {combined_weights.std():.6f}")
        plt.grid(True)
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, f"{checkpoint_name}_{layer_type}_combined.png"))
        plt.close()
        
        # Plot individual layer distributions
        if len(weights_list) > 1:
            rows = (len(weights_list) + 2) // 3  # Ceiling division for rows
            fig, axes = plt.subplots(rows, 3, figsize=(15, 5*rows))
            axes = axes.flatten() if rows > 1 else [axes]
            
            for i, (name, weights) in enumerate(weights_list):
                if i < len(axes):
                    axes[i].hist(weights.flatten(), bins=50, alpha=0.7)
                    axes[i].set_title(f"{name}\nMean: {weights.mean():.4f}, Std: {weights.std():.4f}")
                    axes[i].grid(True)
            
            for i in range(len(weights_list), len(axes)):
                fig.delaxes(axes[i])
                
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{checkpoint_name}_{layer_type}_individual.png"))
            plt.close()
    
    print(f"Saved weight distribution plots to {output_dir}")

if __name__ == "__main__":
    # Specify the specific checkpoint file to analyze
    checkpoint_path = 'checkpoints/checkpoint_react_thick_rprelu_omni_binary.tar'
    
    print(f"Processing {checkpoint_path}")
    plot_weight_distributions(checkpoint_path)