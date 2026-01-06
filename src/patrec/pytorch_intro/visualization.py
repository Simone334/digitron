"""
This module provides visualization functions for the PyTorch introductory
exercise (Step 8), including plotting loss curves and model predictions.
"""
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def plot_loss_comparison(loss_histories: Dict[str, List[float]], output_path: str = None):
    """
    Plots the training loss curves for multiple models on the same graph.

    Args:
        loss_histories (Dict[str, List[float]]): A dictionary where keys are model
                                                 names (e.g., "LSTM") and values
                                                 are the lists of their training losses.
        output_path (str, optional): If provided, the plot is saved to this path.
                                     Defaults to None.
    """
    plt.figure(figsize=(12, 7))
    model_colors = {"Simple RNN": "skyblue", "LSTM": "salmon", "GRU": "lightgreen"}

    # Plot each model's loss history.
    for name, losses in loss_histories.items():
        plt.plot(losses, label=f'{name} Loss', color=model_colors.get(name, None))

    plt.title('Model Training Loss Comparison', fontsize=16)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error Loss')
    plt.legend()
    plt.grid(True)

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"INFO: Loss comparison plot saved to {output_path}")
    plt.show()


def plot_predictions(
    models: Dict[str, nn.Module],
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    num_samples: int = 6,
    output_path: str = None
):
    """
    Visualizes model predictions against the true target on random test samples.

    Args:
        models (Dict[str, nn.Module]): A dictionary of trained model instances.
        X_test (torch.Tensor): The test set input data.
        y_test (torch.Tensor): The test set target data.
        num_samples (int): The number of random samples to plot.
        output_path (str, optional): If provided, the plot is saved to this path.
                                     Defaults to None.
    """
    # Set all models to evaluation mode. This turns off layers like Dropout.
    for model in models.values():
        model.eval()

    # Select random samples from the test set for visualization.
    random_indices = np.random.choice(len(X_test), size=num_samples, replace=False)
    sample_X = X_test[random_indices]
    sample_y = y_test[random_indices]

    # Create a grid of subplots.
    ncols = 3
    nrows = (num_samples + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 6 * nrows))
    fig.suptitle('Sine to Cosine Prediction on Test Set', fontsize=16)
    axes = axes.flatten()
    model_colors = {"Simple RNN": "skyblue", "LSTM": "salmon", "GRU": "lightgreen"}

    # Deactivate gradient calculations for inference.
    with torch.no_grad():
        for i in range(num_samples):
            ax = axes[i]
            x_input = sample_X[i]
            y_true = sample_y[i]

            # Plot the input sine wave and the ground truth cosine wave.
            ax.plot(x_input.numpy(), 'o--', label='Input (Sine)', color='black', alpha=0.7)
            ax.plot(y_true.numpy(), 'o-', label='True Output (Cosine)', color='dimgray', linewidth=2.5)

            # Generate and plot predictions for each model.
            for name, model in models.items():
                # Add a batch dimension (unsqueeze) for the model, then remove it
                # (squeeze) for plotting.
                y_pred = model(x_input.unsqueeze(0)).squeeze(0)
                ax.plot(y_pred.numpy(), 'x--', label=f'Pred: {name}', color=model_colors.get(name, None))

            ax.set_title(f'Test Sample #{i+1}')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True)

    # Hide any unused subplots.
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"INFO: Predictions plot saved to {output_path}")
    plt.show()