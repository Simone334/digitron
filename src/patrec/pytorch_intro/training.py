"""
This module contains the training logic for the PyTorch introductory exercise
(Step 8), including the main training loop function.
"""
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_model(
    model: nn.Module, train_loader: DataLoader, epochs: int = 100, learning_rate: float = 0.01
) -> List[float]:
    """
    A generic function to train a PyTorch sequence-to-sequence model.

    This function iterates through the training data for a specified number of
    epochs, performing the forward pass, loss calculation, backward pass, and
    optimizer step.

    Args:
        model (nn.Module): The PyTorch model instance to be trained.
        train_loader (DataLoader): The DataLoader containing the training data.
        epochs (int): The total number of epochs to train for.
        learning_rate (float): The learning rate for the Adam optimizer.

    Returns:
        List[float]: A list containing the average training loss for each epoch.
    """
    # Use Mean Squared Error (MSE) loss, as this is a regression problem
    # (predicting continuous cosine values).
    criterion = nn.MSELoss()

    # Use the Adam optimizer, a popular and effective choice for many models.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # List to store the average loss of each epoch for later plotting.
    loss_history = []

    # Set the model to training mode. This is important for layers like Dropout
    # or BatchNorm, although not strictly necessary for this simple model.
    model.train()

    print(f"--- Training {model.rnn.__class__.__name__} for {epochs} epochs ---")
    for epoch in range(epochs):
        # Accumulate loss over an epoch to calculate the average.
        epoch_loss = 0.0

        # Iterate over all batches provided by the DataLoader.
        for sequences, labels in train_loader:
            # 1. Reset the gradients from the previous iteration.
            optimizer.zero_grad()

            # 2. Perform the forward pass to get model predictions.
            predictions = model(sequences)

            # 3. Calculate the loss between predictions and true labels.
            loss = criterion(predictions, labels)

            # 4. Perform the backward pass to compute gradients.
            loss.backward()

            # 5. Update the model's weights using the computed gradients.
            optimizer.step()

            # Add the loss of the current batch to the epoch's total loss.
            # .item() extracts the scalar value from the loss tensor.
            epoch_loss += loss.item()

        # Calculate the average loss for the epoch.
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)

        # Print progress every 10 epochs.
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1: >3}/{epochs}, Average Loss: {avg_loss:.6f}")

    print("-> Training complete.")
    return loss_history