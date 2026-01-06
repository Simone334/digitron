"""
Main executable script for the PyTorch Introduction (Step 8).

This script performs the following steps:
1.  Defines hyperparameters for the models and data generation.
2.  Generates the synthetic sine/cosine wave dataset.
3.  Splits the data into training and testing sets and creates DataLoaders.
4.  Initializes and trains three types of recurrent models:
    a) Simple RNN
    b) LSTM
    c) GRU
5.  Visualizes a comparison of the training loss curves for the three models.
6.  Visualizes the models' predictions on a set of random test samples.
"""
import os
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# Import all necessary functions from our structured 'src' directory
from src.patrec.pytorch_intro.data import generate_sine_cosine_data
from src.patrec.pytorch_intro.model import BaseModel
from src.patrec.pytorch_intro.training import train_model
from src.patrec.pytorch_intro.visualization import (
    plot_loss_comparison,
    plot_predictions,
)

# --- 1. Configuration and Hyperparameters ---
# Data parameters
NUM_SEQUENCES = 1000
SEQ_LENGTH = 10
FREQUENCY = 40

# Model parameters
INPUT_DIM = 1
HIDDEN_DIM = 32
OUTPUT_DIM = 1

# Training parameters
EPOCHS = 100
LEARNING_RATE = 0.01
BATCH_SIZE = 512

# Output directory for plots
OUTPUT_DIR = "outputs/pytorch_intro_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. Data Generation and Preparation ---
print("--- [Step 1] Generating and Preparing Data ---")
X_data, y_data = generate_sine_cosine_data(
    num_sequences=NUM_SEQUENCES, seq_length=SEQ_LENGTH, freq=FREQUENCY
)

# Split data into training (80%) and testing (20%) sets.
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42
)

# Create a PyTorch Dataset and DataLoader for efficient batching during training.
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Data prepared. Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# --- 3. Comparative Model Training ---
print("\n--- [Step 2] Training Comparative Models ---")
# Initialize the three different types of recurrent models.
simple_rnn_model = BaseModel(nn.RNN, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
lstm_model = BaseModel(nn.LSTM, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
gru_model = BaseModel(nn.GRU, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)

# Train each model and store its loss history.
rnn_losses = train_model(simple_rnn_model, train_loader, epochs=EPOCHS, learning_rate=LEARNING_RATE)
lstm_losses = train_model(lstm_model, train_loader, epochs=EPOCHS, learning_rate=LEARNING_RATE)
gru_losses = train_model(gru_model, train_loader, epochs=EPOCHS, learning_rate=LEARNING_RATE)

# --- 4. Visualization of Results ---
print("\n--- [Step 3] Visualizing Training Results ---")
# Store all trained models and their loss histories in dictionaries for easy access.
models = {"Simple RNN": simple_rnn_model, "LSTM": lstm_model, "GRU": gru_model}
loss_histories = {"Simple RNN": rnn_losses, "LSTM": lstm_losses, "GRU": gru_losses}

# Plot the comparison of training losses.
plot_loss_comparison(
    loss_histories, output_path=os.path.join(OUTPUT_DIR, "loss_comparison.png")
)

# Plot the predictions on random test samples.
plot_predictions(
    models, X_test, y_test, num_samples=6, output_path=os.path.join(OUTPUT_DIR, "predictions.png")
)

print("\nPyTorch introductory exercise complete.")