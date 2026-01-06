"""
This module contains the data generation logic for the PyTorch introductory
exercise (Step 8), creating sine and cosine wave sequences.
"""
from typing import Tuple

import numpy as np
import torch


def generate_sine_cosine_data(
    num_sequences: int = 1000, seq_length: int = 10, freq: int = 40
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates sequences of sine waves (input) and cosine waves (target).

    This function creates a batch of overlapping sequences suitable for training
    a sequence-to-sequence model. The input `X` will be a sine wave segment,
    and the target `y` will be the corresponding cosine wave segment.

    Args:
        num_sequences (int): The total number of sequence pairs to generate.
        seq_length (int): The length of each individual sequence.
        freq (int): The frequency of the sine and cosine waves in Hz.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
            - X_tensor: The input data (sine waves) with shape
                        (num_sequences, seq_length, 1).
            - y_tensor: The target data (cosine waves) with shape
                        (num_sequences, seq_length, 1).
    """
    print(
        f"Generating {num_sequences} sine/cosine sequences of length {seq_length}..."
    )
    # To properly represent a high-frequency wave, the sampling rate must be
    # at least twice the frequency (Nyquist theorem). A higher rate provides
    # a smoother representation.
    sampling_rate = 20 * freq  # Sampling Frequency in Hz

    # To create 'num_sequences' of overlapping sequences, we need a total
    # number of points equal to the number of sequences plus the length of one sequence.
    num_total_points = num_sequences + seq_length

    # Create a single, long time vector for the entire wave generation.
    time_steps = np.linspace(
        0, num_total_points / sampling_rate, num_total_points, endpoint=False
    )

    # Generate the underlying continuous sine and cosine waves.
    sin_wave = np.sin(2 * np.pi * freq * time_steps)
    cos_wave = np.cos(2 * np.pi * freq * time_steps)

    # Create the overlapping sequences using a sliding window approach.
    X, y = [], []
    for i in range(num_sequences):
        X.append(sin_wave[i : i + seq_length])
        y.append(cos_wave[i : i + seq_length])

    # Convert the Python lists of sequences into NumPy arrays.
    X_np = np.array(X)
    y_np = np.array(y)

    # Reshape the arrays to the format expected by PyTorch's RNN layers:
    # (batch_size, sequence_length, input_dimension).
    # Here, input_dimension is 1 because we have a single value at each time step.
    X_np = X_np.reshape(-1, seq_length, 1)
    y_np = y_np.reshape(-1, seq_length, 1)

    # Convert NumPy arrays to PyTorch tensors of type float32.
    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.float32)

    print("-> Data generation complete.")
    return X_tensor, y_tensor