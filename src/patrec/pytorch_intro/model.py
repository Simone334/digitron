"""
This module defines the neural network architecture for the PyTorch introductory
exercise (Step 8). It features a generic BaseModel that can wrap different
recurrent units like RNN, LSTM, or GRU.
"""
import torch.nn as nn
import torch


class BaseModel(nn.Module):
    """
    A generic sequence-to-sequence model that can use different recurrent units.

    This model consists of a recurrent layer (which can be a simple RNN, LSTM, or GRU)
    followed by a fully connected (Linear) layer that maps the hidden state of each
    time step to an output value.

    Attributes:
        rnn (nn.Module): The core recurrent layer (e.g., nn.LSTM instance).
        fc (nn.Linear): The final fully connected layer for output prediction.
    """

    def __init__(
        self,
        rnn_unit: nn.Module,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ):
        """
        Initializes the BaseModel.

        Args:
            rnn_unit (nn.Module): The type of the recurrent unit to use.
                                  This should be an uninitialized class like
                                  `nn.RNN`, `nn.LSTM`, or `nn.GRU`.
            input_dim (int): The number of features in the input at each time step
                             (e.g., 1 for a single sine wave value).
            hidden_dim (int): The number of features in the hidden state of the
                              recurrent unit.
            output_dim (int): The number of output features desired at each time step
                              (e.g., 1 for a single cosine wave value).
        """
        super(BaseModel, self).__init__()

        # The core recurrent layer.
        # `batch_first=True` is crucial. It means the input and output tensors
        # will have the batch dimension as the first dimension (N, L, D), which
        # is more intuitive than the default (L, N, D).
        self.rnn = rnn_unit(input_dim, hidden_dim, batch_first=True)

        # A fully connected (Linear) layer that maps the hidden state output of
        # the RNN to the desired output dimension.
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor of shape (N, L, D), where
                              N is the batch size, L is the sequence length,
                              and D is the input dimension.

        Returns:
            torch.Tensor: The output tensor of shape (N, L, O), where O is the
                          output dimension.
        """
        # The RNN layer returns two things:
        # 1. `out`: A tensor containing the output of the last layer for each
        #           time step. Shape: (N, L, H), where H is the hidden_dim.
        # 2. `_`: The final hidden state (and cell state for LSTM). We don't
        #         need it for this sequence-to-sequence task, so we ignore it
        #         with the underscore convention.
        out, _ = self.rnn(x)

        # We apply the fully connected layer to every time step's hidden output.
        # PyTorch's nn.Linear layer can handle this automatically. It applies the
        # transformation to the last dimension (the hidden features) of the `out`
        # tensor, resulting in a tensor of shape (N, L, O).
        out = self.fc(out)

        return out