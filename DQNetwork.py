import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import namedtuple, deque

from constants import *

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, NEURON_FC1)
        self.fc2 = nn.Linear(NEURON_FC1, NEURON_FC2)
        self.fc3 = nn.Linear(NEURON_FC2, action_size)
        self.is_lstm = False
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class QNetworkLSTM(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetworkLSTM, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.lstm = nn.LSTM(state_size, NEURON_FC1, NUM_LAYERS_LSTM, batch_first=True)
        self.fc = nn.Linear(NEURON_FC1, action_size)
        self.is_lstm = True
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        h0 = torch.zeros(NUM_LAYERS_LSTM, state.size(0), NEURON_FC1).to(device) 
        c0 = torch.zeros(NUM_LAYERS_LSTM, state.size(0), NEURON_FC1).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(state, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out