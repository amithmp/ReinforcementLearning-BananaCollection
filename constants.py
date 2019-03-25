import torch

print("Loading new constants")

TARGET_SCORE = 13

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 6e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
NEURON_FC1 = 256
NEURON_FC2 = 128
eps_start=1.0
eps_end=0.01
eps_decay=0.992
NUM_EPOCHS = 1

n_episodes=2000
max_t=1000

USE_LSTM = True
NUM_LAYERS_LSTM = 3
SEQUENCE_LEN = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

