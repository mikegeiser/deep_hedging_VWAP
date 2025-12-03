import numpy as np
import sys
from pathlib import Path

# Make repo root importable (so 'lob_simulator' works when run from deep_hedging/)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lob_simulator.santa_fe_param import L


# --- Neural Network Architecture ---
N           = 32 # Trading Nodes
num_layers  = 2   # Depth
num_neurons = 16  # Width
num_outputs = L+1 # Outputs per time step (phi, theta)

# --- Training Parameters ---
epochs                  = 1000  # Number of epochs
batch_size              = 10000 # Number of batches per epoch
#optimizer              = Adam
learning_rate           = 1e-1
early_stopping_patience = 10
lr_schedule_factor      = 0.5
lr_schedule_patience    = 5
min_lr                  = 2e-10