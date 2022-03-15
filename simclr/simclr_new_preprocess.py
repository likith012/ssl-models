from augmentations import *
from loss import loss_fn
from model import sleep_model
from train_new_preprocess import *
from utils import *

from braindecode.util import set_random_seeds

import os
import numpy as np
import copy
import wandb
import torch
from torch.utils.data import DataLoader, Dataset


PATH = '/scratch/SLEEP_data2/'

# Params
SAVE_PATH = "single-epoch-same.pth"
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128
lr = 5e-4
n_epochs = 200
NUM_WORKERS = 10
N_DIM = 256
EPOCH_LEN = 7

####################################################################################################

random_state = 1234
sfreq = 100

# Seeds
rng = np.random.RandomState(random_state)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    print(f"GPU available: {torch.cuda.device_count()}")

set_random_seeds(seed=random_state, cuda=device == "cuda")


##################################################################################################


# Extract number of channels and time steps from dataset
n_channels, input_size_samples = (2, 3000)
model = sleep_model(n_channels, input_size_samples, n_dim = N_DIM)


q_encoder = model.to(device)

optimizer = torch.optim.Adam(q_encoder.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
criterion = loss_fn(device).to(device)

#####################################################################################################

PRETEXT_FILE = os.path.join(PATH,"pretext.pt")
pretext_ds = torch.load(PRETEXT_FILE)
pretext_ds = pretext_data(pretext_ds)
pretext_loader = DataLoader(pretext_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,pin_memory=True,persistent_workers=True)

##############################################################################################################################

wb = wandb.init(
        project="WTM-ssl_models",
        notes="single-epoch, 500 samples, using logistic regression with saga solver, with lr=5e-4",
        save_code=True,
        entity="sleep-staging",
        name="simclr-mulEEG-epochs, T=1",
    )
wb.save('ssl-models/simclr/*.py')
wb.watch([q_encoder],log='all',log_freq=500)

Pretext(q_encoder, optimizer, n_epochs, criterion, pretext_loader, wb, device, SAVE_PATH, BATCH_SIZE)

wb.finish()
