from loss import loss_fn
from model import sleep_model
from train import *
from utils import *

from braindecode.util import set_random_seeds

import os
import torch.nn as nn
import numpy as np
import wandb
import torch
from torch.utils.data import DataLoader, Dataset


PATH = '/scratch/allsamples/'

# Params
SAVE_PATH = "supervised_sleepedf_final.pth"
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128
lr = 5e-4
n_epochs = 200
NUM_WORKERS = 6
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
n_channels, input_size_samples = (1, 3000)
model = sleep_model(n_channels, input_size_samples, n_dim = N_DIM)


q_encoder = model.to(device)
linear_layer = nn.Linear(256,5).to(device)

optimizer = torch.optim.Adam(q_encoder.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()

#####################################################################################################

class train_data(Dataset):

    def __init__(self, filepath):
        
        self.file_path = filepath
        self.idx = np.array(range(len(self.file_path)))

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, index):
        
        path = self.file_path[index]
        data = np.load(path)
        
        return data['x'], data['y']
    
    

PRETEXT_FILE = os.listdir(os.path.join(PATH, "pretext_supervised"))
PRETEXT_FILE.sort(key=natural_keys)
PRETEXT_FILE = [os.path.join(PATH, "pretext_supervised", f) for f in PRETEXT_FILE]

TEST_FILE = os.listdir(os.path.join(PATH, "test_supervised"))
TEST_FILE.sort(key=natural_keys)
TEST_FILE = [os.path.join(PATH, "test_supervised", f) for f in TEST_FILE]

print(f'Number of pretext files: {len(PRETEXT_FILE)}')
print(f'Number of test records: {len(TEST_FILE)}')

pretext_loader = DataLoader(train_data(PRETEXT_FILE), batch_size=BATCH_SIZE, shuffle=True)

test_records = [np.load(f) for f in TEST_FILE]
test_subjects = dict()

for i, rec in enumerate(test_records):
    if rec['_description'][0] not in test_subjects.keys():
        test_subjects[rec['_description'][0]] = [rec]
    else:
        test_subjects[rec['_description'][0]].append(rec)

test_subjects = list(test_subjects.values())


##############################################################################################################################

wb = wandb.init(
        project="WTM-BASELINES",
        notes="Final",
        save_code=True,
        entity="sleep-staging",
        name="supervised-sleepedf-amp, T=1",
    )
wb.save('ssl-models/supervised_sleepedf/*.py')
wb.watch([q_encoder],log='all',log_freq=500)
Pretext(q_encoder,linear_layer, optimizer, n_epochs, criterion, pretext_loader, test_subjects, wb, device, SAVE_PATH, BATCH_SIZE)

wb.finish()
