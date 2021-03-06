from augmentations import *
from loss import loss_fn
from model import sleep_model
from train import *
from utils import *

from braindecode.util import set_random_seeds
from braindecode.datautil.preprocess import zscore

import os
import numpy as np
import copy
import wandb
import torch
from torch.utils.data import DataLoader, Dataset


PATH = '/scratch/shhs_single_epoch/'

# Params
SAVE_PATH = "simclr_final.pth"
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128
lr = 5e-4
n_epochs = 200
NUM_WORKERS = 10
N_DIM = 256
EPOCH_LEN = 7
TEMPERATURE = 10

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
n_channels, input_size_samples = (2, 3750)
model = sleep_model(n_channels, input_size_samples, n_dim = N_DIM)


q_encoder = model.to(device)

optimizer = torch.optim.Adam(q_encoder.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
criterion = loss_fn(device,T=TEMPERATURE).to(device)

#####################################################################################################


class pretext_data(Dataset):

    def __init__(self, filepath):
        
        self.file_path = filepath
        self.idx = np.array(range(len(self.file_path)))

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, index):
        
        path = self.file_path[index]
        data = np.load(path)
        pos = data['pos'] #(7, 2, 3000)
        pos_len = len(pos) # 7
        pos = pos[pos_len // 2] # (2, 3000)
        pos = zscore(pos)
        anc = copy.deepcopy(pos)
        
        # augment
        pos = augment(pos)
        anc = augment(anc)
        return anc, pos
    
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
    
    

PRETEXT_FILE = os.listdir(os.path.join(PATH, "pretext"))
PRETEXT_FILE = [os.path.join(PATH, "pretext", f) for f in PRETEXT_FILE]

TEST_FILE = os.listdir(os.path.join(PATH, "test"))
TEST_FILE = [os.path.join(PATH, "test", f) for f in TEST_FILE]

print(f'Number of pretext files: {len(PRETEXT_FILE)}')
print(f'Number of test records: {len(TEST_FILE)}')

pretext_loader = DataLoader(pretext_data(PRETEXT_FILE), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

test_records = [np.load(f) for f in TEST_FILE]
test_subjects = dict()

for i, rec in enumerate(test_records):
    if i not in  test_subjects:
        test_subjects[i]=rec

test_subjects = list(test_subjects.values())

##############################################################################################################################

wb = wandb.init(
        project="WTM-ssl",
        notes="single-epoch, 500 samples, using logistic regression with saga solver, with lr=5e-4, augmentations_new",
        save_code=True,
        entity="sleep-staging",
        name="simclr-final-shhs, T=10",
    )
wb.save('ssl-models/simclr/*.py')
wb.watch([q_encoder],log='all',log_freq=500)

Pretext(q_encoder, optimizer, n_epochs, criterion, pretext_loader, test_subjects, wb, device, SAVE_PATH, BATCH_SIZE)

wb.finish()
