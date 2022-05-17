from augmentations import *
from loss import loss_fn
from model import contrast_loss
from train import *
from utils import *

from braindecode.util import set_random_seeds

import os
import numpy as np
import copy
import wandb
import torch
from torch.utils.data import DataLoader, Dataset
from config import Config


PATH = '/scratch/allsamples_shhs/'

# Params
SAVE_PATH = "mulEEG-shhs-final2.pth"
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128
lr = 5e-4
n_epochs = 200
NUM_WORKERS = 10
N_DIM = 256
EPOCH_LEN = 7
config = Config()

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
model = contrast_loss(config)

q_encoder = model.to(device)

optimizer = torch.optim.Adam(q_encoder.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
criterion = loss_fn(device).to(device)

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
        pos = data['pos']#(7, 2, 3000)
        pos_len = len(pos) # 7
        pos = pos[pos_len // 2] # (2, 3000)
        pos = np.expand_dims(pos[0],axis=0)
        anc = copy.deepcopy(pos)
        
        # augment
        pos = augment(pos)
        anc = augment(anc)

        return anc, pos
    

PRETEXT_FILE = os.listdir(os.path.join(PATH, "pretext"))
PRETEXT_FILE = [os.path.join(PATH, "pretext", f) for f in PRETEXT_FILE]

print(f'Number of pretext files: {len(PRETEXT_FILE)}')

pretext_loader = DataLoader(pretext_data(PRETEXT_FILE), batch_size=BATCH_SIZE, shuffle=True,num_workers=10)

TEST_FILE = os.listdir(os.path.join(PATH, "test"))
TEST_FILE = [os.path.join(PATH, "test", f) for f in TEST_FILE]
test_subjects = [np.load(file) for file in TEST_FILE]

##############################################################################################################################

wb = wandb.init(
        project="delete",
        notes="removed data with single channel",
        save_code=True,
        entity="sleep-staging",
        name="mulEEG-shhs-baseline",
    )
wb.save('ssl-models/simclr_shhs/*.py')
wb.watch([q_encoder],log='all',log_freq=500)

Pretext(q_encoder, optimizer, n_epochs, criterion, pretext_loader,test_subjects,wb,device, SAVE_PATH, BATCH_SIZE)

wb.finish()
