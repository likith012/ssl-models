from augmentations import *
from loss import loss_fn
from model import sleep_model
from train import *
from utils import *

from braindecode.util import set_random_seeds

import os
import numpy as np
import copy
import wandb
import torch
from torch.utils.data import DataLoader, Dataset

def main():
    
    #PATH = '/scratch/allsamples/'
    PATH = '/scratch/SLEEP_data/'

    # Params
    SAVE_PATH = "simclr_sleepedf_final.pth"
    CHKPOINT_PTH = './pretrained_weights/intra_v1_shhs_det_kfold_t10.pt'
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 128
    lr = 5e-4
    n_epochs = 200
    NUM_WORKERS = 6
    N_DIM = 256

    ####################################################################################################

    random_state = 1234

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
    chkpoint = torch.load(CHKPOINT_PTH,map_location=device)
    q_encoder.load_state_dict(chkpoint['eeg_model_state_dict'])

    optimizer = torch.optim.Adam(q_encoder.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    criterion = loss_fn(device).to(device)

    #####################################################################################################

    PRETEXT_FILE = os.listdir(os.path.join(PATH, "pretext"))
    PRETEXT_FILE = [os.path.join(PATH, "pretext", f) for f in PRETEXT_FILE]

    train_records = [np.load(f) for f in PRETEXT_FILE]
    train_subjects = dict()

    for i, rec in enumerate(train_records):
        if i not in train_subjects.keys():
            train_subjects[i] = [rec]
        else:
            train_subjects[i].append(rec)

    train_subjects = list(train_subjects.values())

    TEST_FILE = os.listdir(os.path.join(PATH, "test"))
    TEST_FILE = [os.path.join(PATH, "test", f) for f in TEST_FILE]

    print(f'Number of pretext files: {len(PRETEXT_FILE)}')
    print(f'Number of test records: {len(TEST_FILE)}')

    test_records = [np.load(f) for f in TEST_FILE]
    test_subjects = dict()

    for i, rec in enumerate(test_records):
        if i not in test_subjects.keys():
            test_subjects[i] = [rec]
        else:
            test_subjects[i].append(rec)

    test_subjects = list(test_subjects.values())


    ##############################################################################################################################

    wb = wandb.init(
            project="SHHS to SleepEDF",
            notes="Final",
            save_code=True,
            entity="sleep-staging",
            name="me moco shhs 7 on sleepedf, T=1"
        )
    wb.save('ssl-models/shhs_to_sleepedf/*.py')
    wb.watch([q_encoder],log='all',log_freq=500)

    Pretext(q_encoder, optimizer, n_epochs, criterion, train_subjects, test_subjects, wb, device, SAVE_PATH, BATCH_SIZE)

    wb.finish()
    
    
    
if __name__ == "__main__":
    
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    main()
