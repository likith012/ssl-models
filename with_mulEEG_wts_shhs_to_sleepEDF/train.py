from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    #f1_score,
    confusion_matrix,
    balanced_accuracy_score
)
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Dataset
from pl_bolts.models.regression import LogisticRegression
import pytorch_lightning as pl
from pl_bolts.datamodules.sklearn_datamodule import SklearnDataset
from pytorch_lightning.callbacks import EarlyStopping
from torch.nn.functional import interpolate
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchmetrics.functional import accuracy,f1,cohen_kappa

import time
import logging
import warnings
import os

logging.getLogger("lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


def task(q_encoder,train_loader,test_loader,device, i):
    
    class FinModel(torch.nn.Module):

        def __init__(self,q_encoder):
            super(FinModel,self).__init__()

            self.q_encoder = q_encoder

            for p in self.q_encoder.parameters():
                p.requires_grad = False

            self.linear_layer = torch.nn.Linear(256,5)

        def forward(self,x):
            return self.linear_layer(self.q_encoder(x))
    
    model = FinModel(q_encoder).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0003,betas=(0.9,0.99),weight_decay=3e-5)
    scheduler = ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=5)

    for epoch in range(200):

        model.train()

        for x,y in tqdm(train_loader):
            x = x[:,:1,:].float().to(device)
            y = y.long().to(device)

            preds = model(x)
            loss = criterion(preds,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()

        with torch.no_grad():

            y =  []
            preds = []
            lss = []
            
            for x,y_each in tqdm(test_loader):
                x = x[:,:1,:].float().to(device)
                y_each = y_each.long()
                y.append(y_each)

                preds_each = model(x).cpu()
                criterion(preds_each,y_each.cpu())
                #preds_each = torch.argmax(preds_each, axis = 1)
                preds.append(preds_each)

                lss.append(loss.item())

            scheduler.step(sum(lss)/len(lss))

            y = torch.cat(y)
            preds = torch.cat(preds)

            acc = accuracy(preds,y)
            #cm = confusion_matrix(y, preds)
            cm = 0
            f1_value = f1(preds,y, average="macro",num_classes=5)
            kappa = cohen_kappa(preds,y,num_classes=5)
            #bal_acc = balanced_accuracy_score(y, preds)
            bal_acc = 0

            print("+"*50)
            print(f"Epoch: {epoch} acc: {acc}")
            print(f"Epoch: {epoch} f1: {f1_value}")
            print(f"Epoch: {epoch} kappa: {kappa}")
            print(f"Epoch: {epoch} bal_acc: {bal_acc}")
            print(f"Epoch: {epoch} lr: {scheduler.optimizer.param_groups[0]['lr']}")
            print("+"*50)
                
    return acc, cm, f1_value, kappa, bal_acc, y, preds

def kfold_evaluate(q_encoder,train_subjects, test_subjects, device, BATCH_SIZE):

    total_acc, total_f1, total_kappa, total_bal_acc = [], [], [], []
    i = 1


    test_subjects_train = train_subjects
    test_subjects_test = test_subjects
    test_subjects_train = [rec for sub in test_subjects_train for rec in sub]
    test_subjects_test = [rec for sub in test_subjects_test for rec in sub]

    #train_loader = DataLoader(TuneDataset(os.path.join("/scratch","SLEEP_data","pretext.pt")), batch_size=BATCH_SIZE*2, shuffle=True,num_workers=6)
    #test_loader = DataLoader(TuneDataset(os.path.join("/scratch","SLEEP_data","test.pt")), batch_size=BATCH_SIZE*2, shuffle= False,num_workers=6)
    train_loader = DataLoader(TuneDataset(test_subjects_train), batch_size=BATCH_SIZE*2, shuffle=True,num_workers=6)
    test_loader = DataLoader(TuneDataset(test_subjects_test), batch_size=BATCH_SIZE*2, shuffle= False,num_workers=6)
    test_acc, _, test_f1, test_kappa, bal_acc, gt, pd = task(q_encoder, train_loader, test_loader, device, i)

    total_acc.append(test_acc)
    total_f1.append(test_f1)
    total_kappa.append(test_kappa)
    total_bal_acc.append(bal_acc)
    
    print("+"*50)
    print(f"Fold: {i} acc: {test_acc}")
    print(f"Fold: {i} f1: {test_f1}")
    print(f"Fold: {i} kappa: {test_kappa}")
    print(f"Fold: {i} bal_acc: {bal_acc}")
    print("+"*50)
    i+=1 

    return np.mean(total_acc), np.mean(total_f1), np.mean(total_kappa), np.mean(total_bal_acc)

class TuneDataset(Dataset):
    """Dataset for train and test"""

    def __init__(self, subjects):
        self.subjects = subjects
        self._add_subjects()

    def __getitem__(self, index):

        X = self.X[index]
        y =  self.y[index]
        return X, y

    def __len__(self):
        return self.X.shape[0]
        
    def _add_subjects(self):
        self.X = []
        self.y = []
        for subject in self.subjects:
            self.X.append(np.transpose(subject['x'],(0,2,1)))
            self.y.append(subject['y'])
        self.X = np.concatenate(self.X, axis=0)
        self.y = np.concatenate(self.y, axis=0)

#class TuneDataset(Dataset):
#    """Dataset for train and test"""
#
#    def __init__(self, pt_file):
#        self.dat = torch.load(pt_file)
#        self.x = self.dat['samples']
#        self.y = self.dat['labels']
#
#    def __getitem__(self, index):
#
#        X = self.x[index]
#        y =  self.y[index]
#        return X, y
#
#    def __len__(self):
#        return self.x.shape[0]



##################################################################################################################################################


# Pretrain
def Pretext(
    q_encoder,
    optimizer,
    Epoch,
    criterion,
    train_subjects,
    test_subjects,
    wandb,
    device, 
    SAVE_PATH,
    BATCH_SIZE
):


    best_f1 = 0
    test_acc, test_f1, test_kappa, bal_acc = kfold_evaluate(
        q_encoder,train_subjects, test_subjects, device, BATCH_SIZE
    )

    wandb.log({"Valid Acc": test_acc, "Epoch": 0})
    wandb.log({"Valid F1": test_f1, "Epoch": 0})
    wandb.log({"Valid Kappa": test_kappa, "Epoch": 0})
    wandb.log({"Valid Balanced Acc": bal_acc, "Epoch": 0})

    if test_f1 > best_f1:   
        best_f1 = test_f1
        torch.save(q_encoder.enc.state_dict(), SAVE_PATH)
        wandb.save(SAVE_PATH)
        print("save best model on test set with best F1 score")
