from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score
)
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from tqdm import tqdm
import numpy as np
import os
import copy
from torch.utils.data import DataLoader, Dataset
from augmentations import *

class pretext_data(Dataset):

    def __init__(self, dataset,wh="pretext"):
        super(pretext_data,self).__init__()
        dataset['samples'] = dataset['samples'][:500]
        dataset['labels'] = dataset['labels'][:500]
        X_train = dataset["samples"]
        y_train = dataset["labels"]


        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
        else:
            self.x_data = X_train
        if isinstance(y_train,np.ndarray):
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.y_data = y_train

        self.len = X_train.shape[0]

        self.wh = wh


    def __len__(self):
        return self.len

    def __getitem__(self, index):

        if self.wh=="pretext":
            pos = self.x_data[index]
            anc = pos.detach().clone()
            pos = pos.cpu().detach().numpy()
            anc = anc.cpu().detach().numpy()
            ## augment
            pos = augment(pos)
            anc = augment(anc)
            return anc, pos

        else:
            return self.x_data[index],self.y_data[index]

def cross_data_generator(data_path,train_idxs,val_idxs,BATCH_SIZE):
    train_ds = torch.load(os.path.join(data_path, "train.pt"))
    #train_ds['samples'] = train_ds['samples'][:,0,:].unsqueeze(1)
    valid_ds = torch.load(os.path.join(data_path, "val.pt"))
    #valid_ds['samples'] = valid_ds['samples'][:,0,:].unsqueeze(1)

    train_subs = [48,72,24,30,34,50,38,15,60,12]
    train_segs = [3937, 2161, 3448, 1783, 3083, 2429, 3647, 2714, 3392, 2029]

    val_subs = [23,26,37,44,49,51,54,59,73,82]
    val_segs = [2633, 2577, 2427, 2287, 2141, 2041, 2864, 3071, 4985, 3070]

    segs = train_segs+val_segs

    if train_idxs !=[]:

        dataset = {}
        train_dataset = {}
        valid_dataset = {}
        dataset['samples'] = torch.from_numpy(np.vstack((train_ds['samples'],valid_ds['samples'])))
        dataset['labels'] = torch.from_numpy(np.hstack((train_ds['labels'],valid_ds['labels'])))

        dataset['samples'] = torch.split(dataset['samples'],segs)
        dataset['labels'] = torch.split(dataset['labels'],segs)
        print('Split Shape',len(dataset['samples']))

        train_dataset['samples'] = [dataset['samples'][i] for i in train_idxs]
        train_dataset['labels'] = [dataset['labels'][i] for i in train_idxs]

        train_dataset['samples'] = torch.cat(train_dataset['samples'])
        train_dataset['labels'] = torch.cat(train_dataset['labels'])

        print('Train Shape',train_dataset['samples'].shape,train_dataset['labels'].shape)
        train_dataset = pretext_data(train_dataset,wh="train")

        valid_dataset['samples'] = [dataset['samples'][i] for i in val_idxs]
        valid_dataset['labels'] = [dataset['labels'][i] for i in val_idxs]

        valid_dataset['samples'] = torch.cat(valid_dataset['samples'])
        valid_dataset['labels'] = torch.cat(valid_dataset['labels'])
        valid_dataset = pretext_data(valid_dataset,wh="valid")


        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                                                   shuffle=True, drop_last=True,
                                                   num_workers=10,pin_memory=True,persistent_workers=True)

        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE,
                                                   shuffle=False, drop_last=True,
                                                   num_workers=10,pin_memory=True,persistent_workers=True)

        del dataset
        del train_dataset
        del valid_dataset

        return train_loader,valid_loader
    ret = len(val_subs)+len(train_subs)
    del train_ds
    del valid_ds
    return ret


# Train, test
def evaluate(q_encoder, train_loader, test_loader, device):

    # eval
    q_encoder.eval()

    # process val
    emb_val, gt_val = [], []

    with torch.no_grad():
        for (X_val, y_val) in train_loader:
            X_val = X_val.float()
            y_val = y_val.long()
            X_val = X_val.to(device)
            emb_val.extend(q_encoder(X_val).cpu().tolist())
            gt_val.extend(y_val.numpy().flatten())
    emb_val, gt_val = np.array(emb_val), np.array(gt_val)

    emb_test, gt_test = [], []

    with torch.no_grad():
        for (X_test, y_test) in test_loader:
            X_test = X_test.float()
            y_test = y_test.long()
            X_test = X_test.to(device)
            emb_test.extend(q_encoder(X_test).cpu().tolist())
            gt_test.extend(y_test.numpy().flatten())

    emb_test, gt_test = np.array(emb_test), np.array(gt_test)

    acc, cm, f1, kappa, bal_acc, gt, pd = task(emb_val, emb_test, gt_val, gt_test)

    q_encoder.train()
    return acc, cm, f1, kappa, bal_acc, gt, pd


def task(X_train, X_test, y_train, y_test):
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    cls = LogisticRegression(penalty='l2', C=1.0, solver="saga", class_weight='balanced', multi_class="multinomial", max_iter= 3000, n_jobs=-1, dual = False, random_state=1234)
    cls.fit(X_train, y_train)
    pred = cls.predict(X_test)

    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")
    kappa = cohen_kappa_score(y_test, pred)
    bal_acc = balanced_accuracy_score(y_test, pred)

    return acc, cm, f1, kappa, bal_acc, y_test, pred

def kfold_evaluate(q_encoder, device, BATCH_SIZE):

    n = cross_data_generator(PATH,[],[],BATCH_SIZE)
    kfold = KFold(n_splits=5, shuffle=False)
    idxs = np.arange(0,n,1)

    total_acc, total_f1, total_kappa, total_bal_acc = [], [], [], []
    i = 1

    for split,(train_idx,val_idx) in enumerate(kfold.split(idxs)):
        train_loader,test_loader = cross_data_generator(PATH,train_idx,val_idx)
        test_acc, _, test_f1, test_kappa, bal_acc, gt, pd = evaluate(q_encoder, train_loader, test_loader, device)

        total_acc.append(test_acc)
        total_f1.append(test_f1)
        total_kappa.append(test_kappa)
        total_bal_acc.append(bal_acc)
        
        print("+"*50)
        print(f"Fold{i} acc: {test_acc}")
        print(f"Fold{i} f1: {test_f1}")
        print(f"Fold{i} kappa: {test_kappa}")
        print(f"Fold{i} bal_acc: {bal_acc}")
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
            self.X.append(subject['windows'])
            self.y.append(subject['y'])
        self.X = np.concatenate(self.X, axis=0)
        self.y = np.concatenate(self.y, axis=0)



##################################################################################################################################################


# Pretrain
def Pretext(
    q_encoder,
    optimizer,
    Epoch,
    criterion,
    pretext_loader,
    wandb,
    device, 
    SAVE_PATH,
    BATCH_SIZE
):

    q_encoder.train()  # for dropout

    step = 0
    best_f1 = 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=5
    )

    all_loss, acc_score = [], []
    pretext_loss = []

    for epoch in range(Epoch):
        
        print('=========================================================\n')
        print("Epoch: {}".format(epoch))
        print('=========================================================\n')
        
        for index, (aug1, aug2) in enumerate(
            tqdm(pretext_loader, desc="pretrain")
        ):

            aug1 = aug1.float()
            aug2 = aug2.float()

            aug1, aug2 = (
                aug1.to(device),
                aug2.to(device),
            )  # (B, 7, 2, 3000)  (B, 7, 2, 3000) (B, 7, 2, 3000)
        
            anc1_features = q_encoder(aug1, proj_first='yes') #(B, 128)
            pos1_features = q_encoder(aug2, proj_first='yes')  # (B, 128)
            
            # backprop
            loss = criterion(anc1_features, pos1_features)

            # loss back
            all_loss.append(loss.item())
            pretext_loss.append(loss.cpu().detach().item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # only update encoder_q

            N = 1000
            if (step + 1) % N == 0:
                scheduler.step(sum(all_loss[-50:]))
                lr = optimizer.param_groups[0]["lr"]
                wandb.log({"ssl_lr": lr, "Epoch": epoch})
            step += 1

        wandb.log({"ssl_loss": np.mean(pretext_loss), "Epoch": epoch})

        if epoch >= 40 and (epoch) % 5 == 0:

            test_acc, test_f1, test_kappa, bal_acc = kfold_evaluate(
                q_encoder, device, BATCH_SIZE
            )

            wandb.log({"Valid Acc": test_acc, "Epoch": epoch})
            wandb.log({"Valid F1": test_f1, "Epoch": epoch})
            wandb.log({"Valid Kappa": test_kappa, "Epoch": epoch})
            wandb.log({"Valid Balanced Acc": bal_acc, "Epoch": epoch})

            if test_f1 > best_f1:   
                best_f1 = test_f1
                torch.save(q_encoder.enc.state_dict(), SAVE_PATH)
                wandb.save(SAVE_PATH)
                print("save best model on test set with best F1 score")
