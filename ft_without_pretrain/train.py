from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score,
    roc_auc_score
)
import torch
import torch.nn as nn
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

import time
import logging
import warnings
from pytorch_lightning import seed_everything

seed_everything(1234, workers=True)

logging.getLogger("lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


# Train, test
def evaluate(q_encoder, linear_layer, test_loader, device,i):

    # eval
    q_encoder.eval()

    emb_test, gt_test = [], []

    with torch.no_grad():
        for (X_test, y_test) in test_loader:
            X_test = X_test.float()
            y_test = y_test.long()
            X_test = X_test.to(device)
            emb_test.extend(linear_layer(q_encoder(X_test, proj="mid")).cpu().tolist())
            gt_test.extend(y_test.numpy().flatten())

    emb_test, gt_test = np.array(emb_test), np.array(gt_test)

    acc, cm, f1, kappa, bal_acc, gt, pd, auc = task(emb_test,gt_test,i)

    q_encoder.train()
    return acc, f1, kappa, bal_acc, auc


def task(X_test,y_test,i):

    start = time.time()

    pred = X_test
    pred = pred>0.5

    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    f1 = f1_score(y_test, pred)
    kappa = cohen_kappa_score(y_test, pred)
    bal_acc = balanced_accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test,pred)

    pit = time.time() - start
    print(f"Took {int(pit // 60)} min:{int(pit % 60)} secs for {i} fold")

    return acc, cm, f1, kappa, bal_acc, y_test, pred, auc


##################################################################################################################################################


# Pretrain
def Pretext(
    q_encoder,
    linear_layer,
    optimizer,
    Epoch,
    criterion,
    pretext_loader,
    test_loader,
    wandb,
    device,
    SAVE_PATH,
    BATCH_SIZE
):


    step = 0
    best_f1 = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=5
    )

    acc_score = []
    pretext_loss = []

    sig = nn.Sigmoid()

    for epoch in range(Epoch):

        print('=========================================================\n')
        print("Epoch: {}".format(epoch))
        print('=========================================================\n')
        all_loss = []
        for index, (x,y) in enumerate(
            tqdm(pretext_loader, desc="pretrain")
        ):
            q_encoder.train()
            linear_layer.train()
            x = x.float()
            y = y.float()
            #y = y.long()

            x, y= (
                x.to(device),
                y.to(device),
            )

#         with torch.cuda.amp.autocast():
            features = q_encoder(x, proj = 'mid') #(B, 128)

            # get preds
            preds = linear_layer(features)

            # backprop
            loss = criterion(sig(preds).squeeze(-1),y)

            # loss back
            all_loss.append(loss.item())
            pretext_loss.append(loss.cpu().detach().item())

            optimizer.zero_grad()
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
            loss.backward()
            optimizer.step()


        scheduler.step(sum(all_loss[-50:]))
        lr = optimizer.param_groups[0]["lr"]
        wandb.log({"ssl_lr": lr, "Epoch": epoch})


        wandb.log({"ssl_loss": np.mean(pretext_loss), "Epoch": epoch})

        if epoch >=10 and (epoch) % 5 == 0:

            test_acc, test_f1, test_kappa, bal_acc, auc = evaluate(q_encoder,linear_layer,test_loader,device,0)
            wandb.log({"Valid Acc": test_acc, "Epoch": epoch})
            wandb.log({"Valid F1": test_f1, "Epoch": epoch})
            wandb.log({"Valid Kappa": test_kappa, "Epoch": epoch})
            wandb.log({"Valid Balanced Acc": bal_acc, "Epoch": epoch})
            wandb.log({"Valid AUC": auc, "Epoch": epoch})

            if test_f1 > best_f1:
                best_f1 = test_f1
                torch.save(q_encoder.enc.state_dict(), SAVE_PATH)
                wandb.save(SAVE_PATH)
                print("save best model on test set with best F1 score")
