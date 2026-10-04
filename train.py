import os
import json
import random
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

import nibabel as nib
from tqdm import tqdm
from transformers import AutoModel


# =====================================================
# LOAD ENV
# =====================================================

load_dotenv()

DATASET_ROOT = os.getenv("DATASET_ROOT")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")

BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
EPOCHS = int(os.getenv("EPOCHS"))
LR = float(os.getenv("LR"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY"))

PATIENCE = int(os.getenv("PATIENCE"))

NUM_CLASSES = int(os.getenv("NUM_CLASSES"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS"))

PIN_MEMORY = os.getenv("PIN_MEMORY") == "True"

DEVICE = os.getenv("DEVICE")
BACKBONE = os.getenv("BACKBONE")

device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

BEST_MODEL = os.path.join(OUTPUT_DIR,"best_model.pth")
LOG_FILE = os.path.join(OUTPUT_DIR,"training_log.txt")
METRICS_CSV = os.path.join(OUTPUT_DIR,"training_metrics.csv")
PARAM_CSV = os.path.join(OUTPUT_DIR,"model_parameters.csv")
TEST_SUBJECTS_FILE = os.path.join(OUTPUT_DIR,"test_subjects.txt")


# =====================================================
# SEED
# =====================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)


# =====================================================
# LOGGER
# =====================================================

def log(msg):

    print(msg)

    with open(LOG_FILE,"a") as f:
        f.write(msg+"\n")


# =====================================================
# DATASET
# =====================================================

class FeTADataset(Dataset):

    def __init__(self,subjects,root):

        self.samples=[]

        for s in subjects:

            anat=os.path.join(root,s,"anat")

            img_path=os.path.join(anat,f"{s}_rec-irtk_T2w.nii.gz")
            lbl_path=os.path.join(anat,f"{s}_rec-irtk_dseg.nii.gz")

            if not os.path.exists(img_path):
                continue

            img_vol=nib.load(img_path).get_fdata().astype("float32")
            lbl_vol=nib.load(lbl_path).get_fdata().astype("int64")

            img_vol=(img_vol-img_vol.min())/(img_vol.max()-img_vol.min()+1e-6)

            depth=img_vol.shape[2]

            for z in range(depth):

                img=torch.tensor(img_vol[:,:,z]).unsqueeze(0)
                img=img.repeat(3,1,1)

                img=F.interpolate(
                    img.unsqueeze(0),
                    size=(IMAGE_SIZE,IMAGE_SIZE),
                    mode="bilinear",
                    align_corners=False
                ).squeeze(0)

                lbl=torch.tensor(lbl_vol[:,:,z]).unsqueeze(0).float()

                lbl=F.interpolate(
                    lbl.unsqueeze(0),
                    size=(IMAGE_SIZE,IMAGE_SIZE),
                    mode="nearest"
                ).squeeze(0).long().squeeze(0)

                self.samples.append((img,lbl))

        log(f"Total slices: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):
        return self.samples[idx]


# =====================================================
# LORA
# =====================================================

class LoRALinear(nn.Module):

    def __init__(self,linear,rank=4):

        super().__init__()

        self.linear=linear

        in_f=linear.in_features
        out_f=linear.out_features

        self.A=nn.Parameter(torch.randn(rank,in_f)*0.01)
        self.B=nn.Parameter(torch.randn(out_f,rank)*0.01)

        for p in self.linear.parameters():
            p.requires_grad=False

    def forward(self,x):

        original=self.linear(x)

        lora=x@self.A.T
        lora=lora@self.B.T

        return original+lora


# =====================================================
# BACKBONE
# =====================================================

class DinoBackboneLoRA(nn.Module):

    def __init__(self):

        super().__init__()

        self.model=AutoModel.from_pretrained(BACKBONE)

        for p in self.model.parameters():
            p.requires_grad=False

        for block in self.model.encoder.layer:

            block.attention.attention.query=LoRALinear(
                block.attention.attention.query
            )

            block.attention.attention.value=LoRALinear(
                block.attention.attention.value
            )

    def forward(self,x):

        out=self.model(pixel_values=x)

        tokens=out.last_hidden_state[:,1:,:]

        return tokens

# =====================================================
# SEGMENTATION MODEL
# =====================================================

class DinoSegmenter(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = DinoBackboneLoRA()

        # embedding size for dinov2-large
        embed = 1024

        self.up1 = nn.ConvTranspose2d(embed,512,2,2)
        self.up2 = nn.ConvTranspose2d(512,256,2,2)
        self.up3 = nn.ConvTranspose2d(256,128,2,2)
        self.up4 = nn.ConvTranspose2d(128,64,2,2)

        self.out = nn.Conv2d(64,NUM_CLASSES,1)

    def forward(self,x):

        B,C,H,W = x.shape

        tokens = self.encoder(x)

        # automatically infer patch grid
        n_tokens = tokens.shape[1]
        grid = int(n_tokens ** 0.5)

        feat = tokens.permute(0,2,1).reshape(B,1024,grid,grid)

        x = F.relu(self.up1(feat))
        x = F.relu(self.up2(x))
        x = F.relu(self.up3(x))
        x = F.relu(self.up4(x))

        x = self.out(x)

        x = F.interpolate(x,size=(H,W),mode="bilinear",align_corners=False)

        return x
# =====================================================
# LOSS
# =====================================================

def dice_loss(pred,target):

    pred=torch.softmax(pred,dim=1)

    target_1hot=F.one_hot(target,NUM_CLASSES)\
        .permute(0,3,1,2).float()

    inter=(pred*target_1hot).sum((2,3))
    union=pred.sum((2,3))+target_1hot.sum((2,3))

    dice=(2*inter+1e-5)/(union+1e-5)

    return 1-dice.mean()


# =====================================================
# TRAIN
# =====================================================

def main():

    subjects=sorted(os.listdir(DATASET_ROOT))

    train_sub,test_sub=train_test_split(subjects,test_size=0.2,random_state=42)

    with open(TEST_SUBJECTS_FILE,"w") as f:
        for s in test_sub:
            f.write(s+"\n")

    train_sub,val_sub=train_test_split(train_sub,test_size=0.1,random_state=42)

    train_set=FeTADataset(train_sub,DATASET_ROOT)
    val_set=FeTADataset(val_sub,DATASET_ROOT)

    train_loader=DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    val_loader=DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    model=DinoSegmenter().to(device)

    total_params=sum(p.numel() for p in model.parameters())
    trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad)

    pd.DataFrame({
        "total_params":[total_params],
        "trainable_params":[trainable_params]
    }).to_csv(PARAM_CSV,index=False)

    ce=nn.CrossEntropyLoss()

    optimizer=torch.optim.AdamW(
        filter(lambda p:p.requires_grad,model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=5,
        factor=0.5
    )

    scaler=GradScaler()

    best_dice=0
    patience_counter=0

    history=[]

    for epoch in range(EPOCHS):

        model.train()

        train_loss=0

        for img,mask in tqdm(train_loader):

            img=img.to(device)
            mask=mask.to(device)

            optimizer.zero_grad()

            with autocast():

                pred=model(img)

                loss=ce(pred,mask)+dice_loss(pred,mask)

            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)

            scaler.step(optimizer)
            scaler.update()

            train_loss+=loss.item()

        model.eval()

        val_dice=0

        with torch.no_grad():

            for img,mask in val_loader:

                img=img.to(device)
                mask=mask.to(device)

                pred=model(img)

                val_dice+=1-dice_loss(pred,mask)

        val_dice/=len(val_loader)

        scheduler.step(val_dice)

        log(f"Epoch {epoch+1} | Train Loss {train_loss/len(train_loader)} | Val Dice {val_dice}")

        history.append({
            "epoch":epoch,
            "train_loss":train_loss/len(train_loader),
            "val_dice":val_dice.item(),
            "lr":optimizer.param_groups[0]["lr"]
        })

        if val_dice>best_dice:

            best_dice=val_dice

            torch.save(model.state_dict(),BEST_MODEL)

            patience_counter=0

        else:

            patience_counter+=1

        if patience_counter>=PATIENCE:

            log("Early stopping triggered")

            break

    pd.DataFrame(history).to_csv(METRICS_CSV,index=False)

    log("Training complete")


if __name__=="__main__":

    main()