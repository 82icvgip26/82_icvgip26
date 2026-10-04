import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
from transformers import AutoModel
import medpy.metric.binary as mmb

# =========================
# PATHS (EDIT ONLY THIS)
# =========================
DATASET_ROOT = "/home/user1/chintan/sarthak/feta_2.4"
MODEL_PATH = "/home/user1/chintan/sarthak/dinov2/dinov2lora/dinov2withlora/outputs/best_model.pth"
TEST_SUBJECTS_FILE = "/home/user1/chintan/sarthak/dinov2/dinov2lora/dinov2withlora/outputs/test_subjects.txt"

BACKBONE = "facebook/dinov2-large"

OUTPUT_DIR = "./eval_output"
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")

os.makedirs(PRED_DIR, exist_ok=True)

NUM_CLASSES = 8
IMAGE_SIZE = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# LOAD SUBJECTS
# =========================
def load_test_subjects():
    subjects = []
    with open(TEST_SUBJECTS_FILE) as f:
        for line in f:
            s = line.strip()
            if os.path.isdir(os.path.join(DATASET_ROOT, s)):
                subjects.append(s)
    return subjects


# =========================
# LORA
# =========================
class LoRALinear(nn.Module):

    def __init__(self, linear, rank=4):
        super().__init__()
        self.linear = linear

        self.A = nn.Parameter(torch.randn(rank, linear.in_features) * 0.01)
        self.B = nn.Parameter(torch.randn(linear.out_features, rank) * 0.01)

        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.linear(x) + (x @ self.A.T) @ self.B.T


# =========================
# BACKBONE
# =========================
class DinoBackboneLoRA(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = AutoModel.from_pretrained(BACKBONE)

        for p in self.model.parameters():
            p.requires_grad = False

        for block in self.model.encoder.layer:
            block.attention.attention.query = LoRALinear(block.attention.attention.query)
            block.attention.attention.value = LoRALinear(block.attention.attention.value)

    def forward(self, x):
        out = self.model(pixel_values=x)
        return out.last_hidden_state[:, 1:, :]


# =========================
# SEGMENTER
# =========================
class DinoSegmenter(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = DinoBackboneLoRA()

        self.up1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, 2)

        self.out = nn.Conv2d(64, NUM_CLASSES, 1)

    def forward(self, x):

        B, C, H, W = x.shape

        tokens = self.encoder(x)
        grid = int(tokens.shape[1] ** 0.5)

        feat = tokens.permute(0, 2, 1).reshape(B, 1024, grid, grid)

        x = F.relu(self.up1(feat))
        x = F.relu(self.up2(x))
        x = F.relu(self.up3(x))
        x = F.relu(self.up4(x))

        x = self.out(x)

        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)


# =========================
# LOAD MODEL
# =========================
model = DinoSegmenter().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


# =========================
# PREDICT
# =========================
def predict_subject(subject):

    anat = os.path.join(DATASET_ROOT, subject, "anat")

    img_path = os.path.join(anat, f"{subject}_rec-irtk_T2w.nii.gz")
    lbl_path = os.path.join(anat, f"{subject}_rec-irtk_dseg.nii.gz")

    if not os.path.exists(img_path):
        print("Missing:", img_path)
        return None

    img_nii = nib.load(img_path)
    lbl_nii = nib.load(lbl_path)

    img_vol = img_nii.get_fdata().astype("float32")
    gt_vol = lbl_nii.get_fdata().astype("int64")

    img_vol = (img_vol - img_vol.min()) / (img_vol.max() - img_vol.min() + 1e-6)

    preds = []

    for z in range(img_vol.shape[2]):

        img = torch.tensor(img_vol[:, :, z]).unsqueeze(0).repeat(3,1,1)
        img = F.interpolate(img.unsqueeze(0), size=(256,256), mode="bilinear", align_corners=False)
        img = img.to(device)

        with torch.no_grad():
            pred = torch.argmax(model(img), dim=1).squeeze().cpu().numpy()

        pred = F.interpolate(
            torch.tensor(pred).unsqueeze(0).unsqueeze(0).float(),
            size=img_vol.shape[:2],
            mode="nearest"
        ).squeeze().numpy()

        preds.append(pred)

    pred_vol = np.stack(preds, axis=2)

    nib.save(
        nib.Nifti1Image(pred_vol.astype(np.uint8), img_nii.affine),
        os.path.join(PRED_DIR, f"{subject}_pred.nii.gz")
    )

    return pred_vol, gt_vol


# =========================
# METRICS
# =========================
def compute_metrics(pred, gt, subject):

    rows = []

    for cls in range(NUM_CLASSES):

        pred_bin = (pred == cls)
        gt_bin = (gt == cls)

        if gt_bin.sum() == 0 and pred_bin.sum() == 0:
            dice, vs, hd95, assd = 1,1,0,0
        elif gt_bin.sum() == 0 or pred_bin.sum() == 0:
            dice, vs, hd95, assd = 0,0,np.nan,np.nan
        else:
            dice = mmb.dc(pred_bin, gt_bin)

            Vp = pred_bin.sum()
            Vg = gt_bin.sum()
            vs = 1 - abs(Vp - Vg) / (Vp + Vg + 1e-8)

            try:
                hd95 = mmb.hd95(pred_bin, gt_bin)
                assd = mmb.assd(pred_bin, gt_bin)
            except:
                hd95, assd = np.nan, np.nan

        rows.append({
            "Subject": subject,
            "Class": cls,
            "Dice": dice,
            "VS": vs,
            "HD95": hd95,
            "ASSD": assd
        })

    return rows


# =========================
# MAIN
# =========================
subjects = load_test_subjects()

all_rows = []

for s in tqdm(subjects):

    result = predict_subject(s)

    if result is None:
        continue

    pred, gt = result
    all_rows.extend(compute_metrics(pred, gt, s))


df = pd.DataFrame(all_rows)
df.to_csv(os.path.join(OUTPUT_DIR, "results.csv"), index=False)

summary = df.groupby("Class").mean(numeric_only=True)
summary.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"))


# =========================
# PLOTS
# =========================
plt.figure()
summary["Dice"].plot()
plt.title("Dice per class")
plt.savefig(os.path.join(OUTPUT_DIR, "dice_per_class.png"))

plt.figure()
summary["HD95"].plot()
plt.title("HD95 per class")
plt.savefig(os.path.join(OUTPUT_DIR, "hd95_per_class.png"))

print("Evaluation complete. Finally behaving like it should.")