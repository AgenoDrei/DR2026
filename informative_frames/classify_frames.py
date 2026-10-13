import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms, datasets
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_recall_curve, auc, confusion_matrix,
                             precision_score, recall_score)
from misc.utils import apply_clahe_cv2

DATA_DIR = 'data/informative_frames/frames_cropped'
MODEL_SAVE_PATH = 'informative_frames/models'
IMG_SIZE = 512
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
N_SPLITS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform: x = self.transform(x)
        return x, y

    def __len__(self): return len(self.subset)


def save_confusion_matrix(y_true, y_pred, epoch, run_name):
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted (1=Informative, 0=Non-Informative)')
    plt.ylabel('Actual')
    plt.title(f'CM - {run_name} - Epoch {epoch}')
    plt.savefig(os.path.join(MODEL_SAVE_PATH, f"{run_name}_cm_epoch_{epoch:03d}.png"))
    plt.close()


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    # 1. Prepare Dataset and Video Groups
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=None)

    # Extract filenames and video IDs for GroupKFold
    samples = full_dataset.samples
    filenames = [os.path.basename(s[0]) for s in samples]
    video_ids = [f.rsplit('_f', 1)[0] for f in filenames]
    labels = np.array([s[1] for s in samples])

    # 2. 5-Fold GroupKFold
    gkf = GroupKFold(n_splits=N_SPLITS)
    splits = list(gkf.split(filenames, labels, groups=video_ids))

    # Track split info across all folds
    all_split_info = []

    for fold, (train_indices, val_indices) in enumerate(splits):
        run_name = f"Selection_B4_Fold{fold}_{timestamp}"
        print(f"\n--- Starting Fold {fold}/{N_SPLITS - 1} ---")
        writer = SummaryWriter(log_dir=os.path.join("runs", run_name))

        # Save split info for this fold
        for idx in train_indices:
            all_split_info.append(
                {'filename': filenames[idx], 'video_id': video_ids[idx], 'label': labels[idx], 'fold': fold,
                 'split': 'train'})
        for idx in val_indices:
            all_split_info.append(
                {'filename': filenames[idx], 'video_id': video_ids[idx], 'label': labels[idx], 'fold': fold,
                 'split': 'val'})

        # 3. Address Class Imbalance (WeightedRandomSampler)
        train_labels = labels[train_indices]
        class_sample_count = np.array([len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in train_labels])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

        # 4. Transforms
        train_trans = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Lambda(apply_clahe_cv2),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),  # Added for medical images
            transforms.RandomRotation(90),  # Increased rotation significantly
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        val_trans = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Lambda(apply_clahe_cv2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        train_loader = DataLoader(TransformSubset(Subset(full_dataset, train_indices), train_trans),
                                  batch_size=BATCH_SIZE, sampler=sampler)
        val_loader = DataLoader(TransformSubset(Subset(full_dataset, val_indices), val_trans),
                                batch_size=BATCH_SIZE, shuffle=False)

        # 5. Model Setup
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        model.to(DEVICE)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        best_pr_auc = 0.0

        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            for imgs, lbls in tqdm(train_loader, desc=f"F{fold} Ep {epoch} Train"):
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE).float().view(-1, 1)
                optimizer.zero_grad()
                loss = criterion(model(imgs), lbls)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            t_loss = running_loss / len(train_loader)

            # Validation
            model.eval()
            y_true, y_probs, v_loss = [], [], 0.0
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE).float().view(-1, 1)
                    outputs = model(imgs)
                    v_loss += criterion(outputs, lbls).item()
                    y_probs.extend(torch.sigmoid(outputs).cpu().numpy())
                    y_true.extend(lbls.cpu().numpy())

            v_loss /= len(val_loader)
            y_true, y_probs = np.array(y_true), np.array(y_probs)
            y_pred = (y_probs > 0.5).astype(float)

            # --- Full Metrics Calculation ---
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.5
            p_curve, r_curve, _ = precision_recall_curve(y_true, y_probs)
            pr_auc = auc(r_curve, p_curve)

            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()

            print(f"Fold {fold} Ep {epoch} | PR-AUC: {pr_auc:.4f} | F1: {f1:.4f} | Loss: {v_loss:.4f}")

            # TensorBoard Logging
            writer.add_scalar('Loss/train', t_loss, epoch)
            writer.add_scalar('Loss/val', v_loss, epoch)
            writer.add_scalar('Metrics/Accuracy', acc, epoch)
            writer.add_scalar('Metrics/F1', f1, epoch)
            writer.add_scalar('Metrics/Precision', prec, epoch)
            writer.add_scalar('Metrics/Recall', rec, epoch)
            writer.add_scalar('Metrics/ROC_AUC', roc_auc, epoch)
            writer.add_scalar('Metrics/PR_AUC', pr_auc, epoch)
            writer.add_scalar('LR', current_lr, epoch)

            if pr_auc > best_pr_auc:
                best_pr_auc = pr_auc
                torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, f"best_{run_name}.pth"))
                save_confusion_matrix(y_true, y_pred, epoch, run_name)

        writer.close()

    # Save final aggregate split info
    pd.DataFrame(all_split_info).to_csv(f'informative_frames/split_info_5fold_{timestamp}.csv', index=False)


if __name__ == "__main__":
    main()