import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score, matthews_corrcoef, accuracy_score, precision_score, recall_score
import pandas as pd

class LazyGAFDataset(Dataset):
    def __init__(self, paths, y=None, augment=False):
        self.paths, self.y = paths, y
        tfms = []
        if augment:
            tfms += [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.2)]
        tfms += [transforms.ToTensor()]
        self.tfm = transforms.Compose(tfms)
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        arr = np.load(self.paths[i])     # (3,H,W)
        img = arr.transpose(1,2,0)       # (H,W,3)
        x = self.tfm(img)
        return (x, int(self.y[i])) if self.y is not None else x

def build_resnet18(num_classes=2, freeze_until="layer3"):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    freeze = True
    for n, p in m.named_parameters():
        if freeze and n.startswith(freeze_until):
            freeze = False
        p.requires_grad = not freeze
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def _eval_binary(y_true, prob, pred):
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, prob) if prob is not None else np.nan,
        "PR_AUC": average_precision_score(y_true, prob) if prob is not None else np.nan,
        "MCC": matthews_corrcoef(y_true, pred),
    }

def run_cnn_cv(X_paths, y, epochs=12, bs=16, lr=1e-3, n_splits=5, seed=42, device=None, augment=True):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    recs = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(skf.split(X_paths, y), 1):
        Xtr = [X_paths[i] for i in tr]
        Xte = [X_paths[i] for i in te]
        ytr, yte = y.iloc[tr].values, y.iloc[te].values

        ds_tr = LazyGAFDataset(Xtr, ytr, augment=augment)
        ds_te = LazyGAFDataset(Xte, yte, augment=False)
        dl_tr = DataLoader(ds_tr, bs, shuffle=True)
        dl_te = DataLoader(ds_te, bs)

        model = build_resnet18(num_classes=2, freeze_until="layer3").to(device)
        criterion = nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        for ep in range(epochs):
            model.train()
            for xb, yb in dl_tr:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                opt.step()
            sch.step()

        model.eval()
        all_prob, all_pred, all_true = [], [], []
        with torch.no_grad():
            for xb, yb in dl_te:
                logits = model(xb.to(device))
                p = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
                yhat = (p >= 0.5).astype(int)
                all_prob.append(p); all_pred.append(yhat); all_true.append(yb.numpy())
        prob = np.concatenate(all_prob); pred = np.concatenate(all_pred); true = np.concatenate(all_true)
        m = _eval_binary(true, prob, pred)
        m.update({"fold":fold, "model":"ResNet18-GAF"})
        recs.append(m)
    return pd.DataFrame(recs)
