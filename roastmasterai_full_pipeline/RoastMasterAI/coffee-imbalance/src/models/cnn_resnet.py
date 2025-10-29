import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass

class SimpleCNN(nn.Module):
    def __init__(self, in_ch=3, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(32, num_classes)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)

@dataclass
class TrainCfg:
    lr: float = 1e-3
    epochs: int = 50
    batch: int = 32
    patience: int = 5
    seed: int = 42
    pos_weight: float | None = None

def train_cnn(model, train_loader, val_loader, cfg: TrainCfg, device="cpu"):
    torch.manual_seed(cfg.seed)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    pos_w = None if cfg.pos_weight is None else torch.tensor([cfg.pos_weight], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w) if model.fc.out_features==1 else nn.CrossEntropyLoss(weight=None)
    best_loss = float("inf"); best_state = None; no_improve = 0
    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            if logits.shape[1]==1: yb = yb.float().unsqueeze(1)
            loss = crit(logits, yb if logits.shape[1]>1 else (yb))
            opt.zero_grad(); loss.backward(); opt.step()
        # val
        model.eval(); vloss=0.0; n=0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                if logits.shape[1]==1: yb = yb.float().unsqueeze(1)
                loss = crit(logits, yb if logits.shape[1]>1 else (yb))
                vloss += float(loss.item())*xb.size(0); n+=xb.size(0)
        vloss/=max(1,n)
        if vloss < best_loss-1e-6:
            best_loss = vloss; best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; no_improve=0
        else:
            no_improve+=1
            if no_improve>=cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model
