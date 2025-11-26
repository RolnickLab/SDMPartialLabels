"""
random_forest_baseline.py

End-to-end Random Forest baseline using a PyTorch Dataset.
- Uses scikit-learn RandomForestClassifier
- Works with any PyTorch Dataset that returns (x, y)
"""
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from main import load_config
from src.dataloaders.splot_dataloader import sPlotDataModule


# ============================================================
# 1. Reproducibility
# ============================================================

def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. Helper: DataLoader -> NumPy arrays
# ============================================================

def dataloader_to_numpy(dataloader: DataLoader, device: str = "cpu"):
    """
    Collects all batches from a DataLoader into NumPy arrays.

    Assumes each batch is (x, y) or {"data": x, "targets": y}.
    """
    xs = []
    ys = []

    for batch in dataloader:
        # Support (x, y) or dict-style batches
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, y = batch
        elif isinstance(batch, dict):
            x = batch["data"]
            y = batch["targets"]
        else:
            raise ValueError("Unsupported batch format, expected (x, y) or dict with 'data' and 'targets'.")

        x = x.to(device)
        y = y.to(device)

        xs.append(x.detach().cpu())
        ys.append(y.detach().cpu())

    X = torch.cat(xs, dim=0).numpy()
    y = torch.cat(ys, dim=0).numpy()
    return X, y


# ============================================================
# 4. Train + Evaluate Random Forest
# ============================================================
NUM_CLASSES = 1000


def train_random_forest(
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        n_estimators: int = 2,
        max_depth=None,
        n_jobs: int = -1,
        random_state: int = 42,
        device: str = "cpu",
):
    """
    Trains a RandomForestClassifier on data from PyTorch DataLoaders.
    Works for standard single-label classification (y shape: (N,))
    """

    # ----- 4.1 Collect train data -----
    print("Collecting training data from DataLoader...")
    X_train, y_train = dataloader_to_numpy(train_loader, device=device)
    # y_train = y_train[:, :NUM_CLASSES]
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # ----- 4.2 Initialize RF -----
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=1
    )

    # ----- 4.3 Fit -----
    print("Fitting RandomForestClassifier...")
    rf.fit(X_train, y_train)

    # ----- 4.4 Validation evaluation -----
    if val_loader is not None:
        print("\n=== Validation Evaluation ===")
        X_val, y_val = dataloader_to_numpy(val_loader, device=device)
        y_val = y_val[:, :NUM_CLASSES]
        y_val_pred = rf.predict(X_val)
        val_auc = roc_auc_score(y_val, y_val_pred, multi_class="ovr")
        print(f"Validation accuracy: {val_auc:.4f}")
    else:
        val_auc = None

    # ----- 4.5 Test evaluation -----
    if test_loader is not None:
        print("\n=== Test Evaluation ===")
        X_test, y_test = dataloader_to_numpy(test_loader, device=device)
        y_test = y_test[:, :NUM_CLASSES]
        y_test_pred = rf.predict(X_test)
        test_auc = roc_auc_score(y_test, y_test_pred, multi_class="ovr")
        print(f"Test accuracy: {test_auc:.4f}")
    else:
        test_auc = None

    return rf, val_auc, test_auc


# ============================================================
# 5. Example usage
# ============================================================

def main():
    set_seed(1337)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ---- 5.1 Create datasets (REPLACE with your real datasets) ----
    config = load_config(os.path.join(os.getcwd(), "configs/satbird/config_mlp.yaml"))

    data_module_class = sPlotDataModule(config.data)
    data_module_class.setup()
    # ---- 5.2 DataLoaders ----
    train_loader = data_module_class.train_dataloader()
    val_loader = data_module_class.val_dataloader()
    test_loader = data_module_class.test_dataloader()

    # ---- 5.3 Train RF baseline ----
    rf_model, val_acc, test_acc = train_random_forest(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=1337,
        device=device,
    )

    print("\nDone. Random Forest baseline trained.")
    print(val_acc, test_acc)


if __name__ == "__main__":
    main()
