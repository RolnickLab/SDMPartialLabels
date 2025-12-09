"""
random_forest_satbird.py

End-to-end Random Forest baseline using a PyTorch Dataset for SatBird
- Uses scikit-learn RandomForestRegressor
- Works with any PyTorch Dataset that returns (x, y)
"""
import csv
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestRegressor

from src.dataloaders.dataloader import SDMDataModule
from src.metrics import CustomTopK, MaskedMAE
from src.utils import load_opts, eval_species_split


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


def compute_masked_mae_from_numpy(y_true_np, y_pred_np, mask_np=None):
    """
    y_true_np: (N, ...) ground truth
    y_pred_np: (N, ...) predictions
    mask_np:   (N, ...) optional boolean or {0,1} mask; if None, all elements used
    """
    y_true = torch.from_numpy(y_true_np).float()
    preds = torch.from_numpy(y_pred_np).float()

    if mask_np is not None:
        mask = torch.from_numpy(mask_np.astype(bool))
    else:
        mask = None

    metric = MaskedMAE()
    metric.update(target=y_true, preds=preds, mask=mask)
    return metric.compute().item()


def compute_custom_topk_from_numpy(y_true_np, y_pred_scores_np):
    """
    y_true_np: (N,) with class indices OR (N, C) with multi-hot targets.
    y_pred_scores_np: (N, C) with per-class scores/probabilities.
    """
    y_true = torch.from_numpy(y_true_np)
    preds = torch.from_numpy(y_pred_scores_np).float()

    # If y is 1D (class indices), convert to one-hot so it matches CustomTopK expectation.
    if y_true.ndim == 1:
        num_classes = preds.shape[1]
        y_onehot = torch.zeros((y_true.shape[0], num_classes), dtype=torch.float32)
        y_onehot[torch.arange(y_true.shape[0]), y_true.long()] = 1.0
    else:
        # assume already multi-hot, just cast
        y_onehot = y_true.float()

    metric = CustomTopK()
    metric.update(target=y_onehot, preds=preds)
    return metric.compute().item()

# ============================================================
# 4. Train + Evaluate Random Forest
# ============================================================
# NUM_CLASSES = 1000


def train_random_forest(
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: dict,
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
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # ----- 4.2 Initialize RF -----
    rf = RandomForestRegressor(
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

        # For RandomForest, use predict_proba to get per-class scores
        # predict_proba returns (N, C) for single-output multi-class
        y_val_proba = rf.predict(X_val)
        # In binary classification, predict_proba returns (N, 2)
        # that's still fine for CustomTopK

        val_custom_k = compute_custom_topk_from_numpy(y_val, y_val_proba)
        print(f"Validation TopK: {val_custom_k:.4f}")
        # Masked MAE (no mask here – use mask_np if you have one)
        val_mae = compute_masked_mae_from_numpy(y_val, y_val_proba)

        print(f"Validation MAE:  {val_mae:.4f}")

    test_topk = []
    test_mae = []
    # ----- 4.5 Test evaluation -----
    print("\n=== Test Evaluation ===")
    X_test, y_test = dataloader_to_numpy(test_loader, device=device)
    y_test_proba = rf.predict(X_test)

    test_topk_ = compute_custom_topk_from_numpy(y_test, y_test_proba)
    test_mae_ = compute_masked_mae_from_numpy(y_test, y_test_proba)

    print(f"Test TopK: {test_topk_:.4f}")
    print(f"Test MAE:  {test_mae_:.4f}")
    test_topk.append(test_topk_)
    test_mae.append(test_mae_)
    # next test on sub-species we have such as
    indices = [0, 1]
    for ind in indices:
        print("=====species index===:: ", ind)
        base_data_folder = os.path.join(
            config.data.files.base,
            config.data.files.satbird_species_indices_path,
        )
        species_indices_to_eval = eval_species_split(index=ind, base_data_folder=base_data_folder)
        predictions = y_test_proba[:, species_indices_to_eval]
        targets = y_test[:, species_indices_to_eval]

        test_topk_ = compute_custom_topk_from_numpy(targets, predictions)
        test_mae_ = compute_masked_mae_from_numpy(targets, predictions)

        print(f"Test TopK: {test_topk_:.4f}")
        print(f"Test MAE:  {test_mae_:.4f}")
        test_topk.append(test_topk_)
        test_mae.append(test_mae_)

    return rf, test_topk, test_mae


def main():
    seed = 1337
    run_id = 1
    global_seed = (run_id * (seed + (run_id - 1))) % (2**31 - 1)

    set_seed(global_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ---- 5.1 Create datasets (REPLACE with your real datasets) ----
    default_config = os.path.join(os.getcwd(), "configs/defaults.yaml")

    config = load_opts(os.path.join(os.getcwd(), "configs/satbird/config_mlp.yaml"), default=default_config)
    data_module_class = SDMDataModule(config)
    data_module_class.setup()

    # ---- 5.2 DataLoaders ----
    train_loader = data_module_class.train_dataloader()
    val_loader = data_module_class.val_dataloader()
    test_loader = data_module_class.test_dataloader()

    # ---- 5.3 Train RF baseline ----
    rf_model, test_topk, test_mae = train_random_forest(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config = config,
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=global_seed,
        device=device,
    )
    with open(f"RF_satbird_topk_results.csv", "a", newline="") as f:
        writer = csv.writer(f)
        # writer.writerow(["All_species", "non-songbird", "songbird"])
        writer.writerow(test_topk)

    with open(f"RF_satbird_mae_results.csv", "a", newline="") as f:
        writer = csv.writer(f)
        # writer.writerow(["All_species", "non-songbird", "songbird"])
        writer.writerow(test_mae)

    print("\nDone. Random Forest baseline trained.")


if __name__ == "__main__":
    main()
