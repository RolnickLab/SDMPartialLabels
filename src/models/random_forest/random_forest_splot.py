"""
random_forest_baseline.py

End-to-end Random Forest baseline using a PyTorch Dataset.
- Uses scikit-learn RandomForestClassifier
- Works with any PyTorch Dataset that returns (x, y)
"""
import csv
import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score
from torchmetrics.classification import MultilabelAUROC

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

def trees_masking(index, config):
    targets = np.load(os.path.join(config.base, config.targets))
    species_df = pd.read_csv(os.path.join(config.base, config.species_list))

    species_indices = np.where(
        targets.sum(axis=0) >= config.species_occurrences_threshold
    )[0]

    species_df = species_df.loc[species_indices]
    species_df = species_df.reset_index(drop=True)

    # 0: not trees, 1 : trees
    indices_to_predict = np.where(
        species_df["isTree"] == index
    )[0]

    return indices_to_predict

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
    # y_train = y_train[:, :NUM_CLASSES]
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # ----- 4.2 Initialize RF -----
    rf = RandomForestRegressor(
        n_estimators=n_estimators,  # don’t go crazy with 500+ here
        max_depth=20,  # limit depth
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state,
    )

    # ----- 4.3 Fit -----
    print("Fitting RandomForestClassifier...")
    rf.fit(X_train, y_train)

    # ----- 4.4 Validation evaluation -----
    print("\n=== Validation Evaluation ===")
    X_val, y_val = dataloader_to_numpy(val_loader, device=device)
    probs_val = rf.predict(X_val)
    print("Example probs shape:", np.unique(probs_val[0], return_counts=True))
    y_val_valid = y_val
    # probs_val, y_val_valid = rf_predict_proba_matrix_and_valid_outputs(
    #     rf, X_val, y_val
    # )

    if probs_val is None:
        print("Validation AUROC: no valid outputs (all single-class). Setting to NaN.")
        val_auc = float("nan")
    else:
        tm = MultilabelAUROC(num_labels=probs_val.shape[1], average="macro")
        val_auc = tm(
            torch.tensor(probs_val, dtype=torch.float32),
            torch.tensor(y_val_valid, dtype=torch.int64),
        ).item()
        print(f"Validation AUROC (macro over {probs_val.shape[1]} valid outputs): {val_auc:.4f}")

    print(f"Validation accuracy: {val_auc:.4f}")

    test_AUC = []
    # ----- 4.5 Test evaluation -----
    print("\n=== Test Evaluation ===")
    X_test, y_test = dataloader_to_numpy(test_loader, device=device)
    # probs_test, y_test_valid= rf_predict_proba_matrix_and_valid_outputs(rf, X_test, y_test)
    probs_test = rf.predict(X_test)
    y_test_valid = y_test

    tm = MultilabelAUROC(num_labels=probs_test.shape[1], average="macro")
    test_auc = tm(
        torch.tensor(probs_test, dtype=torch.float32),
        torch.tensor(y_test_valid, dtype=torch.int64),
    ).item()
    print(f"Test AUROC (macro over {probs_test.shape[1]} valid outputs): {val_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    test_AUC.append(test_auc)

    # next test on sub-species we have such as
    indices = [0, 1]
    for ind in indices:
        print("=====species index===:: ", ind)
        species_indices_to_eval = trees_masking(index=ind, config=config)
        predictions = probs_test[:, species_indices_to_eval]
        targets = y_test_valid[:, species_indices_to_eval]

        tm = MultilabelAUROC(num_labels=predictions.shape[1], average="macro")
        test_auc_ = tm(
            torch.tensor(predictions, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.int64),
        ).item()
        print(f"Test AUC: {test_auc_:.4f}")
        test_AUC.append(test_auc_)

    return rf, test_AUC


# ============================================================
# 5. Example usage
# ============================================================

def main():
    seed = 1337
    run_id = 1
    global_seed = (run_id * (seed + (run_id - 1))) % (2 ** 31 - 1)

    set_seed(global_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ---- 5.1 Create datasets (REPLACE with your real datasets) ----
    config = load_config(os.path.join(os.getcwd(), "configs/splot/config_mlp.yaml"))

    data_module_class = sPlotDataModule(config.data)
    data_module_class.setup()
    # ---- 5.2 DataLoaders ----
    train_loader = data_module_class.train_dataloader()
    val_loader = data_module_class.val_dataloader()
    test_loader = data_module_class.test_dataloader()

    # ---- 5.3 Train RF baseline ----
    rf_model, test_auc = train_random_forest(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config.data,
        n_estimators=60,
        max_depth=None,
        n_jobs=-1,
        random_state=1337,
        device="mps",
    )

    with open(f"RF_splot_auc_results.csv", "a", newline="") as f:
        writer = csv.writer(f)
        # writer.writerow(["All_species", "non-tree", "tree"])
        writer.writerow(test_auc)

    print("\nDone. Random Forest baseline trained.")

if __name__ == "__main__":
    main()
