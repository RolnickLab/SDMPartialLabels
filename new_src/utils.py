import torch


def multi_label_accuracy(y_pred, y_true, threshold=0.5):
    y_pred_binary = (y_pred > threshold).float()

    correct_per_label = (y_pred_binary == y_true).sum(dim=0).float()
    accuracy_per_label = correct_per_label / y_true.size(0)  # divide by batch size

    accuracy = accuracy_per_label.mean().item()

    return accuracy
