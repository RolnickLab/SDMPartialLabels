import torch


def compute_sampling_weights(
    targets: torch.Tensor, per_taxa_species_count: list[int]
) -> torch.Tensor:
    unique_classes = torch.empty(targets.shape[0], dtype=torch.long)
    missing_species_counts = (targets == -2).sum(dim=1)

    # colocated data
    unique_classes[missing_species_counts == 0] = 0
    # butterfly-only data
    unique_classes[missing_species_counts == per_taxa_species_count[1]] = 1
    # bird-only data
    unique_classes[missing_species_counts == per_taxa_species_count[0]] = 2

    def calculate_class_weights(labels):
        labels = labels.long()
        class_counts = torch.bincount(labels, minlength=3)
        class_weights = 100000.0 / (class_counts + 1e-6)

        sample_weights = class_weights[labels]
        return sample_weights

    sample_weights = calculate_class_weights(unique_classes)

    return sample_weights
