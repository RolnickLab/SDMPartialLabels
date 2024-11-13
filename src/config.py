from typing import Literal, Optional

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name: Optional[str] = Field(..., description="Name of the model")
    input_dim: Optional[int] = Field(..., description="Input dimension of the data")
    hidden_dim: Optional[int] = Field(..., description="Hidden dimension of the MLP")
    output_dim: Optional[int] = Field(..., description="Output dimension of the MLP")
    backbone: Optional[str] = Field(..., description="Name of the backbone")


class TrainingConfig(BaseModel):
    seed: Optional[int] = Field(None, description="Random seed")
    learning_rate: Optional[float] = Field(
        ..., description="Learning rate for the optimizer"
    )
    max_epochs: Optional[int] = Field(..., description="Number of epochs to train")
    accelerator: Optional[str] = Field(
        ..., description="Accelerator for training: gpu, cpu, or auto"
    )
    devices: Optional[int] = Field(
        ..., description="Accelerator for training: gpu, cpu, or auto"
    )


class PartialLabels(BaseModel):
    use: bool = Field(..., description="Training with Partial labels or not")
    predict_family_of_species: int = Field(
        ..., description="what family of species to predict during testing"
    )
    train_known_ratio: Optional[float] = Field(
        0, description="Maximum known labels ratio for training"
    )
    eval_known_ratio: Optional[float] = Field(
        0, description="Maximum known labels ratio for eval"
    )


class DataPathConfig(BaseModel):
    dataloader_to_use: Optional[str] = Field(None, description="name of Data loader to use")
    base: Optional[str] = Field(..., description="Base path for data")
    train: str = Field(..., description="Path to training indices")
    validation: str = Field(..., description="Path to validation indices")
    test: str = Field(..., description="Path to test indices")
    targets: str = Field(..., description="Path to targets")
    worldclim_data_path: str = Field(..., description="Path to WorldClim data")
    soilgrid_data_path: str = Field(..., description="Path to SoilGrid data")
    species_occurrences_threshold: int = Field(
        ..., description="Species occurrences threshold"
    )
    batch_size: Optional[int] = Field(..., description="Batch size for training")
    species_list: str = Field(..., description="Path to list of species names")
    env_columns: list[str] = Field(..., description="List of column names of environment features")

    partial_labels: PartialLabels = Field(..., description="Partial labels")


class LoggingConfig(BaseModel):
    project_name: str = Field(..., description="Project name")
    experiment_name: str = Field(..., description="Experiment name")
    experiment_key: Optional[str] = Field(..., description="Experiment key")
    checkpoint_path: Optional[str] = Field(..., description="Checkpoint path")
    checkpoint_name: Optional[str] = Field(..., description="Checkpoint name")


class Config(BaseModel):
    mode: Literal["train", "test"] = Field(..., description="Mode of operation")
    dataset_name: str = Field(..., description="Dataset name")
    model: ModelConfig = Field(..., description="Model configuration")
    data: DataPathConfig = Field(..., description="Data paths configuration")
    training: TrainingConfig = Field(..., description="Training configuration")
    logger: LoggingConfig = Field(..., description="Logging configuration")