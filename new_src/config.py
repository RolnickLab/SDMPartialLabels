from typing import Literal, Optional

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name: Optional[str] = Field(..., description="Name of the model")
    input_dim: Optional[int] = Field(..., description="Input dimension of the data")
    hidden_dim: Optional[int] = Field(..., description="Hidden dimension of the MLP")
    output_dim: Optional[int] = Field(..., description="Output dimension of the MLP")


class TrainingConfig(BaseModel):
    batch_size: Optional[int] = Field(..., description="Batch size for training")
    learning_rate: Optional[float] = Field(..., description="Learning rate for the optimizer")
    max_epochs: Optional[int] = Field(..., description="Number of epochs to train")

class DataPathConfig(BaseModel):
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


class LoggingConfig(BaseModel):
    project_name: str = Field(..., description="Project name")
    experiment_name: str = Field(..., description="Experiment name")
    experiment_key: Optional[str] = Field(..., description="Experiment key")


class Config(BaseModel):
    mode: Literal["train", "test"] = Field(..., description="Mode of operation")
    model: ModelConfig = Field(..., description="Model configuration")
    data: DataPathConfig = Field(..., description="Data paths configuration")
    training: TrainingConfig = Field(..., description="Training configuration")
    logger: LoggingConfig = Field(..., description="Logging configuration")