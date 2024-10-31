from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class Hyperparameters:
    learning_rate: list[float] = field(default=[])
    reg_strength: list[float] = field(default=[])  # weight_decay
    epochs: list[int] = field(default=[])
    batch_size: list[int] = field(default=[])
    optimizer_params: dict[str, Any] = field(default={})

    def to_df(self) -> pd.DataFrame:
        df = pd.DataFrame(
            [
                self.learning_rate,
                self.reg_strength,
                self.epochs,
                self.batch_size,
                self.optimizer_params,
            ],
            columns=[
                "Learning Rate",
                "Regularization Strength",
                "Epochs",
                "Batch Size",
                "Optimizer parameters",
            ],
        )
        return df
