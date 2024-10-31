from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Hyperparameters:
    learning_rate: list[float] = field(default=[])
    weight_decay: list[float] = field(default=[])  # regularization_strengths
    epochs: list[int] = field(default=[])
    batch_size: list[int] = field(default=[])

    def to_df(self) -> pd.DataFrame:
        df = pd.DataFrame(
            [
                self.learning_rate,
                self.weight_decay,
                self.epochs,
                self.batch_size,
            ],
            columns=[
                "Learning Rate",
                "Weight Decay",
                "Epochs",
                "Batch Size",
            ],
        )
        return df
