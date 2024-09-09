from dataclasses import dataclass, field

import pandas as pd
from entities.hyperparameters import Hyperparameters


@dataclass
class TrainingMetrics:
    hyperparameters: Hyperparameters = field()
    loss_history: list[float] = field(default=[])
    train_acc_history: list[float] = field(default=[])
    val_acc_history: list[float] = field(default=[])

    def to_df(self) -> pd.DataFrame:
        hyperparameters_df = self.hyperparameters.to_df()
        df = pd.DataFrame(
            [
                hyperparameters_df,
                self.loss_history,
                self.train_acc_history,
                self.val_acc_history,
            ],
            columns=[
                hyperparameters_df.columns,
                "Loss",
                "Train Set Accuracy",
                "Validation Set Accuracy",
            ],
        )
        return df
