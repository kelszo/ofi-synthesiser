from typing import Tuple

import numpy as np
from sklearn.svm import LinearSVC
import optuna
from sklearn.metrics import roc_auc_score
import sdv.tabular


class CTGAN(sdv.tabular.CTGAN):
    def __init__(self, seed: int, debug: bool, *args, **kwargs):
        self.seed = seed
        if debug:
            kwargs["batch_size"] = 50
            kwargs["epochs"] = 1

        super(CTGAN, self).__init__(*args, **kwargs)
