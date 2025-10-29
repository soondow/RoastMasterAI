from abc import ABC, abstractmethod
import pandas as pd

class ISampler(ABC):
    name: str = "base"
    @abstractmethod
    def fit_resample(self, X: pd.DataFrame, y: pd.Series, **params) -> tuple[pd.DataFrame, pd.Series] | tuple[None, None]:
        raise NotImplementedError
