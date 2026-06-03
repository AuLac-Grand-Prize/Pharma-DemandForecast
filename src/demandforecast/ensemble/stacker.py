"""Ensemble stacker — learn per-SKU weights for Prophet/LSTM/TFT predictions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EnsembleWeights:
    prophet: float
    lstm: float
    tft: float

    def normalize(self) -> "EnsembleWeights":
        s = self.prophet + self.lstm + self.tft
        if s == 0:
            return EnsembleWeights(1 / 3, 1 / 3, 1 / 3)
        return EnsembleWeights(self.prophet / s, self.lstm / s, self.tft / s)


class EnsembleStacker:
    def __init__(self, weights: EnsembleWeights | None = None) -> None:
        self.weights = (weights or EnsembleWeights(1 / 3, 1 / 3, 1 / 3)).normalize()

    def combine(self, prophet_pred: float, lstm_pred: float, tft_pred: float) -> float:
        w = self.weights
        return w.prophet * prophet_pred + w.lstm * lstm_pred + w.tft * tft_pred
