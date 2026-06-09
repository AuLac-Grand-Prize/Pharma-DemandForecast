"""Temporal Fusion Transformer wrapper (Darts)."""


class TFTModel:
    def __init__(self, input_chunk_length: int = 60, output_chunk_length: int = 30) -> None:
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        # TODO: darts.models.TFTModel(...)

    def fit(self, series, past_covariates=None, future_covariates=None) -> None:
        pass

    def predict(self, n: int, past_covariates=None, future_covariates=None):
        pass
