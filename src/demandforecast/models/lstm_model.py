"""LSTM forecasting model (Darts wrapper)."""


class LSTMModel:
    def __init__(self, input_chunk_length: int = 30, output_chunk_length: int = 30) -> None:
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        # TODO: darts.models.RNNModel(model="LSTM", ...)

    def fit(self, series) -> None:
        pass

    def predict(self, n: int):
        pass
