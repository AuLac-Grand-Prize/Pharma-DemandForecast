"""Prophet wrapper với Vietnamese holidays."""


class ProphetModel:
    def __init__(self) -> None:
        self.model = None  # TODO: prophet.Prophet(holidays=vietnamese_holidays())

    def fit(self, df) -> None:  # noqa: ANN001
        pass

    def predict(self, periods: int):  # noqa: ANN201
        pass
