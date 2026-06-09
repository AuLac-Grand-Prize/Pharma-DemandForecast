"""Prophet wrapper với Vietnamese holidays."""


class ProphetModel:
    def __init__(self) -> None:
        self.model = None  # TODO: prophet.Prophet(holidays=vietnamese_holidays())

    def fit(self, df) -> None:
        pass

    def predict(self, periods: int):
        pass
