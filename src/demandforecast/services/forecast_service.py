"""Orchestrate sales history + covariates → ensemble prediction."""


class ForecastService:
    def __init__(self) -> None:
        # TODO: load Prophet/LSTM/TFT + EnsembleStacker per pharmacy_id
        pass

    async def forecast(
        self, pharmacy_id: str, sku_ids: list[str], horizon: int
    ) -> list[dict]:
        # TODO: pull history + covariates → run ensemble → format
        return []
