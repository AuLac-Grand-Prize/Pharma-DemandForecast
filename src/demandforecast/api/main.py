from fastapi import FastAPI

from demandforecast import __version__
from demandforecast.api.routes import forecast, health, reorder

app = FastAPI(
    title="DemandForecast AI",
    description="Dự báo nhu cầu thuốc cho nhà thuốc Việt Nam — engine của PharmLink AI.",
    version=__version__,
)

app.include_router(health.router, tags=["health"])
app.include_router(forecast.router, prefix="/v1", tags=["forecast"])
app.include_router(reorder.router, prefix="/v1", tags=["reorder"])
