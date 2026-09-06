"""Current weather via the free Open-Meteo API (no API key required)."""

from __future__ import annotations

from typing import Type

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

_GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

_WMO = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog", 51: "light drizzle", 53: "drizzle",
    55: "dense drizzle", 61: "slight rain", 63: "rain", 65: "heavy rain",
    71: "slight snow", 73: "snow", 75: "heavy snow", 80: "rain showers",
    81: "rain showers", 82: "violent rain showers", 95: "thunderstorm",
    96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}


class WeatherInput(BaseModel):
    location: str = Field(..., description="City or place name, e.g. 'Hyderabad'")


class WeatherTool(BaseTool):
    name: str = "Weather"
    description: str = (
        "Returns current weather (temperature, conditions, wind, humidity) for a "
        "city or place name. Uses the free Open-Meteo service."
    )
    args_schema: Type[BaseModel] = WeatherInput

    def _run(self, location: str) -> str:
        try:
            geo = requests.get(
                _GEO_URL,
                params={"name": location, "count": 1},
                timeout=15,
            ).json()
            if not geo.get("results"):
                return f"Could not find a location called '{location}'."
            place = geo["results"][0]
            lat, lon = place["latitude"], place["longitude"]
            label = ", ".join(
                p for p in (place.get("name"), place.get("country")) if p
            )

            data = requests.get(
                _FORECAST_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,apparent_temperature,"
                    "weather_code,wind_speed_10m",
                },
                timeout=15,
            ).json()["current"]

            code = data.get("weather_code")
            return (
                f"Weather in {label}:\n"
                f"- Condition: {_WMO.get(code, 'unknown')}\n"
                f"- Temperature: {data['temperature_2m']}°C "
                f"(feels like {data['apparent_temperature']}°C)\n"
                f"- Humidity: {data['relative_humidity_2m']}%\n"
                f"- Wind: {data['wind_speed_10m']} km/h"
            )
        except Exception as exc:  # noqa: BLE001
            return f"Weather lookup error: {exc}"
