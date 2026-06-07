from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import urllib3


MOSCOW_CENTER = (55.7558, 37.6173)

MOSCOW_ZONE_POINTS = [
    {"zone_name": "Tverskoy", "lat": 55.7652, "lon": 37.6056},
    {"zone_name": "Arbat", "lat": 55.7522, "lon": 37.5907},
    {"zone_name": "Moscow City", "lat": 55.7473, "lon": 37.5398},
    {"zone_name": "Sokolniki", "lat": 55.7903, "lon": 37.6790},
    {"zone_name": "Tagansky", "lat": 55.7424, "lon": 37.6536},
    {"zone_name": "Khamovniki", "lat": 55.7272, "lon": 37.5671},
    {"zone_name": "VDNH", "lat": 55.8240, "lon": 37.6385},
    {"zone_name": "Sheremetyevo", "lat": 55.9726, "lon": 37.4146},
    {"zone_name": "Vnukovo", "lat": 55.5991, "lon": 37.2732},
    {"zone_name": "Domodedovo", "lat": 55.4146, "lon": 37.8990},
    {"zone_name": "Yasenevo", "lat": 55.6046, "lon": 37.5360},
    {"zone_name": "Kuzminki", "lat": 55.7054, "lon": 37.7650},
]


PRESET_POINTS = {
    "red square": (55.7539, 37.6208, "Red Square"),
    "красная площадь": (55.7539, 37.6208, "Красная площадь"),
    "moscow city": (55.7473, 37.5398, "Moscow City"),
    "москва сити": (55.7473, 37.5398, "Москва-Сити"),
    "москва-сити": (55.7473, 37.5398, "Москва-Сити"),
    "москва city": (55.7473, 37.5398, "Москва-Сити"),
    "sheremetyevo": (55.9726, 37.4146, "Sheremetyevo"),
    "шереметьево": (55.9726, 37.4146, "Шереметьево"),
    "vnukovo": (55.5991, 37.2732, "Vnukovo"),
    "внуково": (55.5991, 37.2732, "Внуково"),
    "domodedovo": (55.4146, 37.8990, "Domodedovo"),
    "домодедово": (55.4146, 37.8990, "Домодедово"),
    "vdnh": (55.8240, 37.6385, "VDNH"),
    "вднх": (55.8240, 37.6385, "ВДНХ"),
    "arbat": (55.7522, 37.5907, "Arbat"),
    "арбат": (55.7522, 37.5907, "Арбат"),
    "kuzminki": (55.7054, 37.7650, "Кузьминки"),
    "кузьминки": (55.7054, 37.7650, "Кузьминки"),
    "метро кузьминки": (55.7054, 37.7650, "Метро Кузьминки"),
}


TARIFFS = {
    "Econom": {
        "base_rub": 189,
        "included_km": 1.0,
        "included_min": 3.0,
        "per_km_rub": 13,
        "per_min_rub": 13,
        "minimum_rub": 189,
    },
    "Comfort": {
        "base_rub": 219,
        "included_km": 1.0,
        "included_min": 3.0,
        "per_km_rub": 15,
        "per_min_rub": 15,
        "minimum_rub": 219,
    },
    "Comfort+": {
        "base_rub": 269,
        "included_km": 1.0,
        "included_min": 3.0,
        "per_km_rub": 18,
        "per_min_rub": 18,
        "minimum_rub": 269,
    },
    "Business": {
        "base_rub": 419,
        "included_km": 1.0,
        "included_min": 3.0,
        "per_km_rub": 25,
        "per_min_rub": 25,
        "minimum_rub": 419,
    },
}


@dataclass(frozen=True)
class GeocodeResult:
    lat: float
    lon: float
    display_name: str
    source: str


@dataclass(frozen=True)
class RouteInfo:
    distance_km: float
    duration_min: float
    coordinates: list[tuple[float, float]]
    source: str


@dataclass(frozen=True)
class WeatherInfo:
    temperature_c: float | None
    precipitation_mm: float | None
    rain_mm: float | None
    snowfall_cm: float | None
    wind_speed_ms: float | None
    weather_code: int | None
    factor: float
    source: str


def geocode_place(query: str, timeout: int = 10) -> GeocodeResult:
    cleaned = query.strip()
    if not cleaned:
        raise ValueError("Введите адрес или название места")

    parsed = parse_coordinates(cleaned)
    if parsed:
        return GeocodeResult(parsed[0], parsed[1], cleaned, "typed coordinates")

    normalized = normalize_place_key(cleaned)
    if normalized in PRESET_POINTS:
        lat, lon, display = PRESET_POINTS[normalized]
        return GeocodeResult(lat, lon, display, "local preset")

    lower = cleaned.lower().replace("ё", "е")
    search_query = cleaned if "москва" in lower or "moscow" in lower else f"{cleaned}, Москва"
    errors = []

    for provider in (_geocode_arcgis, _geocode_photon, _geocode_nominatim):
        try:
            result = provider(search_query, timeout)
            if result is not None:
                return result
        except Exception as exc:
            errors.append(f"{provider.__name__}: {exc}")

    details = "; ".join(errors[-2:])
    raise ValueError(f"Адрес не найден: {query}. Уточните улицу и номер дома. {details}")


def _geocode_arcgis(query: str, timeout: int) -> GeocodeResult | None:
    url = "https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/findAddressCandidates"
    params = {
        "SingleLine": query,
        "f": "json",
        "countryCode": "RUS",
        "location": f"{MOSCOW_CENTER[1]},{MOSCOW_CENTER[0]}",
        "maxLocations": 5,
        "outFields": "Match_addr,Addr_type",
    }
    payload = _get_json(url, params, timeout)
    for candidate in payload.get("candidates", []):
        location = candidate.get("location", {})
        lat, lon = float(location.get("y", 0)), float(location.get("x", 0))
        if _is_moscow_area(lat, lon):
            display = candidate.get("address") or query
            return GeocodeResult(lat, lon, display, "ArcGIS live geocoding")
    return None


def _geocode_photon(query: str, timeout: int) -> GeocodeResult | None:
    url = "https://photon.komoot.io/api/"
    params = {
        "q": query,
        "limit": 8,
        "lat": MOSCOW_CENTER[0],
        "lon": MOSCOW_CENTER[1],
    }
    payload = _get_json(url, params, timeout)
    candidates = []
    for feature in payload.get("features", []):
        coordinates = feature.get("geometry", {}).get("coordinates", [])
        properties = feature.get("properties", {})
        if len(coordinates) < 2:
            continue
        lon, lat = float(coordinates[0]), float(coordinates[1])
        if not _is_moscow_area(lat, lon):
            continue
        city = str(properties.get("city", "")).lower()
        state = str(properties.get("state", "")).lower()
        score = 2 if "москва" in city else 1 if "москва" in state else 0
        candidates.append((score, lat, lon, properties))

    if not candidates:
        return None

    _, lat, lon, properties = max(candidates, key=lambda item: item[0])
    display = _photon_display_name(properties, query)
    return GeocodeResult(lat, lon, display, "Photon live geocoding")


def _geocode_nominatim(query: str, timeout: int) -> GeocodeResult | None:
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
        "format": "jsonv2",
        "limit": 5,
        "countrycodes": "ru",
        "accept-language": "ru,en",
        "viewbox": "36.80,56.10,38.20,55.30",
        "bounded": 1,
    }
    payload = _get_json(url, params, timeout)
    for item in payload:
        lat, lon = float(item["lat"]), float(item["lon"])
        if _is_moscow_area(lat, lon):
            return GeocodeResult(lat, lon, item.get("display_name", query), "Nominatim live geocoding")
    return None


def _get_json(url: str, params: dict[str, Any], timeout: int) -> Any:
    headers = {"User-Agent": "RideFlowNN-Capstone/1.1"}
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        response = requests.get(url, params=params, headers=headers, timeout=timeout, verify=False)
    response.raise_for_status()
    return response.json()


def _is_moscow_area(lat: float, lon: float) -> bool:
    return 55.30 <= lat <= 56.10 and 36.80 <= lon <= 38.20


def _photon_display_name(properties: dict[str, Any], fallback: str) -> str:
    parts = []
    name = properties.get("name")
    street = properties.get("street")
    house = properties.get("housenumber")
    district = properties.get("district")
    city = properties.get("city")
    if name:
        parts.append(str(name))
    if street and str(street) not in parts:
        street_part = f"{street}, {house}" if house else str(street)
        parts.append(street_part)
    if district:
        parts.append(str(district))
    if city and str(city) not in parts:
        parts.append(str(city))
    return ", ".join(parts) or fallback


def normalize_place_key(text: str) -> str:
    normalized = text.lower().replace("ё", "е")
    normalized = normalized.replace("—", "-").replace("–", "-")
    normalized = re.sub(r"[\s_]+", " ", normalized).strip()
    compact = normalized.replace("-", " ")
    compact = re.sub(r"\s+", " ", compact).strip()
    if compact in PRESET_POINTS:
        return compact
    return normalized


def parse_coordinates(text: str) -> tuple[float, float] | None:
    match = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*[,; ]\s*(-?\d+(?:\.\d+)?)\s*$", text)
    if not match:
        return None
    lat = float(match.group(1))
    lon = float(match.group(2))
    if -90 <= lat <= 90 and -180 <= lon <= 180:
        return lat, lon
    return None


def haversine_km(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    radius_km = 6371.0
    lat1 = math.radians(a_lat)
    lat2 = math.radians(b_lat)
    d_lat = math.radians(b_lat - a_lat)
    d_lon = math.radians(b_lon - a_lon)
    value = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
    return 2 * radius_km * math.asin(math.sqrt(value))


def get_route(a_lat: float, a_lon: float, b_lat: float, b_lon: float, timeout: int = 8) -> RouteInfo:
    url = (
        "https://router.project-osrm.org/route/v1/driving/"
        f"{a_lon},{a_lat};{b_lon},{b_lat}?overview=full&geometries=geojson"
    )
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        route = payload["routes"][0]
        coordinates = [(lat, lon) for lon, lat in route["geometry"]["coordinates"]]
        return RouteInfo(
            distance_km=float(route["distance"]) / 1000.0,
            duration_min=float(route["duration"]) / 60.0,
            coordinates=coordinates,
            source="OSRM live route",
        )
    except Exception:
        direct_km = haversine_km(a_lat, a_lon, b_lat, b_lon)
        distance_km = direct_km * 1.35
        duration_min = distance_km / 28.0 * 60.0
        return RouteInfo(
            distance_km=distance_km,
            duration_min=duration_min,
            coordinates=[(a_lat, a_lon), (b_lat, b_lon)],
            source="Fallback haversine route",
        )


def get_weather(lat: float, lon: float, timeout: int = 8) -> WeatherInfo:
    return get_weather_at_time(lat, lon, pd.Timestamp.now().to_pydatetime(), timeout)


def get_weather_at_time(lat: float, lon: float, when: datetime | pd.Timestamp, timeout: int = 8) -> WeatherInfo:
    target = pd.Timestamp(when).floor("h")
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation,rain,snowfall,weather_code,wind_speed_10m",
        "timezone": "Europe/Moscow",
        "wind_speed_unit": "ms",
        "start_date": target.date().isoformat(),
        "end_date": target.date().isoformat(),
    }
    try:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            response = requests.get(url, params=params, timeout=timeout, verify=False)
        response.raise_for_status()
        hourly = response.json().get("hourly", {})
        times = pd.to_datetime(hourly.get("time", []))
        if len(times) == 0:
            raise ValueError("Open-Meteo returned no hourly forecast")
        idx = int(abs(times - target).argmin())
        precipitation = _list_float(hourly, "precipitation", idx)
        rain = _list_float(hourly, "rain", idx)
        snowfall = _list_float(hourly, "snowfall", idx)
        wind = _list_float(hourly, "wind_speed_10m", idx)
        weather_code = _list_int(hourly, "weather_code", idx)
        factor = weather_factor(precipitation, rain, snowfall, wind, weather_code)
        return WeatherInfo(
            temperature_c=_list_float(hourly, "temperature_2m", idx),
            precipitation_mm=precipitation,
            rain_mm=rain,
            snowfall_cm=snowfall,
            wind_speed_ms=wind,
            weather_code=weather_code,
            factor=factor,
            source=f"Open-Meteo forecast for {times[idx]}",
        )
    except Exception:
        return get_wttr_weather_at_time(lat, lon, target, timeout)


def get_wttr_weather_at_time(lat: float, lon: float, when: pd.Timestamp, timeout: int = 8) -> WeatherInfo:
    try:
        url = f"https://wttr.in/{lat:.4f},{lon:.4f}"
        params = {"format": "j1"}
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            response = requests.get(url, params=params, timeout=timeout, verify=False)
        response.raise_for_status()
        payload = response.json()

        day = None
        target_date = when.strftime("%Y-%m-%d")
        for item in payload.get("weather", []):
            if item.get("date") == target_date:
                day = item
                break

        hourly = None
        if day:
            target_hour = when.hour
            hourly_items = day.get("hourly", [])
            if hourly_items:
                hourly = min(hourly_items, key=lambda item: abs(int(item.get("time", "0") or 0) // 100 - target_hour))

        if hourly:
            temperature = _float_or_none(hourly.get("tempC"))
            precipitation = _float_or_none(hourly.get("precipMM"))
            wind = _float_or_none(hourly.get("windspeedKmph"))
            wind_ms = wind / 3.6 if wind is not None else None
            weather_code = _int_or_none(hourly.get("weatherCode"))
            factor = weather_factor(precipitation, precipitation, None, wind_ms, weather_code)
            return WeatherInfo(
                temperature_c=temperature,
                precipitation_mm=precipitation,
                rain_mm=precipitation,
                snowfall_cm=None,
                wind_speed_ms=wind_ms,
                weather_code=weather_code,
                factor=factor,
                source=f"wttr.in forecast for {target_date} {when.hour:02d}:00",
            )

        current = (payload.get("current_condition") or [{}])[0]
        temperature = _float_or_none(current.get("temp_C"))
        precipitation = _float_or_none(current.get("precipMM"))
        wind = _float_or_none(current.get("windspeedKmph"))
        wind_ms = wind / 3.6 if wind is not None else None
        weather_code = _int_or_none(current.get("weatherCode"))
        factor = weather_factor(precipitation, precipitation, None, wind_ms, weather_code)
        return WeatherInfo(
            temperature_c=temperature,
            precipitation_mm=precipitation,
            rain_mm=precipitation,
            snowfall_cm=None,
            wind_speed_ms=wind_ms,
            weather_code=weather_code,
            factor=factor,
            source="wttr.in current weather fallback",
        )
    except Exception:
        return WeatherInfo(None, None, None, None, None, None, 1.0, "Weather forecast unavailable")


def weather_factor(
    precipitation_mm: float | None,
    rain_mm: float | None,
    snowfall_cm: float | None,
    wind_speed_ms: float | None,
    weather_code: int | None,
) -> float:
    factor = 1.0
    if precipitation_mm and precipitation_mm > 0.2:
        factor += 0.06
    if rain_mm and rain_mm > 0.2:
        factor += 0.07
    if snowfall_cm and snowfall_cm > 0.0:
        factor += 0.14
    if wind_speed_ms and wind_speed_ms >= 10:
        factor += 0.05
    if weather_code in {45, 48}:
        factor += 0.03
    if weather_code in {95, 96, 99}:
        factor += 0.10
    return min(factor, 1.35)


def nearest_zone(lat: float, lon: float) -> dict[str, Any]:
    return min(
        MOSCOW_ZONE_POINTS,
        key=lambda zone: haversine_km(lat, lon, float(zone["lat"]), float(zone["lon"])),
    )


def demand_factor(forecast_path: Path, lat: float, lon: float) -> tuple[float, str, float | None]:
    return demand_factor_at_time(forecast_path, lat, lon, None)[:3]


def demand_factor_at_time(
    forecast_path: Path,
    lat: float,
    lon: float,
    when: datetime | pd.Timestamp | None,
) -> tuple[float, str, float | None, str]:
    zone_name = nearest_zone(lat, lon)["zone_name"]
    if not forecast_path.exists():
        return 1.0, zone_name, None, "forecast file missing"

    forecast = pd.read_csv(forecast_path)
    if forecast.empty or "zone_name" not in forecast:
        return 1.0, zone_name, None, "forecast file empty"

    forecast["timestamp"] = pd.to_datetime(forecast["timestamp"])
    target = pd.Timestamp(when).floor("h") if when is not None else forecast["timestamp"].min()

    exact_slice = forecast[forecast["timestamp"] == target]
    if exact_slice.empty:
        same_hour = forecast[forecast["timestamp"].dt.hour == target.hour]
        if not same_hour.empty:
            ranked_slice = same_hour.groupby("zone_name", as_index=False)["forecast_demand"].mean()
            source = f"demand fallback: average forecast for hour {target.hour:02d}:00"
        else:
            ranked_slice = forecast.groupby("zone_name", as_index=False)["forecast_demand"].mean()
            source = "demand fallback: average forecast horizon"
    else:
        ranked_slice = exact_slice
        source = f"demand forecast for {target}"

    zone_row = ranked_slice[ranked_slice["zone_name"] == zone_name]
    if zone_row.empty:
        return 1.0, zone_name, None, source

    demand = float(zone_row["forecast_demand"].iloc[0])
    cross_zone_percentile = float((ranked_slice["forecast_demand"] <= demand).mean())
    zone_history = forecast[forecast["zone_name"] == zone_name]["forecast_demand"]
    zone_time_percentile = float((zone_history <= demand).mean()) if not zone_history.empty else 0.0
    percentile = max(cross_zone_percentile, zone_time_percentile)
    if percentile >= 0.95:
        factor = 1.75
    elif percentile >= 0.9:
        factor = 1.55
    elif percentile >= 0.75:
        factor = 1.35
    elif percentile >= 0.55:
        factor = 1.15
    elif percentile >= 0.40:
        factor = 1.08
    else:
        factor = 1.0
    level = "low/normal" if factor == 1.0 else "medium" if factor <= 1.15 else "high"
    details = (
        f"{source}; cross-zone percentile={cross_zone_percentile:.0%}; "
        f"zone-time percentile={zone_time_percentile:.0%}; level={level}"
    )
    return factor, zone_name, demand, details


def traffic_factor_from_speed(route: RouteInfo, manual_factor: float) -> float:
    return max(route_speed_factor(route), manual_factor)


def route_speed_factor(route: RouteInfo) -> float:
    if route.duration_min <= 0:
        return 1.0
    speed_kmh = route.distance_km / (route.duration_min / 60.0)
    if speed_kmh < 12:
        return 1.60
    if speed_kmh < 18:
        return 1.40
    if speed_kmh < 25:
        return 1.25
    return 1.0


def time_factor(when: datetime | pd.Timestamp) -> tuple[float, str]:
    ts = pd.Timestamp(when)
    hour = ts.hour
    weekday = ts.weekday()
    factor = 1.0
    reasons = []

    if weekday < 5 and 7 <= hour <= 10:
        factor += 0.35
        reasons.append("weekday morning peak")
    if weekday < 5 and 17 <= hour <= 20:
        factor += 0.45
        reasons.append("weekday evening peak")
    if weekday == 4 and 18 <= hour <= 23:
        factor += 0.15
        reasons.append("Friday evening")
    if weekday >= 5 and (hour >= 21 or hour <= 2):
        factor += 0.25
        reasons.append("weekend night")
    if 0 <= hour <= 5:
        factor += 0.12
        reasons.append("low driver supply at night")

    return min(factor, 1.75), ", ".join(reasons) if reasons else "normal time"


def automatic_traffic_factor(route: RouteInfo, when: datetime | pd.Timestamp) -> tuple[float, str]:
    speed_factor = route_speed_factor(route)
    clock_factor, reason = time_factor(when)
    factor = 1.0 + (speed_factor - 1.0) * 0.55 + (clock_factor - 1.0) * 0.65
    factor = min(max(factor, speed_factor, clock_factor), 1.90)
    source = f"route speed x{speed_factor:.2f}; time x{clock_factor:.2f} ({reason})"
    return factor, source


def estimate_fare(
    tariff_name: str,
    route: RouteInfo,
    demand_multiplier: float,
    weather_multiplier: float,
    traffic_multiplier: float,
) -> dict[str, float]:
    tariff = TARIFFS[tariff_name]
    extra_km = max(0.0, route.distance_km - tariff["included_km"])
    extra_min = max(0.0, route.duration_min - tariff["included_min"])
    raw = tariff["base_rub"] + extra_km * tariff["per_km_rub"] + extra_min * tariff["per_min_rub"]
    raw = max(raw, tariff["minimum_rub"])
    multiplier = demand_multiplier * weather_multiplier * traffic_multiplier
    total = raw * multiplier
    return {
        "base_price_rub": round(raw, 0),
        "multiplier": round(multiplier, 2),
        "estimated_price_rub": round(total, 0),
        "demand_multiplier": round(demand_multiplier, 2),
        "weather_multiplier": round(weather_multiplier, 2),
        "traffic_multiplier": round(traffic_multiplier, 2),
    }


def _list_float(hourly: dict[str, list[Any]], key: str, idx: int) -> float | None:
    values = hourly.get(key) or []
    if idx >= len(values):
        return None
    return _float_or_none(values[idx])


def _list_int(hourly: dict[str, list[Any]], key: str, idx: int) -> int | None:
    values = hourly.get(key) or []
    if idx >= len(values):
        return None
    return _int_or_none(values[idx])


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
