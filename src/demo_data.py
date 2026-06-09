from __future__ import annotations

import numpy as np
import pandas as pd


MOSCOW_ZONES = [
    (1, "Tverskoy", 42.0),
    (2, "Arbat", 30.0),
    (3, "Moscow City", 55.0),
    (4, "Sokolniki", 22.0),
    (5, "Tagansky", 28.0),
    (6, "Khamovniki", 34.0),
    (7, "VDNH", 24.0),
    (8, "Sheremetyevo", 48.0),
    (9, "Vnukovo", 36.0),
    (10, "Domodedovo", 44.0),
    (11, "Yasenevo", 18.0),
    (12, "Kuzminki", 21.0),
]


def generate_hourly_demo(city: str = "moscow", days: int = 60, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if city.lower() != "moscow":
        raise ValueError("RideFlow NN supports Moscow demand data only.")
    zones = MOSCOW_ZONES
    end_time = pd.Timestamp.now(tz="Europe/Moscow").tz_localize(None).floor("h") - pd.Timedelta(hours=1)
    hours = pd.date_range(end=end_time, periods=days * 24, freq="h")
    rows = []

    event_days = set(rng.choice(np.arange(3, days - 3), size=max(days // 10, 1), replace=False))

    for zone_id, zone_name, base in zones:
        airport = "Airport" in zone_name or zone_name in {"Sheremetyevo", "Vnukovo", "Domodedovo"}
        business = zone_name == "Moscow City"
        center = zone_name in {"Tverskoy", "Arbat"}

        for hour in hours:
            hour_of_day = hour.hour
            weekday = hour.weekday()
            day_index = (hour.normalize() - hours[0].normalize()).days

            morning_peak = np.exp(-((hour_of_day - 8) ** 2) / 8.0)
            evening_peak = np.exp(-((hour_of_day - 18) ** 2) / 10.0)
            night_peak = np.exp(-((hour_of_day - 23) ** 2) / 7.0)
            airport_peak = 0.7 + 0.4 * np.sin((hour_of_day / 24.0) * 2 * np.pi + 0.8)

            weekday_factor = 1.15 if weekday < 5 else 0.92
            weekend_night = 1.35 if weekday >= 5 and hour_of_day >= 20 else 1.0
            event_factor = 1.45 if day_index in event_days and center and 18 <= hour_of_day <= 23 else 1.0

            demand = base
            if airport:
                demand *= 0.75 + airport_peak
            elif business:
                demand *= 0.75 + 0.9 * morning_peak + 0.65 * evening_peak
            elif center:
                demand *= 0.85 + 0.45 * evening_peak + 0.55 * night_peak
            else:
                demand *= 0.8 + 0.55 * morning_peak + 0.5 * evening_peak

            demand *= weekday_factor * weekend_night * event_factor
            demand += 0.03 * day_index * base
            demand += rng.normal(0.0, max(base * 0.08, 2.0))
            demand = max(demand, 0.0)

            rows.append(
                {
                    "city": city.lower(),
                    "zone_id": zone_id,
                    "zone_name": zone_name,
                    "hour": hour,
                    "demand": int(round(demand)),
                }
            )

    return pd.DataFrame(rows)
