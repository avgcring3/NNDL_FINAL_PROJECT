from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from ride_pricing import (
    MOSCOW_CENTER,
    MOSCOW_ZONE_POINTS,
    TARIFFS,
    automatic_traffic_factor,
    demand_factor_at_time,
    estimate_fare,
    geocode_place,
    get_route,
    get_weather_at_time,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"


st.set_page_config(page_title="RideFlow NN", page_icon="RF", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.3rem; padding-bottom: 2rem; max-width: 1320px; }
    .hero {
        background: linear-gradient(135deg, #101828 0%, #1d2939 48%, #0f6cbd 100%);
        color: white;
        padding: 30px 34px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .hero h1 { margin: 0 0 8px 0; font-size: 44px; letter-spacing: 0; }
    .hero p { margin: 0; color: #d6e4ff; font-size: 18px; }
    .metric-card {
        background: #ffffff;
        border: 1px solid #d0d7de;
        border-radius: 8px;
        padding: 18px 18px 14px;
        min-height: 112px;
    }
    .metric-label { color: #667085; font-size: 13px; text-transform: uppercase; font-weight: 700; }
    .metric-value { color: #101828; font-size: 34px; font-weight: 800; margin-top: 6px; }
    .metric-note { color: #667085; font-size: 13px; margin-top: 4px; }
    .status-pill {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        background: #ecfdf3;
        color: #027a48;
        font-weight: 700;
        font-size: 13px;
        margin-right: 8px;
    }
    .warn-pill {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        background: #fff7ed;
        color: #b45309;
        font-weight: 700;
        font-size: 13px;
        margin-right: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


PRESET_POINTS = {
    "Red Square": (55.7539, 37.6208),
    "Moscow City": (55.7473, 37.5398),
    "Sheremetyevo": (55.9726, 37.4146),
    "Vnukovo": (55.5991, 37.2732),
    "Domodedovo": (55.4146, 37.8990),
    "VDNH": (55.8240, 37.6385),
    "Arbat": (55.7522, 37.5907),
    "Khamovniki": (55.7272, 37.5671),
}


@st.cache_data
def load_metrics() -> dict:
    path = ARTIFACTS / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_predictions() -> pd.DataFrame:
    path = ARTIFACTS / "predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


@st.cache_data
def load_future_forecast() -> pd.DataFrame:
    path = ARTIFACTS / "future_forecast.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


@st.cache_data
def load_training_curve() -> pd.DataFrame:
    path = ARTIFACTS / "training_curve.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=86400)
def cached_geocode(query: str):
    return geocode_place(query)


@st.cache_data(ttl=300)
def cached_route(a_lat: float, a_lon: float, b_lat: float, b_lon: float):
    return get_route(a_lat, a_lon, b_lat, b_lon)


@st.cache_data(ttl=120)
def cached_weather(lat: float, lon: float, when_iso: str):
    return get_weather_at_time(lat, lon, pd.Timestamp(when_iso))


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pct_improvement(model_value: float, baseline_value: float) -> float:
    if baseline_value == 0:
        return 0.0
    return (baseline_value - model_value) / baseline_value * 100.0


def point_label(point: tuple[float, float] | None) -> str:
    if not point:
        return "not selected"
    return f"{point[0]:.5f}, {point[1]:.5f}"


def build_price_map(point_a: tuple[float, float] | None, point_b: tuple[float, float] | None, route_points=None):
    center = point_a or point_b or MOSCOW_CENTER
    taxi_map = folium.Map(location=center, zoom_start=11, control_scale=True, tiles="OpenStreetMap")

    for zone in MOSCOW_ZONE_POINTS:
        folium.CircleMarker(
            location=(zone["lat"], zone["lon"]),
            radius=4,
            color="#0f6cbd",
            fill=True,
            fill_opacity=0.6,
            tooltip=zone["zone_name"],
        ).add_to(taxi_map)

    if point_a:
        folium.Marker(point_a, tooltip="Point A", icon=folium.Icon(color="green", icon="play")).add_to(taxi_map)
    if point_b:
        folium.Marker(point_b, tooltip="Point B", icon=folium.Icon(color="red", icon="stop")).add_to(taxi_map)
    if route_points:
        folium.PolyLine(route_points, color="#0f6cbd", weight=5, opacity=0.85, tooltip="Route").add_to(taxi_map)

    return taxi_map


def geocode_into_state(address_key: str, point_key: str, label_key: str) -> None:
    result = cached_geocode(st.session_state[address_key])
    st.session_state[point_key] = (result.lat, result.lon)
    st.session_state[label_key] = result.display_name
    st.toast(f"Found {result.display_name}")


if "address_a" not in st.session_state:
    st.session_state.address_a = "Красная площадь"
if "address_b" not in st.session_state:
    st.session_state.address_b = "Москва-Сити"
if "point_a" not in st.session_state:
    st.session_state.point_a = PRESET_POINTS["Red Square"]
if "point_b" not in st.session_state:
    st.session_state.point_b = PRESET_POINTS["Moscow City"]
if "point_a_label" not in st.session_state:
    st.session_state.point_a_label = "Красная площадь"
if "point_b_label" not in st.session_state:
    st.session_state.point_b_label = "Москва-Сити"


metrics = load_metrics()
predictions = load_predictions()
future = load_future_forecast()
curve = load_training_curve()

st.markdown(
    """
    <div class="hero">
      <h1>RideFlow NN</h1>
      <p>Neural taxi demand forecasting, route pricing, weather and demand-aware dispatch planning.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not metrics or predictions.empty:
    st.error("No artifacts found. Run START_RIDEFLOW.bat or python -m src.train --mode demo --city moscow --epochs 5")
    st.stop()

metadata = metrics.get("metadata", {})
model_names = [name for name, value in metrics.items() if isinstance(value, dict) and "mae" in value and not name.startswith("baseline")]
model_name = model_names[0] if model_names else "model"
model_metrics = metrics[model_name]
last_metrics = metrics.get("baseline_last", {})
mae_gain = pct_improvement(model_metrics["mae"], last_metrics.get("mae", model_metrics["mae"]))
peak_gain = pct_improvement(model_metrics["peak_mae"], last_metrics.get("peak_mae", model_metrics["peak_mae"]))

st.markdown(
    f"""
    <span class="status-pill">Ready</span>
    <span class="status-pill">{metadata.get("mode", "demo").upper()}</span>
    <span class="status-pill">{metadata.get("future_hours", 0)}h future forecast</span>
    <span class="warn-pill">Estimated taxi fare, not official Yandex Go price</span>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(["Overview", "Ride Price", "Future Forecast", "Backtest", "Operations", "Model", "Pitch"])

with tabs[0]:
    st.subheader("Executive Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Transformer MAE", f"{model_metrics['mae']:.2f}", f"{mae_gain:.1f}% better than last-value")
    with col2:
        metric_card("Peak MAE", f"{model_metrics['peak_mae']:.2f}", f"{peak_gain:.1f}% better on peak demand")
    with col3:
        metric_card("Future Horizon", f"{metadata.get('future_hours', 0)}h", "forecast by zone")
    with col4:
        metric_card("Lookback", f"{metadata.get('lookback_hours', 0)}h", "sequence window")

    st.divider()
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("#### Model vs Baselines")
        metric_rows = []
        for name, values in metrics.items():
            if isinstance(values, dict) and "mae" in values:
                metric_rows.append({"model": name, **values})
        st.dataframe(pd.DataFrame(metric_rows), width="stretch", hide_index=True)
    with right:
        st.markdown("#### Training Curve")
        if not curve.empty:
            st.line_chart(curve.set_index("epoch")[["train_loss", "val_loss"]])
        else:
            st.info("Training curve not found.")

with tabs[1]:
    st.subheader("Route Price By Address And Time")
    st.caption("Type addresses, choose trip date/time, and the app computes the coefficient from demand forecast, weather forecast, route speed and time-of-day conditions.")

    now_dt = pd.Timestamp.now(tz="Europe/Moscow").tz_localize(None).ceil("h")
    min_dt = now_dt
    if future.empty:
        max_dt = now_dt + pd.Timedelta(days=14)
    else:
        max_dt = max(future["timestamp"].max(), now_dt + pd.Timedelta(days=1))
    default_dt = min(now_dt + pd.Timedelta(hours=1), max_dt)

    controls, map_col = st.columns([0.92, 1.35])
    with controls:
        st.markdown("#### Trip Inputs")
        st.text_input("Point A", key="address_a", placeholder="Например: Красная площадь")
        if st.button("Find Point A", width="stretch"):
            try:
                geocode_into_state("address_a", "point_a", "point_a_label")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

        st.text_input("Point B", key="address_b", placeholder="Например: Москва-Сити")
        if st.button("Find Point B", width="stretch"):
            try:
                geocode_into_state("address_b", "point_b", "point_b_label")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

        trip_date = st.date_input("Trip date", value=default_dt.date(), min_value=min_dt.date(), max_value=max_dt.date())
        trip_time = st.time_input("Trip time", value=default_dt.time().replace(minute=0, second=0, microsecond=0), step=3600)
        selected_dt = pd.Timestamp(datetime.combine(trip_date, trip_time)).floor("h")
        if selected_dt < min_dt or selected_dt > max_dt:
            st.warning(f"Demand forecast covers {min_dt} to {max_dt}. Outside this range, demand falls back to same-hour profile.")

        tariff = st.selectbox("Tariff", list(TARIFFS), index=0)

        st.markdown("#### Resolved Points")
        st.write(f"A: `{st.session_state.point_a_label}`")
        st.write(f"`{point_label(st.session_state.point_a)}`")
        st.write(f"B: `{st.session_state.point_b_label}`")
        st.write(f"`{point_label(st.session_state.point_b)}`")

    route_for_map = None
    if st.session_state.point_a and st.session_state.point_b:
        route_for_map = cached_route(*st.session_state.point_a, *st.session_state.point_b)

    with map_col:
        taxi_map = build_price_map(
            st.session_state.point_a,
            st.session_state.point_b,
            route_for_map.coordinates if route_for_map else None,
        )
        map_state = st_folium(taxi_map, height=560, width=850, returned_objects=["last_clicked"])
        clicked = map_state.get("last_clicked") if map_state else None
        if clicked:
            clicked_point = (float(clicked["lat"]), float(clicked["lng"]))
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                if st.button("Set Click As A", width="stretch"):
                    st.session_state.point_a = clicked_point
                    st.session_state.point_a_label = "Map click"
                    st.rerun()
            with c2:
                if st.button("Set Click As B", width="stretch"):
                    st.session_state.point_b = clicked_point
                    st.session_state.point_b_label = "Map click"
                    st.rerun()
            with c3:
                st.write(f"Clicked: `{point_label(clicked_point)}`")

    if st.session_state.point_a and st.session_state.point_b:
        route = cached_route(*st.session_state.point_a, *st.session_state.point_b)
        mid_lat = (st.session_state.point_a[0] + st.session_state.point_b[0]) / 2
        mid_lon = (st.session_state.point_a[1] + st.session_state.point_b[1]) / 2
        weather = cached_weather(mid_lat, mid_lon, selected_dt.isoformat())
        demand_multiplier, demand_zone, demand_value, demand_source = demand_factor_at_time(
            ARTIFACTS / "future_forecast.csv",
            st.session_state.point_a[0],
            st.session_state.point_a[1],
            selected_dt,
        )
        traffic_multiplier, traffic_source = automatic_traffic_factor(route, selected_dt)
        fare = estimate_fare(tariff, route, demand_multiplier, weather.factor, traffic_multiplier)

        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Estimated Price", f"{fare['estimated_price_rub']:.0f} ₽", f"{tariff}, coefficient x{fare['multiplier']:.2f}")
        with c2:
            metric_card("Distance", f"{route.distance_km:.1f} km", route.source)
        with c3:
            metric_card("Duration", f"{route.duration_min:.0f} min", "route estimate")
        with c4:
            metric_card("Trip Time", selected_dt.strftime("%d.%m %H:%M"), "Moscow time")

        st.markdown("#### Automatic Coefficient Breakdown")
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            metric_card("Demand", f"x{fare['demand_multiplier']:.2f}", f"{demand_zone}: forecast {demand_value:.1f}" if demand_value is not None else demand_source)
        with f2:
            metric_card("Weather", f"x{fare['weather_multiplier']:.2f}", weather.source)
        with f3:
            metric_card("Traffic / Time", f"x{fare['traffic_multiplier']:.2f}", traffic_source)
        with f4:
            metric_card("Combined", f"x{fare['multiplier']:.2f}", "demand x weather x traffic/time")
        st.caption("A coefficient of x1.00 means that the selected condition is normal, not that the component was skipped.")
        factor_frame = pd.DataFrame(
            [
                {"component": "Base fare before coefficients", "value": fare["base_price_rub"], "source": "tariff preset"},
                {"component": "Demand coefficient", "value": fare["demand_multiplier"], "source": f"{demand_zone}, forecast={demand_value:.1f}; {demand_source}" if demand_value else demand_source},
                {"component": "Weather coefficient", "value": fare["weather_multiplier"], "source": weather.source},
                {"component": "Traffic/time coefficient", "value": fare["traffic_multiplier"], "source": traffic_source},
                {"component": "Final combined coefficient", "value": fare["multiplier"], "source": "demand x weather x traffic/time"},
            ]
        )
        st.dataframe(factor_frame, width="stretch", hide_index=True)
        st.write(
            f"Weather forecast: temp `{weather.temperature_c}`, precipitation `{weather.precipitation_mm}`, "
            f"rain `{weather.rain_mm}`, snowfall `{weather.snowfall_cm}`, wind `{weather.wind_speed_ms}`"
        )

with tabs[2]:
    st.subheader("Future Forecast")
    if future.empty:
        st.warning("Future forecast is missing. Run START_RIDEFLOW.bat again to generate artifacts/future_forecast.csv.")
    else:
        min_time = future["timestamp"].min()
        max_time = future["timestamp"].max()
        zone_names = sorted(future["zone_name"].unique())
        selected_zone = st.selectbox("Zone", zone_names, key="future_zone")
        horizon_hours = st.slider("Hours to display", 12, int(metadata.get("future_hours", 72)), min(72, int(metadata.get("future_hours", 72))), step=12)

        zone_future = future[(future["zone_name"] == selected_zone) & (future["step_hour"] <= horizon_hours)].sort_values("timestamp")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.line_chart(zone_future.set_index("timestamp")[["forecast_demand"]])
        with col2:
            next_24 = future[future["step_hour"] <= 24]
            zone_24 = next_24[next_24["zone_name"] == selected_zone]["forecast_demand"].sum()
            peak_row = zone_future.loc[zone_future["forecast_demand"].idxmax()]
            metric_card("Next 24h Demand", f"{zone_24:.0f}", selected_zone)
            metric_card("Peak Hour", f"{peak_row['forecast_demand']:.0f}", str(peak_row["timestamp"]))

        st.markdown(f"Forecast window: `{min_time}` to `{max_time}`")
        st.markdown("#### Future Rows")
        st.dataframe(zone_future, width="stretch", hide_index=True)

with tabs[3]:
    st.subheader("Backtest: Prediction vs Actual")
    zone_names = sorted(predictions["zone_name"].unique())
    selected_zone = st.selectbox("Zone", zone_names, key="backtest_zone")
    zone_frame = predictions[predictions["zone_name"] == selected_zone].sort_values("timestamp")

    col1, col2 = st.columns([2, 1])
    with col1:
        chart_frame = zone_frame.set_index("timestamp")[["actual", "model_prediction", "baseline_day_ago"]]
        st.line_chart(chart_frame)
    with col2:
        zone_mae = (zone_frame["model_prediction"] - zone_frame["actual"]).abs().mean()
        baseline_mae = (zone_frame["baseline_day_ago"] - zone_frame["actual"]).abs().mean()
        metric_card("Zone MAE", f"{zone_mae:.2f}", selected_zone)
        metric_card("Day Baseline MAE", f"{baseline_mae:.2f}", "same hour yesterday")

    st.markdown("#### Backtest Rows")
    st.dataframe(zone_frame, width="stretch", hide_index=True)

with tabs[4]:
    st.subheader("Operational Recommendations")
    if future.empty:
        st.warning("Future forecast is missing. Run START_RIDEFLOW.bat again.")
    else:
        next_hour = future[future["step_hour"] == 1].copy()
        next_24 = (
            future[future["step_hour"] <= 24]
            .groupby("zone_name", as_index=False)
            .agg(next_24h_demand=("forecast_demand", "sum"), peak_hour_demand=("forecast_demand", "max"))
            .sort_values("next_24h_demand", ascending=False)
        )
        next_72 = (
            future.groupby("zone_name", as_index=False)
            .agg(next_72h_demand=("forecast_demand", "sum"), peak_hour_demand=("forecast_demand", "max"))
            .sort_values("next_72h_demand", ascending=False)
        )

        left, right = st.columns(2)
        with left:
            st.markdown("#### Send Drivers Now")
            st.dataframe(
                next_hour.sort_values("forecast_demand", ascending=False).head(10)[["zone_name", "timestamp", "forecast_demand"]],
                width="stretch",
                hide_index=True,
            )
        with right:
            st.markdown("#### Top Zones Next 24h")
            st.dataframe(next_24.head(10), width="stretch", hide_index=True)

        st.markdown("#### Demand Plan")
        st.dataframe(next_72.head(12), width="stretch", hide_index=True)

with tabs[5]:
    st.subheader("Model And Data")
    st.markdown(
        f"""
        - **Model:** {model_name}
        - **Mode:** {metadata.get("mode")}
        - **Source:** `{metadata.get("source", "unknown")}`
        - **Lookback:** {metadata.get("lookback_hours")} hours
        - **Backtest horizon:** {metadata.get("horizon_hours")} hour
        - **Future forecast:** {metadata.get("future_hours", 0)} hours
        - **Samples:** {metadata.get("samples")}
        - **Latest TLC public data:** April 2026 for a June 2026 pitch, per NYC TLC page.
        """
    )
    st.markdown("#### Capstone Fit")
    st.markdown(
        """
        - Real operational dataset path: NYC TLC trip records.
        - Neural network: Transformer encoder trained on hourly demand windows.
        - Course concepts: self-attention, MLP baseline, custom weighted loss, time-series representation learning.
        - Moscow path: same pipeline works with partner pickup logs.
        - Price module: typed addresses, selected trip time, route, weather, traffic/time and demand forecast are combined into an estimated fare.
        """
    )

with tabs[6]:
    st.subheader("Pitch Assets")
    st.markdown(
        """
        Use these files for presentation:

        - `docs/pitch_outline.md`
        - `docs/pitch_deck.html`
        - `docs/moscow_data_note.md`
        - `reports/demo_report.html`
        """
    )
    report_path = REPORTS / "demo_report.html"
    deck_path = DOCS / "pitch_deck.html"
    col1, col2 = st.columns(2)
    with col1:
        if report_path.exists():
            st.link_button("Open HTML Report", str(report_path))
    with col2:
        if deck_path.exists():
            st.link_button("Open Pitch Deck", str(deck_path))
