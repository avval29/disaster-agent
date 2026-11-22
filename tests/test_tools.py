import requests

import app.tools.base as base
from app.tools.quake import get_recent_earthquakes
from app.tools.weather import get_weather_forecast

DELHI = (28.6139, 77.2090)


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)

    def json(self):
        return self._payload


def install_fake_get(monkeypatch, behaviour):
    calls = {"n": 0}

    def fake_get(url, params=None, timeout=None):
        calls["n"] += 1
        return behaviour(calls["n"])

    monkeypatch.setattr(base.requests, "get", fake_get)
    return calls


def always_fail(n):
    raise requests.ConnectionError("network down")


def weather_payload(rain=(0, 0, 80, 120, 30), tmax=(33, 34, 35, 36, 37)):
    return {
        "timezone": "Asia/Kolkata",
        "current": {"temperature_2m": 30.0, "relative_humidity_2m": 60, "precipitation": 0.0},
        "daily": {
            "time": [f"2026-11-{20 + i}" for i in range(5)],
            "precipitation_sum": list(rain),
            "precipitation_probability_max": [10, 20, 90, 95, 60],
            "temperature_2m_max": list(tmax),
            "temperature_2m_min": [22] * 5,
            "apparent_temperature_max": [t + 3 for t in tmax],
        },
    }


def quake_payload():
    def feat(mag, lon, lat, depth):
        return {
            "properties": {"mag": mag, "place": "Test place", "time": 1763800000000,
                           "tsunami": 0, "alert": None, "url": "http://example.org"},
            "geometry": {"coordinates": [lon, lat, depth]},
        }
    return {"features": [feat(3.1, 77.2, 28.7, 5.0), feat(4.6, 77.5, 29.0, 10.0), feat(None, 77.0, 28.0, 1.0)]}


# ------------------------------------------------------------ weather
def test_weather_live_parsing(monkeypatch):
    install_fake_get(monkeypatch, lambda n: FakeResponse(weather_payload()))
    res = get_weather_forecast(*DELHI)
    assert res.ok and res.source == "live"
    s = res.data["summary"]
    assert s["past_2d_rain_mm"] == 0
    assert s["today_rain_mm"] == 80
    assert s["next_3d_rain_mm"] == 230
    assert s["max_temp_next_3d_c"] == 37
    assert s["max_feels_like_next_3d_c"] == 40


def test_retry_then_success(monkeypatch):
    def flaky(n):
        if n < 3:
            raise requests.ConnectionError("temporary")
        return FakeResponse(weather_payload())

    calls = install_fake_get(monkeypatch, flaky)
    res = get_weather_forecast(*DELHI)
    assert res.ok and res.source == "live"
    assert calls["n"] == 3


def test_client_error_not_retried(monkeypatch):
    calls = install_fake_get(monkeypatch, lambda n: FakeResponse({}, status=400))
    res = get_weather_forecast(*DELHI)
    assert not res.ok
    assert calls["n"] == 1


def test_fallback_to_cache(monkeypatch):
    install_fake_get(monkeypatch, lambda n: FakeResponse(weather_payload()))
    assert get_weather_forecast(*DELHI).source == "live"

    install_fake_get(monkeypatch, always_fail)
    res = get_weather_forecast(*DELHI)
    assert res.ok and res.source == "cache"
    assert "live fetch failed" in res.error
    assert res.data["summary"]["next_3d_rain_mm"] == 230


def test_no_cache_returns_failure(monkeypatch):
    calls = install_fake_get(monkeypatch, always_fail)
    res = get_weather_forecast(*DELHI)
    assert not res.ok and res.source == "none"
    assert calls["n"] == 3


def test_simulated_failure_uses_cache(monkeypatch):
    install_fake_get(monkeypatch, lambda n: FakeResponse(weather_payload()))
    get_weather_forecast(*DELHI)
    monkeypatch.setenv("SIMULATE_API_FAILURE", "weather")
    res = get_weather_forecast(*DELHI)
    assert res.ok and res.source == "cache"
    assert "simulated" in res.error


def test_simulated_failure_no_cache(monkeypatch):
    calls = install_fake_get(monkeypatch, lambda n: FakeResponse(weather_payload()))
    monkeypatch.setenv("SIMULATE_API_FAILURE", "weather")
    res = get_weather_forecast(*DELHI)
    assert not res.ok and "simulated" in res.error
    assert calls["n"] == 0  # the network was never touched


def test_invalid_coordinates(monkeypatch):
    calls = install_fake_get(monkeypatch, lambda n: FakeResponse(weather_payload()))
    res = get_weather_forecast(95, 0)
    assert not res.ok and "latitude" in res.error
    assert calls["n"] == 0


def test_malformed_payload(monkeypatch):
    install_fake_get(monkeypatch, lambda n: FakeResponse({"unexpected": True}))
    res = get_weather_forecast(*DELHI)
    assert not res.ok and res.source == "none"


# ------------------------------------------------------------ earthquakes
def test_quake_parsing(monkeypatch):
    install_fake_get(monkeypatch, lambda n: FakeResponse(quake_payload()))
    res = get_recent_earthquakes(*DELHI)
    assert res.ok
    assert res.data["count"] == 2  # the event with no magnitude is skipped
    assert res.data["max_magnitude"] == 4.6
    assert res.data["strongest"]["magnitude"] == 4.6
    assert 40 < res.data["strongest"]["distance_km"] < 65


def test_quake_no_events(monkeypatch):
    install_fake_get(monkeypatch, lambda n: FakeResponse({"features": []}))
    res = get_recent_earthquakes(*DELHI)
    assert res.ok and res.data["count"] == 0
    assert res.data["max_magnitude"] is None


def test_quake_invalid_input():
    res = get_recent_earthquakes(*DELHI, radius_km=0)
    assert not res.ok and "radius_km" in res.error