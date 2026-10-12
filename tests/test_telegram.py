import numpy as np
import pandas as pd
import pytest

from airraid.telegram import FEATURES, available_messages, feature_row, parse_page
from airraid.telegram_experiment import block_comparison, fit_with_text


def messages():
    d = pd.DataFrame([
        (1, "2026-09-30T20:00Z", "2026-09-30T20:01Z", False, True, True),
        (2, "2026-09-30T21:00Z", "2026-09-30T21:01Z", False, True, True),
        (3, "2026-09-30T19:00Z", "2026-10-12T08:00Z", False, True, True),
        (4, "2026-09-30T18:00Z", "2026-09-30T18:01Z", True, True, False),
        (5, "2026-09-30T17:00Z", "2026-09-30T17:01Z", False, False, False),
    ], columns=["message_id", "published_at", "observed_at", "edited_marker", "has_text", "threat_keyword"])
    for c in ["published_at", "observed_at"]:
        d[c] = pd.to_datetime(d[c], utc=True, format="mixed")
    return d


def test_parser_excludes_reply_and_current_reactions():
    page = '''<div data-post="war_monitor/1"><div class="tgme_widget_message_text js-message_reply_text">ракета</div>
    <div class="tgme_widget_message_text js-message_text">тест<br>рядок &amp; дані</div>
    <span class="tgme_widget_message_meta">edited</span><span>ракет 999 reactions</span>
    <a><time datetime="2026-09-30T20:00:00+00:00">23:00</time></a></div>'''
    r = parse_page(page, "2026-10-12T08:00:00Z")[0]
    assert r["message_id"] == 1 and r["has_text"] and r["edited_marker"]
    assert not r["threat_keyword"] and "text" not in r


def test_parser_collapses_identical_mobile_desktop_bodies_only():
    body = '<div class="tgme_widget_message_text js-message_text">загроза</div>'
    page = '<div data-post="war_monitor/1">' + body * 2 + '<time datetime="2026-09-30T20:00:00Z"></time></div>'
    assert len(parse_page(page, "2026-10-12T08:00Z")) == 1
    with pytest.raises(ValueError, match="Ambiguous"):
        parse_page(page.replace(body + body, body + body.replace("загроза", "зліт")), "2026-10-12T08:00Z")


def test_observed_mode_does_not_backdate_fetched_text():
    d = messages()
    row = feature_row(d, "2026-10-01", "observed")
    assert row["posts_24h"] == 2  # post 1 and edited post 4 captured before Kyiv midnight
    assert row["threat_posts_24h"] == 1
    exploratory = feature_row(d, "2026-10-01", "publication_assumed")
    assert exploratory["posts_24h"] == 2  # post 1 and unproven post 3, edited post 4 excluded
    assert exploratory["threat_posts_24h"] == 2


def test_future_observation_or_future_publication_cannot_change_strict_features():
    d = messages()
    original = feature_row(d, "2026-10-01")
    future = d.iloc[[0]].copy()
    future["observed_at"] = pd.Timestamp("2026-10-12T09:00Z")
    future["threat_keyword"] = False
    assert feature_row(pd.concat([d, future]), "2026-10-01") == original
    future["published_at"] = pd.Timestamp("2026-10-02T00:00Z")
    assert feature_row(pd.concat([d, future]), "2026-10-01") == original


def test_publication_at_midnight_and_observation_at_midnight_excluded():
    origin = pd.Timestamp("2026-10-01T00:00:00+03:00")
    d = messages().iloc[[0]].copy()
    d["observed_at"] = origin
    assert available_messages(d, origin).empty
    d["published_at"] = origin
    assert available_messages(d, origin, "publication_assumed").empty


def test_24_elapsed_hour_window_on_dst_day():
    d = messages().iloc[[0]].copy()
    d["published_at"] = pd.Timestamp("2026-03-28T20:30Z")
    d["observed_at"] = pd.Timestamp("2026-03-28T20:31Z")
    # Kyiv midnight March 30 is 21Z; previous 24 elapsed hours begin March 28 21Z.
    assert feature_row(d, "2026-03-30")["posts_24h"] == 0


def test_unknown_mode_or_naive_origin_rejected():
    with pytest.raises(ValueError):
        available_messages(messages(), pd.Timestamp("2026-10-01"))
    with pytest.raises(ValueError):
        feature_row(messages(), "2026-10-01", "invented")


def test_exogenous_calendar_alignment_and_nan_rejected():
    with pytest.raises(ValueError, match="Misaligned"):
        fit_with_text(np.ones(60), np.ones((59, 3)), np.ones((2, 3)), "arima")
    with pytest.raises(ValueError, match="Nonfinite"):
        fit_with_text(np.ones(60), np.full((60, 3), np.nan), np.ones((2, 3)), "arima")


def test_constant_series_remains_in_support():
    point, status, _ = fit_with_text(np.zeros(60), np.arange(180).reshape(60, 3), np.ones((2, 3)), "arima")
    assert point == 0 and status == "constant_series"


def test_block_comparison_preserves_pairs_and_never_certifies_assumed_success():
    rows = []
    for date in pd.date_range("2026-09-13", periods=28).strftime("%Y-%m-%d"):
        for region in ["A", "B"]:
            for model, prediction in [("arima_without_telegram", 2), ("arima_with_telegram_assumed", 1),
                                      ("mean_7", 2), ("original_selected", 2)]:
                rows.append(dict(date=date, region=region, target="count", model=model, prediction=prediction, actual=0))
    d = pd.DataFrame(rows)
    result = block_comparison(d, repetitions=30)
    assert result.meets_numeric_5_percent.all() and not result.certified_success.any()
    assert (result.mae_difference_ci_low == 1).all()
    with pytest.raises(ValueError, match="Unpaired"):
        block_comparison(d.iloc[1:], repetitions=10)


def test_scaler_is_fitted_on_training_features_only(monkeypatch):
    import airraid.telegram_experiment as module
    captured = {}
    class FakeFit:
        mle_retvals = {"converged": True}
        def forecast(self, steps, exog):
            captured["future"] = exog
            return np.zeros(steps)
    def fake_arima(y, **kwargs):
        captured["train"] = kwargs["exog"]
        class FakeModel:
            def fit(self, **kwargs):
                return FakeFit()
        return FakeModel()
    monkeypatch.setattr(module, "ARIMA", fake_arima)
    y = np.arange(60) % 3
    x = np.arange(180).reshape(60, 3)
    fit_with_text(y, x, np.full((2, 3), 1000000), "arima")
    assert np.allclose(captured["train"].mean(axis=0), 0)
    assert np.allclose(captured["train"].std(axis=0), 1)
    first = captured["train"].copy()
    fit_with_text(y, x, np.zeros((2, 3)), "arima")
    assert np.array_equal(first, captured["train"])


def test_strict_experiment_refuses_to_turn_unobserved_history_into_zeros(tmp_path, monkeypatch):
    import json
    import airraid.telegram_experiment as module
    data, reports, output = tmp_path / "data", tmp_path / "reports", tmp_path / "strict"
    (data / "telegram").mkdir(parents=True)
    reports.mkdir()
    (data / "telegram/message_versions.csv.gz").write_bytes(b"placeholder")
    (data / "telegram/checksums.json").write_text("{}")
    from airraid.telegram import KEYWORDS
    (data / "telegram/source_audit.json").write_text(json.dumps({"threat_substrings": KEYWORDS}))
    (reports / "protocol.json").write_text(json.dumps({"validation_start": "2026-09-13", "test_end": "2026-10-10"}))
    for name in ["predictions_unlabelled.csv", "model_selection.csv", "test_predictions.csv"]:
        (reports / name).write_text("frozen")
    d = messages()
    d["observed_at"] = pd.Timestamp("2026-10-12T09:00Z")
    monkeypatch.setattr(module, "verify_checksums", lambda p: None)
    monkeypatch.setattr(module, "load_messages", lambda p: d)
    module.experiment(data, reports, output)
    assert json.loads((output / "status.json").read_text())["scores_produced"] is False
    assert not (output / "daily_features.csv").exists()
    assert not (output / "predictions_unlabelled.csv").exists()
