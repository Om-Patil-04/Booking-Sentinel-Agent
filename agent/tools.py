import os
import joblib
import numpy as np

MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model/model.pkl")

MEAL_PLAN_MAP = {
    "Breakfast Only": 0.0,
    "Breakfast + Dinner": 1.0,
    "All Meals": 3.0,
    "No Meal Plan": 0.0
}

MARKET_SEGMENT_MAP = {
    "Online": 4.0,
    "Offline": 3.0,
    "Corporate": 2.0,
    "Aviation": 1.0,
    "Complementary": 0.0
}

ROOM_TYPE_MAP = {
    "Room Type 1": 0.0,
    "Room Type 2": 1.0,
    "Room Type 3": 2.0,
    "Room Type 4": 3.0
}


def _safe_float(value, min_val=None, max_val=None, default=0.0) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        val = float(default)

    if min_val is not None and val < min_val:
        val = min_val
    if max_val is not None and val > max_val:
        val = max_val
    return val


def _encode_categorical(value, mapping, default_key=None) -> float:
    if value in mapping:
        return mapping[value]
    if default_key and default_key in mapping:
        return mapping[default_key]
    return next(iter(mapping.values()))


def validate_and_encode(raw_input: dict) -> dict:
    return {
        "lead_time": _safe_float(raw_input.get("lead_time"), min_val=0),
        "no_of_special_requests": _safe_float(
            raw_input.get("no_of_special_requests"), min_val=0, max_val=10
        ),
        "avg_price_per_room": _safe_float(raw_input.get("avg_price_per_room"), min_val=0),
        "arrival_month": _safe_float(raw_input.get("arrival_month"), min_val=1, max_val=12),
        "arrival_date": _safe_float(raw_input.get("arrival_date"), min_val=1, max_val=31),
        "no_of_week_nights": _safe_float(raw_input.get("no_of_week_nights"), min_val=0),
        "no_of_weekend_nights": _safe_float(raw_input.get("no_of_weekend_nights"), min_val=0),
        "type_of_meal_plan": _encode_categorical(
            raw_input.get("type_of_meal_plan"), MEAL_PLAN_MAP, default_key="Breakfast Only"
        ),
        "market_segment_type": _encode_categorical(
            raw_input.get("market_segment_type"), MARKET_SEGMENT_MAP, default_key="Online"
        ),
        "room_type_reserved": _encode_categorical(
            raw_input.get("room_type_reserved"), ROOM_TYPE_MAP, default_key="Room Type 1"
        ),
    }


def predict_booking_risk(data: dict):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle["model"]

    processed = validate_and_encode(data)
    feature_order = [
        "lead_time",
        "no_of_special_requests",
        "avg_price_per_room",
        "market_segment_type",
        "arrival_month",
        "arrival_date",
        "no_of_week_nights",
        "no_of_weekend_nights",
        "type_of_meal_plan",
        "room_type_reserved",
    ]

    X = np.array([processed[col] for col in feature_order]).reshape(1, -1)

    probs = model.predict_proba(X)[0]
    prob_canceled = float(probs[0])
    prob_not_canceled = float(probs[1])

    prediction_label = 0 if prob_canceled >= 0.5 else 1
    prediction_text = "Canceled" if prediction_label == 0 else "Not Canceled"

    return {
        "prediction": prediction_text,
        "prediction_label": prediction_label,
        "probability_canceled": prob_canceled,
        "probability_not_canceled": prob_not_canceled,
        "probability_canceled_percent": round(prob_canceled * 100, 2),
        "probability_not_canceled_percent": round(prob_not_canceled * 100, 2),
        "threshold": 0.5,
    }