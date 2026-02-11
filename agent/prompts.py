SYSTEM_PROMPT = (
    "You are a hotel booking risk assistant. "
    "Extract booking details from the user message, "
    "run the prediction tool, interpret the risk, "
    "and provide actionable guidance."
)

EXTRACTION_PROMPT = (
    "Extract booking details from this message and return ONLY valid JSON "
    "with the following required keys:\n"
    "- lead_time (number)\n"
    "- no_of_special_requests (number)\n"
    "- avg_price_per_room (number)\n"
    "- market_segment_type (string)\n"
    "- arrival_month (number)\n"
    "- arrival_date (number)\n"
    "- no_of_week_nights (number)\n"
    "- no_of_weekend_nights (number)\n"
    "- type_of_meal_plan (string)\n"
    "- room_type_reserved (string)\n\n"
    "If a value is missing, infer a reasonable default.\n\n"
    "Message:\n{message}"
)