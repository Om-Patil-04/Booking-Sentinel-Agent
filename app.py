from flask import Flask, request, jsonify, render_template
from agent.agent import HotelAgent
import logging

app = Flask(__name__)
agent = HotelAgent()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

@app.route("/", methods=["GET"])
def home():
    logging.info("Serving home page.")
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    logging.info("New /chat request received")
    data = request.get_json(silent=True) or {}
    logging.info("Parsed JSON payload from client.")

    user_input = (data.get("message") or "").strip()
    if not user_input:
        logging.warning("No message provided by user.")
        return jsonify({"message": "Please provide a non-empty 'message' field."}), 400

    logging.info("User message validated.")
    logging.info("User message: %s", user_input)
    logging.info("UI payload: %s", data)

    logging.info("Calling HotelAgent to process request...")
    result = agent.run(user_input, data)

    logging.info("HotelAgent returned response.")
    logging.info("Agent message generated.")
    logging.info("Prediction payload ready.")

    logging.info("Agent response: %s", result.get("message", ""))
    logging.info("Prediction: %s", result.get("prediction", {}))

    logging.info("=== /chat request complete ===")
    return jsonify(result)

if __name__ == "__main__":
    logging.info("Starting Flask server at http://127.0.0.1:8000")
    app.run(host="127.0.0.1", port=8000, debug=True)