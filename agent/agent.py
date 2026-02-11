import json
import os
import shutil
import subprocess
import logging
from ollama import chat
from .tools import predict_booking_risk
from .prompts import SYSTEM_PROMPT, EXTRACTION_PROMPT

os.environ.setdefault("OLLAMA_HOST", "http://127.0.0.1:11434")


class HotelAgent:
    def __init__(self, model: str = "mistral:7b-instruct"):
        self.model = model
        self.memory = []
        self._ensure_model_available()

    def _ensure_model_available(self) -> None:
        auto_pull = os.getenv("AUTO_PULL_OLLAMA_MODEL", "true").lower() in {
            "1",
            "true",
            "yes",
        }
        if not auto_pull:
            return

        ollama_cli = os.getenv("OLLAMA_CLI") or shutil.which("ollama")
        if not ollama_cli:
            raise RuntimeError(
                "Ollama CLI not found. Install Ollama and add it to PATH, "
                "or set OLLAMA_CLI to the full path, or disable auto-pull with "
                "AUTO_PULL_OLLAMA_MODEL=false."
            )

        try:
            result = subprocess.run(
                [ollama_cli, "show", self.model],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if result.returncode != 0:
                subprocess.run([ollama_cli, "pull", self.model], check=True)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Ollama CLI not found. Install Ollama and ensure it is on PATH."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Failed to pull model '{self.model}'. Ensure Ollama is running."
            ) from exc

    def _extract_features(self, user_message: str) -> dict:
        logging.info("Agent: extracting structured features from user message.")
        prompt = EXTRACTION_PROMPT.format(message=user_message)
        response = chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        content = response["message"]["content"].strip()
        try:
            features = json.loads(content)
            logging.info("Agent: extracted features from LLM successfully.")
            return features
        except json.JSONDecodeError:
            logging.warning("Agent: LLM returned invalid JSON, using empty defaults.")
            return {}

    def _interpret_risk(self, prediction: dict) -> str:
        prob = prediction.get("probability_canceled")
        if prob is None:
            return "I couldn’t determine the risk because the predictor returned an unexpected response."

        if prob >= 0.7:
            return "High risk of cancellation."
        if prob >= 0.4:
            return "Moderate risk of cancellation."
        return "Low risk of cancellation."

    def run(self, user_message: str, ui_data: dict | None = None):
        logging.info("Agent: starting request pipeline.")
        self.memory.append({"role": "user", "content": user_message})

        llm_features = self._extract_features(user_message)
        ui_data = ui_data or {}

        logging.info("Agent: merging UI inputs with LLM features.")
        merged_features = {
            **llm_features,
            **{k: v for k, v in ui_data.items() if v not in (None, "", [])}
        }
        logging.info("Agent: merged features ready: %s", merged_features)

        logging.info("Agent: running prediction model.")
        prediction = predict_booking_risk(merged_features)
        logging.info("Agent: model prediction result: %s", prediction)

        risk_summary = self._interpret_risk(prediction)
        logging.info("Agent: risk summary: %s", risk_summary)

        logging.info("Agent: generating final response with LLM.")
        followup_prompt = (
            "You are a premium hotel booking assistant. "
            "Explain the risk outcome in polished, confident language. "
            "Be concise. Suggest one smart next step.\n\n"
            f"Prediction: {prediction}\n"
            f"Risk summary: {risk_summary}\n"
            f"User message: {user_message}\n"
            f"User-provided inputs: {merged_features}"
        )

        response = chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": followup_prompt},
            ],
        )

        logging.info("Agent: response generated successfully.")
        return {
            "message": response["message"]["content"],
            "prediction": prediction,
            "risk_summary": risk_summary,
        }