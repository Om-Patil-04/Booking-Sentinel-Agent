# Booking Sentinel Agent 
**Continuation of the Hotel-Management-System**

This repository extends the **Hotel-Management-System** project by adding an AI-driven agent that integrates:

- a **local pre-trained `.pkl` model** from the original pipeline, and  
- **Mistral 7B Instruct via Ollama** for structured extraction and response generation.

It is designed as the **AI layer** on top of your existing hotel decision intelligence system.

---

## 📁 Project Structure

```
hotel-ai-agent/
├── agent/
│   ├── __init__.py
│   ├── agent.py
│   ├── prompts.py
│   └── tools.py
├── docs/
│   ├── agent-chat-1.png
│   └── agent-chat-2.png
├── models/
│   └── best_model/
│       └── model.pkl
├── templates/
│   └── index.html
├── app.py
├── Dockerfile
├── README.md
└── requirements.txt
```

---

## ✅ Prerequisites

### 1) Install Ollama  
https://ollama.com

### 2) Pull the Mistral model

```
ollama pull mistral:7b-instruct
```

---

## 📌 Model Location

By default, place your trained model here:

```
models/best_model/model.pkl
```

You can also override the path using an environment variable:

**Linux / macOS**
```
export MODEL_PATH=/path/to/your/model.pkl
```

**Windows PowerShell**
```
$env:MODEL_PATH="C:\path\to\your\model.pkl"
```

---

## ⚙️ Optional Environment Variables

If Ollama is not in your PATH:

```
$env:OLLAMA_CLI="C:\Program Files\Ollama\ollama.exe"
```

To skip auto-pulling the Ollama model:

```
$env:AUTO_PULL_OLLAMA_MODEL="false"
```

---

## ▶️ Run the API

```
uvicorn app:app --host 0.0.0.0 --port 8080
```

---

## 🔌 API Usage

**Endpoint:** `POST /chat`

**Example request:**
```json
{
  "message": "I'm planning a 3-night stay in June. Is this booking risky?"
}
```

---

## 🧠 Notes

- Input schema and categorical mappings are aligned with the original **Hotel-Management-System** pipeline.
- This repository acts as an AI decision layer on top of the existing ML model.
- You can replace the model path at runtime with `MODEL_PATH`.

---

## 📎 Related Repository

**Hotel-Management-System**  
(Original training and pipeline repository)