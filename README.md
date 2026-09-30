# Smart Parking Optimization using IoT, DNN and GenAI

An integrated smart parking management system that monitors parking slot occupancy using IoT sensors, predicts parking availability patterns using Deep Neural Networks (LSTM), optimizes slot allocation, and provides natural-language assistance using Generative AI.

---

## 🎯 Project Overview & Goals

1. **IoT Parking Slot Monitoring:** Live occupancy tracking for 4 parking slots (`P1`, `P2`, `P3`, `P4`) using ultrasonic distance sensors.
2. **DNN-based Occupancy Prediction:** Time-series forecasting using an LSTM model to predict availability patterns.
3. **Parking Slot Optimization:** Rule-based algorithm to recommend the best slot based on proximity, walking distance, and availability.
4. **GenAI Parking Assistant:** Natural-language conversational interface to answer driver queries (e.g., *"Is slot P2 available?"*, *"Where should I park?"*).
5. **Interactive Dashboard:** Streamlit dashboard providing real-time parking visualization, analytics, and assistant chat.

---

## 🏗️ System Architecture

```text
ESP32 + HC-SR04 Sensors (or IoT Simulator)
                    ↓
        Parking Occupancy Data
                    ↓
             Backend (FastAPI)
                    ↓
      SQLite Database & Historical CSV
                    ↓
         LSTM Occupancy Prediction
                    ↓
          Slot Recommendation Engine
                    ↓
         GenAI Natural Language Assistant
                    ↓
            Streamlit Dashboard
```

---

## 📁 Project Structure

```text
smart-parking/
│
├── backend/                  # FastAPI web server and database models
│   ├── main.py
│   ├── database.py
│   └── models.py
│
├── data/                     # Raw and processed datasets
│   ├── parking_data.csv      # Sensor occupancy readings
│   └── processed_data.csv    # Preprocessed data for model training
│
├── dnn/                      # Deep Neural Network (LSTM) pipeline
│   ├── preprocess.py         # Feature engineering & scaling
│   ├── train_lstm.py         # LSTM model training
│   └── predict.py            # Future occupancy inference
│
├── optimization/             # Slot recommendation logic
│   └── slot_optimizer.py
│
├── genai/                    # LLM / GenAI conversational interface
│   └── parking_assistant.py
│
├── dashboard/                # User & Admin visualization
│   └── app.py                # Streamlit web application
│
├── iot/                      # Hardware code & software simulation
│   └── simulator.py          # Simulated HC-SR04 ultrasonic sensors
│
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 🛠️ Hardware Setup (College Project)

- **Available:** Raspberry Pi 5, Raspberry Pi Camera Module 3 (reserved for future optional vision enhancements)
- **Target IoT Setup:**
  - ESP32 Microcontroller
  - 4x HC-SR04 Ultrasonic Distance Sensors
  - LEDs (Green = Empty, Red = Occupied)
  - Breadboard & Jumper wires

---

## 🚀 Getting Started (Phase 1)

### 1. Prerequisites
- Python 3.10+ installed on your system.

### 2. Activate Virtual Environment
On Windows (PowerShell):
```powershell
.\venv\Scripts\Activate.ps1
```
*(Or on Command Prompt: `.\venv\Scripts\activate.bat`)*

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Run the IoT Simulator
Generate 7 days of simulated sensor data for slots P1–P4:
```powershell
python iot/simulator.py --days 7 --interval 15
```

To run in **live streaming mode** (simulates real-time sensor updates every 3 seconds):
```powershell
python iot/simulator.py --live --live-interval 3
```

Generated data will be written to `data/parking_data.csv`.

### 5. Start the FastAPI Backend (Phase 2)
```powershell
uvicorn backend.main:app --reload
```
- Interactive Swagger UI documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Alternative ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### 6. Run Simulator with HTTP Backend Transmission
Stream live simulated sensor readings and automatically push them into SQLite database via REST API:
```powershell
python iot/simulator.py --live --live-interval 3 --send-api
```

### 7. Run Data Preprocessing (Phase 3)
```powershell
python dnn/preprocess.py --window 8 --split 0.8
```

### 8. Train the LSTM Model (Phase 3)
```powershell
python dnn/train_lstm.py --epochs 25 --batch-size 32
```

### 9. Predict Future Occupancy for Parking Slots (Phase 3)
```powershell
# Predict for a specific slot
python dnn/predict.py --slot P1

# Predict for all slots (P1 to P4)
python dnn/predict.py
```

### 10. Run Parking Slot Optimizer (Phase 4)
```powershell
python optimization/slot_optimizer.py
```

### 11. Configure Google Gemini API & Run Assistant
#### A. Configure API Key
1. Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/).
2. Create or edit `.env` in the project root:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```
> 🔒 **Security Notice:**
> - Never hardcode or commit your `GEMINI_API_KEY` to git or GitHub.
> - The `.env` file is already listed in `.gitignore` to prevent accidental credential leakage.
> - The API key is securely consumed exclusively by the Python backend and is **never** sent to the browser or React frontend.
> - If no key is set or the API is unreachable, the system automatically falls back to deterministic template mode without interruption.

#### B. CLI Query Test
```powershell
# Run a specific query
python genai/parking_assistant.py --query "Where should I park?"

# Run all test queries in suite
python genai/parking_assistant.py --test-all
```

### 12. Run the React Dashboard (Phase 5)
```powershell
cd frontend
npm install
npm run dev
```
Access the dashboard at [http://localhost:5173](http://localhost:5173).

---

## 🖥️ Complete End-to-End Demonstration (3-Terminal Workflow)

To demonstrate the full integrated pipeline in action:

### Terminal 1: Start Backend Server
```powershell
.\venv\Scripts\Activate.ps1
uvicorn backend.main:app --reload
```
*Starts the FastAPI backend at [http://127.0.0.1:8000](http://127.0.0.1:8000).*

### Terminal 2: Start IoT Sensor Simulator with HTTP Push
```powershell
.\venv\Scripts\Activate.ps1
python iot/simulator.py --live --live-interval 3 --send-api
```
*Simulates real-time ultrasonic sensor toggles and streams them to the FastAPI backend and SQLite database every 3 seconds.*

### Terminal 3: Start React Frontend
```powershell
cd frontend
npm run dev
```
*Opens the interactive React dashboard at [http://localhost:5173](http://localhost:5173) with live slot occupancy indicators, optimizer recommendations, LSTM occupancy charts, and the GenAI parking assistant.*

---

## 📡 REST API Endpoints (Phase 2 & Phase 5)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/sensor/update` | Ingest sensor telemetry and update slot state |
| `GET` | `/api/slots` | Fetch real-time status of all slots (P1–P4) |
| `GET` | `/api/slots/{slot_id}` | Fetch current status of a specific slot |
| `GET` | `/api/analytics/history` | Query historical sensor logs (optional `?slot_id=P1&limit=50`) |
| `GET` | `/api/optimizer/recommendation` | Fetch optimal slot recommendation and candidate rankings |
| `GET` | `/api/dnn/predictions` | Fetch latest LSTM occupancy probability forecasts |
| `POST` | `/api/genai/chat` | Query the GenAI natural language parking assistant |
| `GET` | `/docs` | Swagger interactive API testing UI |

---

## 📋 Roadmap by Phases

- **Phase 1 (Completed):** Project structure, virtual environment setup, initial requirements, documentation, and IoT sensor simulator with dataset generation.
- **Phase 2 (Completed):** FastAPI backend, SQLite database (`parking_slots` and `sensor_logs`), Pydantic validation, REST APIs, and simulator HTTP streaming.
- **Phase 3 (Completed):** Time-series sliding-window preprocessing, chronological train/test split, simple explainable LSTM network training, comprehensive evaluation, and slot prediction CLI/module.
- **Phase 4 (Completed):** Multi-factor parking slot utility optimization, proximity normalization, strict constraint enforcement, and GenAI natural-language assistant with zero-dependency template fallback and API key security.
- **Phase 5 (Completed — FINAL PHASE):** High-performance React web dashboard with Vite, glassmorphism dark mode UI, real-time slot monitoring, optimizer ranking tables, DNN analytics & LSTM prediction charts, and GenAI interactive parking assistant.
