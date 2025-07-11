# Flight Phase Dashboard

A real-time web dashboard that classifies and visualizes **aircraft flight phases** (Climb, Cruise, Descent) using live ADS-B data or simulated dummy data. Built using **FastAPI**, **scikit-learn**, **Leaflet.js**, and **CSS**, it features a global map, live updates, ML-based classification, and user location search with geocoding.

---

## Features

- 🔮 **Flight Phase Classification**  
  Classifies aircraft into Climb, Cruise, or Descent phases using altitude, speed, and their rates of change with a trained machine learning model.

- 🌐 **Real-Time ADS-B Data**  
  Connects to a remote TCP feed from an ADS-B receiver (e.g., `dump1090`) and updates the map live.

- 🧪 **Dummy Mode**  
  Switch to simulated flights when real-time feed is unavailable. Toggle instantly with a button.

- 🗺️ **Interactive Global Map**  
  Leaflet-based map with color-coded aircraft:
  - 🟢 Climb
  - 🔵 Cruise
  - 🔴 Descent

- 🔍 **Search Location**  
  Use the built-in geocoder to search for any city or airport to quickly zoom the map.

- ⚡ **FastAPI Backend**  
  Handles socket streaming, ML predictions, toggle logic, and REST endpoints.

---

## 🧠 How It Works

- Connects to a BaseStation-compatible TCP server (e.g., `dump1090-mutability`) streaming real-time ADS-B messages.
- Parses each ADS-B message to extract:
  - Flight ID
  - Altitude
  - Ground Speed
  - Latitude / Longitude
  - Timestamp
- Computes:
  - `alt_rate` — change in altitude per second
  - `speed_rate` — change in speed per second
- The four time-series features `[altitude, speed, alt_rate, speed_rate]` are processed by a **pre-trained LSTM (Long Short-Term Memory)** model.
- The LSTM is trained to predict **flight phase**: `Climb`, `Cruise`, or `Descent`.

### 🧠 Why LSTM?

- Unlike static classifiers like Random Forest, **LSTM** captures **temporal patterns** and **flight dynamics over time**.
- Aircraft transitions between phases (e.g., climb to cruise) are sequential and depend on historical context.
- LSTM improves **generalization**, **sequence awareness**, and **accuracy**, especially for overlapping phases or noisy telemetry.

> During training, the LSTM model achieved >99% accuracy on a labeled ADS-B dataset with temporal windowing.

### 🧪 Dummy Mode

When live mode is turned off (default), the app switches to simulated aircraft data with randomized positions, altitude, and behavior — useful for offline testing, demos, or in case of feed unavailability.

## 📁 Directory Structure

```bash
.
├── app.py                  # FastAPI backend with live ADS-B socket handler
├── dashboard.html          # Frontend dashboard (Leaflet + Tailwind UI)
├── model.pkl               # Trained ML model for phase classification
├── scaler.pkl              # StandardScaler used during training
├── README.md               # This file
├── test.py                 # Code for testing.
├── aircraft-data_nov_dec.csv #Data used for training source: [Kaggle](https://www.kaggle.com/datasets/brianwarner/aircraft-data-from-nov-2022-through-dec-31-2022)
```

## ⚙️ Setup & Run

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/flight-phase-dashboard.git
cd flight-phase-dashboard
