from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import random
import requests
import socket
import threading
import joblib
import numpy as np
from datetime import datetime

app = FastAPI()
app.state.use_dummy = True  
app.state.flight_data = {}
app.state.last_data = {}

clf = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

host = '62.45.168.247'
port = 7878

# Dummy + realistic looking flights
dummy_flights = [
    {"flight": "BG101", "lat": 23.81, "long": 90.41, "phase": "Climb", "last_phase": "Cruise"},
    {"flight": "AI223", "lat": 24.21, "long": 88.93, "phase": "Descent", "last_phase": "Cruise"},
    {"flight": "EK582", "lat": 22.82, "long": 91.18, "phase": "Cruise", "last_phase": "Climb"},
    {"flight": "QF109", "lat": 25.02, "long": 89.23, "phase": "Climb", "last_phase": "Climb"},
    {"flight": "QR404", "lat": 24.50, "long": 90.23, "phase": "Descent", "last_phase": "Cruise"}
]

def parse_basestation_line(line):
    parts = line.strip().split(',')
    if len(parts) < 15 or not parts[0].startswith('MSG'):
        return None
    try:
        flight_id = parts[4].strip()
        alt = int(parts[11]) if parts[11] else None
        spd = int(parts[12]) if parts[12] else None
        lat = float(parts[14]) if parts[14] else None
        lon = float(parts[13]) if parts[13] else None
        ts = datetime.strptime(parts[6] + ' ' + parts[7], "%Y/%m/%d %H:%M:%S.%f")
        return flight_id, alt, spd, lat, lon, ts
    except:
        return None

def compute_rates(flight_id, alt, speed, ts):
    if flight_id not in app.state.last_data:
        app.state.last_data[flight_id] = (alt, speed, ts)
        return 0.0, 0.0
    alt_prev, spd_prev, ts_prev = app.state.last_data[flight_id]
    dt = (ts - ts_prev).total_seconds()
    if dt == 0:
        return 0.0, 0.0
    alt_rate = (alt - alt_prev) / dt if alt and alt_prev else 0.0
    spd_rate = (speed - spd_prev) / dt if speed and spd_prev else 0.0
    app.state.last_data[flight_id] = (alt, speed, ts)
    return alt_rate, spd_rate

def start_socket_listener():
    def run():
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((host, port))
                print("✅ Connected to ADS-B source.")
                buffer = ""
                while True:
                    if app.state.use_dummy:
                        continue
                    data = s.recv(4096).decode(errors='ignore')
                    if not data:
                        break
                    buffer += data
                    lines = buffer.split('\n')
                    buffer = lines[-1]
                    for line in lines[:-1]:
                        parsed = parse_basestation_line(line)
                        if not parsed:
                            continue
                        flight_id, alt, spd, lat, lon, ts = parsed
                        alt_rate, spd_rate = compute_rates(flight_id, alt, spd, ts)
                        X_input = np.array([[alt, spd, alt_rate, spd_rate]])
                        X_scaled = scaler.transform(X_input)
                        phase = clf.predict(X_scaled)[0]
                        prev_phase = app.state.flight_data.get(flight_id, {}).get("phase")
                        app.state.flight_data[flight_id] = {
                            "flight": flight_id,
                            "lat": lat,
                            "long": lon,
                            "phase": phase,
                            "last_phase": prev_phase or "None"
                        }
        except Exception as e:
            print("❌ Socket error:", e)

    threading.Thread(target=run, daemon=True).start()

@app.get("/api/flights")
async def get_flight_data():
    if app.state.use_dummy:
        for flight in dummy_flights:
            flight["lat"] += random.uniform(-0.01, 0.01)
            flight["long"] += random.uniform(-0.01, 0.01)
        return dummy_flights
    else:
        return list(app.state.flight_data.values())

@app.get("/")
async def dashboard():
    return FileResponse("dashboard.html")

@app.get("/api/toggle")
async def toggle_data(request: Request):
    app.state.use_dummy = not app.state.use_dummy
    return {"using_dummy": app.state.use_dummy}

@app.get("/api/geocode")
async def geocode(query: str):
    """Get coordinates for a given city/location name."""
    url = f"https://nominatim.openstreetmap.org/search?format=json&q={query}"
    headers = {"User-Agent": "Flight-Phase-Dashboard/1.0"}
    try:
        response = requests.get(url, headers=headers)
        results = response.json()
        if results:
            lat = float(results[0]["lat"])
            lon = float(results[0]["lon"])
            return {"lat": lat, "long": lon, "display_name": results[0].get("display_name", query)}
        return JSONResponse({"error": "Location not found"}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

start_socket_listener()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)