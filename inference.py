import socket
import joblib
import numpy as np
from datetime import datetime

clf = joblib.load("model.pkl")        
scaler = joblib.load("scaler.pkl")

host = '62.45.168.247'
port = 7878
print("Connecting to ADS-B API at {}:{}...".format(host, port))

# Keep track of previous altitude/speed for rate computation
last_flight_data = {}

def parse_basestation_line(line):
    """Parse BaseStation format: return flight_id, alt, speed, timestamp"""
    parts = line.strip().split(',')
    if len(parts) < 15 or not parts[0].startswith('MSG'):
        return None

    try:
        flight_id = parts[4].strip()
        altitude = int(parts[11]) if parts[11] else None
        speed = int(parts[12]) if parts[12] else None
        timestamp = datetime.strptime(parts[6] + ' ' + parts[7], "%Y/%m/%d %H:%M:%S.%f")
        return flight_id, altitude, speed, timestamp
    except Exception:
        return None

def compute_rates(flight_id, alt, speed, timestamp):
    if flight_id not in last_flight_data:
        last_flight_data[flight_id] = (alt, speed, timestamp)
        return 0.0, 0.0

    alt_prev, speed_prev, time_prev = last_flight_data[flight_id]
    time_diff = (timestamp - time_prev).total_seconds()
    if time_diff == 0:
        return 0.0, 0.0

    alt_rate = (alt - alt_prev) / time_diff if alt and alt_prev else 0.0
    speed_rate = (speed - speed_prev) / time_diff if speed and speed_prev else 0.0
    last_flight_data[flight_id] = (alt, speed, timestamp)
    return alt_rate, speed_rate

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((host, port))
    print("Connected. Listening for ADS-B messages...")

    buffer = ""
    while True:
        data = s.recv(4096).decode(errors='ignore')
        if not data:
            break

        buffer += data
        lines = buffer.split('\n')
        buffer = lines[-1]  # keep incomplete line

        for line in lines[:-1]:
            parsed = parse_basestation_line(line)
            if not parsed:
                continue

            flight_id, alt, speed, timestamp = parsed
            alt_rate, speed_rate = compute_rates(flight_id, alt, speed, timestamp)

            # Features
            X_input = np.array([[alt, speed, alt_rate, speed_rate]])
            X_scaled = scaler.transform(X_input)
            phase = clf.predict(X_scaled)[0]

            # Display info
            print(f"[✈️] {flight_id:8s} | Alt: {alt:5d} ft | Spd: {speed:3d} kt | ΔAlt: {alt_rate:6.2f} ft/s | Phase: {phase}")
