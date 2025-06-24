import subprocess
import re
import time
import statistics
from datetime import datetime
from collections import defaultdict, deque

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

clf = joblib.load("flight_phase_model.pkl")
scaler = joblib.load("scaler.pkl")

print("Connecting to SDR at 62.45.168.247:7878 and decoding + classifying ADS-B flight phases...")

# Launch IQ stream pipeline
nc_proc = subprocess.Popen(
    ['nc', '62.45.168.247', '7878'],
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL
)

decoder_proc = subprocess.Popen(
    ['dump1090-mutability', '--ifile', '-', '--iformat', 'UC8', '--net'],
    stdin=nc_proc.stdout,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

# Regex patterns
altitude_pattern = re.compile(r'Altitude:\s+(\d+)\s+ft')
speed_pattern = re.compile(r'Speed:\s+(\d+)\s+kt')
track_pattern = re.compile(r'Track:\s+(\d+)')
callsign_pattern = re.compile(r'Flight:\s+(\w+)')

# Storage for tracking
altitudes, speeds, tracks, callsigns, timestamps = [], [], [], [], []
history = defaultdict(lambda: deque(maxlen=2))  # flight → deque of (time, alt, speed)

# Helpers
def estimate_temp(alt_ft):
    alt_km = alt_ft * 0.0003048
    if alt_km <= 11:
        return 15.0 - (6.5 * alt_km)
    elif alt_km <= 20:
        return -56.5
    return -56.5 + (alt_km - 20)

def pressure_altitude_to_pressure(alt_ft):
    return 1013.25 * ((1 - 0.0065 * alt_ft * 0.0003048 / 288.15) ** 5.255)

def calculate_wind_direction(tracks):
    return (statistics.mean(tracks) + 180) % 360 if len(tracks) > 2 else None

line_count = 0
start_time = time.time()

try:
    for line in decoder_proc.stdout:
        line_count += 1
        now = datetime.utcnow()

        if not any(k in line for k in ['Altitude:', 'Speed:', 'Track:', 'Flight:']):
            continue

        # Extract all fields
        alt_match = altitude_pattern.search(line)
        spd_match = speed_pattern.search(line)
        trk_match = track_pattern.search(line)
        cs_match  = callsign_pattern.search(line)

        if not (alt_match and spd_match and cs_match):
            continue

        alt = int(alt_match.group(1))
        spd = int(spd_match.group(1))
        cs = cs_match.group(1)
        trk = int(trk_match.group(1)) if trk_match else None

        # Save basic stats
        altitudes.append(alt)
        speeds.append(spd)
        timestamps.append(now)
        callsigns.append(cs)
        if trk: tracks.append(trk)

        # Add to history
        history[cs].append((now, alt, spd))
        if len(history[cs]) < 2:
            continue

        # Calculate delta
        t1, a1, s1 = history[cs][0]
        t2, a2, s2 = history[cs][-1]
        dt = (t2 - t1).total_seconds()
        if dt == 0:
            continue
        alt_rate = (a2 - a1) / dt
        mph_rate = (s2 - s1) / dt

        # Predict phase
        feat = [[alt, spd, alt_rate, mph_rate]]
        phase = clf.predict(scaler.transform(feat))[0]

        print(f"[{cs}] Alt: {alt} ft | Spd: {spd} kt | Δalt/s: {alt_rate:.2f} | Phase: {phase}")

        if len(altitudes) >= 20:
            break

        if line_count % 100 == 0:
            print(f"Processed {line_count} lines in {time.time() - start_time:.1f}s...")

    # Summary
    print("\n" + "="*60)
    print("FINAL AIRCRAFT WEATHER + PHASE SUMMARY")
    print("="*60)

    if altitudes:
        avg_alt = statistics.mean(altitudes)
        print(f"Altitude: avg={avg_alt:.0f} ft, min={min(altitudes)} ft, max={max(altitudes)} ft")
        if len(altitudes) > 1:
            print(f"   Std Dev: {statistics.stdev(altitudes):.0f} ft")
        print(f"Temp @ avg alt: {estimate_temp(avg_alt):.1f} °C")
        print(f"Pressure @ avg alt: {pressure_altitude_to_pressure(avg_alt):.1f} hPa")

    if tracks:
        print(f"Wind Dir Est.: {calculate_wind_direction(tracks):.0f}°")
    if speeds:
        print(f"Avg Speed: {statistics.mean(speeds):.0f} kt")

    print(f"\n✈️ Unique Flights: {len(set(callsigns))}")
    if len(timestamps) > 1:
        print(f"🕒 Duration: {(timestamps[-1] - timestamps[0]).total_seconds():.1f}s")
    print("="*60)

except KeyboardInterrupt:
    print("\n❌ Interrupted by user.")
except Exception as e:
    print(f"\n❌ Error: {e}")
finally:
    nc_proc.kill()
    decoder_proc.kill()
    print("\n🔌 SDR connection closed.")
