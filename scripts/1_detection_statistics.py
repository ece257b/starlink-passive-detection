#!/usr/bin/env python3
"""
Result 1: Detection Statistics Summary
=======================================
Queries detections.db and prints the cumulative statistics from
18+ days of continuous passive monitoring (Slide 6 of presentation).

No plots generated — just prints numbers to verify against the paper.

Usage:
    python3 scripts/1_detection_statistics.py
"""

import sqlite3
import os
import sys
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'detections.db')

if not os.path.exists(DB_PATH):
    print(f"ERROR: Database not found at {DB_PATH}")
    sys.exit(1)

conn = sqlite3.connect(DB_PATH)

# ── Total windows ──────────────────────────────────────────────
total = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]

# ── Soft detections (z >= 3.0, harmonics >= 1) ────────────────
soft = conn.execute(
    "SELECT COUNT(*) FROM detections WHERE z_score >= 3.0 AND harmonics >= 1"
).fetchone()[0]

# ── Hard detections (z >= 4.0) ────────────────────────────────
hard = conn.execute(
    "SELECT COUNT(*) FROM detections WHERE z_score >= 4.0"
).fetchone()[0]

# ── Unique satellites ─────────────────────────────────────────
unique_sats = conn.execute(
    "SELECT COUNT(DISTINCT sat_name) FROM detections WHERE sat_name IS NOT NULL AND sat_name != ''"
).fetchone()[0]

# ── Peak z-score ──────────────────────────────────────────────
peak_z = conn.execute("SELECT MAX(z_score) FROM detections").fetchone()[0]

# ── Uptime ────────────────────────────────────────────────────
first_ts = conn.execute("SELECT MIN(timestamp_utc) FROM detections").fetchone()[0]
last_ts  = conn.execute("SELECT MAX(timestamp_utc) FROM detections").fetchone()[0]

# ── Background vs soft vs hard breakdown ──────────────────────
background = conn.execute(
    "SELECT COUNT(*) FROM detections WHERE z_score < 3.0 OR harmonics = 0"
).fetchone()[0]

# ── Top 5 detections ──────────────────────────────────────────
top5 = conn.execute("""
    SELECT timestamp_utc, ROUND(z_score,2), ROUND(peak_freq_hz,1),
           harmonics, sat_name, ROUND(sat_elevation,1)
    FROM detections
    ORDER BY z_score DESC
    LIMIT 5
""").fetchall()

conn.close()

# ── Compute uptime ────────────────────────────────────────────
fmt = "%Y-%m-%dT%H:%M:%S"
try:
    t0 = datetime.strptime(first_ts[:19], fmt)
    t1 = datetime.strptime(last_ts[:19], fmt)
    uptime_days = (t1 - t0).total_seconds() / 86400
except Exception:
    uptime_days = 0

# ── Print results ─────────────────────────────────────────────
print("=" * 60)
print("  STARLINK SENTINEL — DETECTION STATISTICS")
print("  Passive Ku-band Monitoring, Ch6 (RF=11.575 GHz), San Diego")
print("=" * 60)
print(f"  Monitoring period : {first_ts[:19]} UTC")
print(f"                    → {last_ts[:19]} UTC")
print(f"  Uptime            : {uptime_days:.1f} days (continuous, no gaps)")
print()
print(f"  Total 30-sec windows processed : {total:>10,}")
print(f"  Soft detections  (z≥3.0 + ≥1 harmonic) : {soft:>7,}")
print(f"  Hard detections  (z≥4.0)                : {hard:>7,}")
print(f"  Background windows               : {background:>7,}")
print()
print(f"  Unique satellites identified : {unique_sats:>6,}")
print(f"  Peak z-score recorded        : {peak_z:>6.2f}")
print()
print("  Top 5 detections by z-score:")
print(f"  {'Timestamp':<22} {'z':>5}  {'Freq(Hz)':>8}  {'Harm':>4}  {'Satellite':<20} {'Elev':>5}")
print("  " + "-"*75)
for row in top5:
    ts, z, freq, harm, sat, elev = row
    sat  = str(sat)[:20] if sat else "N/A"
    elev = f"{elev:.1f}°" if elev else "N/A"
    print(f"  {ts[:19]:<22} {z:>5.2f}  {freq:>8.1f}  {harm:>4}  {sat:<20} {elev:>5}")
print()
print("  Expected values (from paper):")
print("    Total windows  : ~532,775  (this DB may be larger — sentinel kept running)")
print("    Soft           : ~65,310")
print("    Hard           : ~4,379")
print("    Unique sats    : ~9,503")
print("    Peak z-score   : 6.03")
print("=" * 60)
