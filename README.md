# Passive Starlink Signal Detection

**Akshat Gupta · Luke Wilson — WCSNG Lab, UC San Diego — March 2026**

---

## What We Did

We built a passive radio receiver that listens to Starlink Ku-band downlink signals and detects individual satellites — without decoding any data and without cooperation from SpaceX. The system ran continuously for 20+ days from a rooftop dish in San Diego, logging every detected satellite pass.

**The core finding:** every Starlink satellite's signal has a spectral fingerprint at **750 Hz** — the OFDM frame repetition rate (1 frame every 1.33 ms). By computing the power spectrum of the received signal's power envelope, this fingerprint appears as a narrow peak at 750 Hz plus harmonics. Over 20+ days, we detected 9,630 unique satellites across the entire Starlink constellation.

### Hardware (for reference — not needed to reproduce results)

```
Starlink satellite (11.575 GHz Ku-band)
  → Parabolic dish (rooftop, UCSD San Diego, 32.88°N, −117.23°W)
  → OK2ZAW MK3 LNBF (LO = 9.75 GHz, 60 dB gain, NF = 0.8 dB)
  → Ettus USRP X310 @ 50 MS/s, tuned to IF = 1825 MHz (RF = 11.575 GHz)
  → Starlink Sentinel daemon (see sentinel/)
  → SQLite database + raw IQ snippets
```

---

## Results

| Script | What it shows | Key number |
|--------|--------------|------------|
| `1_detection_statistics.py` | 20+ day cumulative stats | 619k windows, 9,630 unique satellites |
| `2_plot_750hz_fingerprint.py` | The 750 Hz OFDM frame fingerprint | Median peak = 758 Hz (expected: 750 Hz) |
| `3_cell_broadcast_validation.py` | Before vs. after deploying a real Starlink terminal | Mean z-score change = −0.3% (≈ zero) |
| `4_plot_doppler_isolation.py` | Per-satellite Doppler separation in raw IQ | Dominant peak at +6 kHz → +156 m/s radial velocity |

Pre-generated figures are in `figures/` — the scripts regenerate them when run.

---

## How to Run

**Requirements:** Python 3.8+, no SDR hardware needed.

**Step 1 — Clone the repository**
```bash
git clone https://github.com/lukepwilson/starlink-passive-detection.git
cd starlink-passive-detection
```

**Step 2 — Install dependencies**
```bash
pip install numpy scipy matplotlib
```

**Step 3 — Print detection statistics**
```bash
python3 scripts/1_detection_statistics.py
```
Prints a summary table: total monitoring windows, soft/hard detection counts, unique satellites, and peak z-score. No output file generated.

**Step 4 — Plot the 750 Hz fingerprint**
```bash
python3 scripts/2_plot_750hz_fingerprint.py
```
Generates `figures/750hz_fingerprint.png` — the core result showing the Starlink OFDM frame rate spectral peak at 750 Hz across 5 independent detection events.

**Step 5 — Cell broadcast validation**
```bash
python3 scripts/3_cell_broadcast_validation.py
```
Generates `figures/cell_broadcast_validation.png` — compares z-score distributions before and after deploying a real Starlink terminal. The ~0% change proves Starlink uses cell-broadcast architecture.

**Step 6 — Doppler satellite isolation**
```bash
python3 scripts/4_plot_doppler_isolation.py
```
Generates `figures/doppler_isolation.png` — loads the raw IQ snippet and shows individual satellites separated by Doppler shift. Takes ~30 seconds to process.

All output figures are also pre-generated in `figures/` if you just want to view the results without running anything.

---

## Repository Layout

```
starlink-repo/
├── README.md
├── requirements.txt
├── scripts/
│   ├── 1_detection_statistics.py       — print stats from detections.db
│   ├── 2_plot_750hz_fingerprint.py     — 750 Hz fingerprint from .npy envelopes
│   ├── 3_cell_broadcast_validation.py  — before/after terminal deployment
│   └── 4_plot_doppler_isolation.py     — Doppler separation from raw IQ
├── data/
│   ├── detections.db                   — SQLite: 619k detection records, 20+ days
│   ├── envelopes/                      — 5 .npy files: Welch PSD + power envelopes
│   └── snippets/
│       └── snip_z5.2_*.cfile           — Raw IQ: best hard detection (38 MB, 100 ms)
├── figures/                            — Pre-generated output plots
│   ├── 750hz_fingerprint.png
│   ├── cell_broadcast_validation.png
│   └── doppler_isolation.png
└── sentinel/
    └── starlink_sentinel.py            — The always-on detection daemon (read-only)
```

---

## Data Formats

| File | Format | Contents |
|------|--------|----------|
| `detections.db` | SQLite3 | One row per 30-sec window: `timestamp_utc`, `unix_time`, `z_score`, `peak_freq_hz`, `harmonics`, `sat_name`, `sat_elevation`, `confidence` |
| `env_z*.npy` | NumPy dict | Keys: `freqs` (Hz), `psd` (power), `envelope` (30-sec envelope at 10 kHz), `z_score`, `peak_freq_hz`, `timestamp_utc` |
| `snip_z*.cfile` | Binary float32 | Complex64 I/Q @ 50 MS/s. Load with `np.fromfile(path, dtype=np.complex64)` |

---

## The Sentinel (sentinel/starlink_sentinel.py)

The detection daemon that produced all the data. It runs on the machine connected to the USRP and:

1. Captures 1 second of I/Q at a time from the USRP X310
2. Computes the power envelope at 10 kHz resolution
3. Accumulates 30 seconds of envelope, then runs a Welch PSD
4. Detects the 750 Hz Starlink fingerprint using a z-score against the local noise floor
5. Logs every window to `detections.db` with satellite ID from TLE matching (Celestrak)
6. Saves `.npy` envelope files for soft detections (z ≥ 3.0) and `.cfile` IQ snippets for hard detections (z ≥ 4.0)

To run it yourself you need a USRP with UHD (`pip install uhd`) and a Ku-band dish. The config block at the top of the file has everything you'd need to change for a different setup.

---

## Key Results Explained

### 1 — 750 Hz Fingerprint
Starlink uses OFDM with a 1.33 ms frame period (750 Hz). The received signal's power envelope oscillates at this rate, producing a narrow spectral peak at 750 Hz plus harmonics at 1500, 2250 Hz, etc. This is detectable passively because the OFDM structure is fixed by the waveform standard — not encrypted or randomized.

### 2 — Cell Broadcast Validation
On March 4 we deployed a real Starlink Gen3 terminal on the same rooftop as our dish. If SpaceX steered beams toward individual users, activating a terminal would boost the signal we received. Instead, z-score statistics before and after deployment are statistically identical (−0.3% mean change). Starlink broadcasts to geographic cells unconditionally — any receiver in the footprint sees the full waveform.

### 3 — Doppler Satellite Isolation
At 11.575 GHz, each m/s of radial velocity produces 38.6 Hz of Doppler shift. Satellites at different positions in the sky have different radial velocities and therefore appear at different frequencies in a fine-resolution spectrum. The +6 kHz peak in script 4 corresponds to a satellite approaching at ~156 m/s — consistent with orbital mechanics for a satellite at ~83° elevation. This separation enables per-satellite isolation with a single receiver, and differential phase measurement with two.
