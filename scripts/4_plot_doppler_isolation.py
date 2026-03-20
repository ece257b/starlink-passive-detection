#!/usr/bin/env python3
"""
Result 4: Doppler Satellite Isolation
=======================================
Loads a raw IQ hard-detection snippet (.cfile) and computes a
fine-resolution power spectrum to show individual satellites
separated by their unique Doppler frequency shift.

Each overhead satellite moves at a different velocity relative to
the ground receiver, shifting its signal by a different amount.
At 11.575 GHz, 1 m/s of radial velocity = 38.6 Hz of Doppler shift.
Satellites typically separate by >30 kHz, enabling clean isolation.

This is the mechanism used for per-satellite isolation in the
planned two-receiver localization experiment (Slide 9).

NOTE ON TERMINOLOGY:
    The slide calls these peaks "CW pilot tones." Whether they are
    strictly the pilot tones characterized by Kozhaya et al. 2025
    (9 tones, ~43.9 kHz spacing, power dropped ~30 dB post-2023)
    or another persistent OFDM spectral artifact is an open question.
    What is confirmed: they shift with Doppler exactly as orbital
    mechanics predicts and enable satellite isolation.

Usage:
    python3 scripts/4_plot_doppler_isolation.py

Output:
    doppler_isolation.png
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys

SNIPPET_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data', 'snippets')
OUTPUT       = os.path.join(os.path.dirname(__file__), '..', 'figures', 'doppler_isolation.png')

SAMPLE_RATE  = 50e6        # 50 MS/s
RF_FREQ      = 11.575e9    # Hz  (Ku-band after 9.75 GHz LNB downconversion)
C            = 3e8         # m/s
HZ_PER_MPS   = RF_FREQ / C  # Doppler scale: 38.6 Hz per m/s at 11.575 GHz

# ── Load .cfile ───────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(SNIPPET_DIR, '*.cfile')))
if not files:
    print(f"ERROR: No .cfile found in {SNIPPET_DIR}")
    sys.exit(1)

# Pick highest z-score file
def z_from_name(f):
    try:
        return float(os.path.basename(f).split('_z')[1].split('_')[0])
    except Exception:
        return 0.0

cfile = max(files, key=z_from_name)
z_val = z_from_name(cfile)
fname = os.path.basename(cfile)

print(f"Loading: {fname}  (z={z_val:.1f})")
iq = np.fromfile(cfile, dtype=np.complex64)
n_samples = len(iq)
duration_ms = n_samples / SAMPLE_RATE * 1000
print(f"Samples: {n_samples:,}   Duration: {duration_ms:.0f} ms")

# ── Power-spectrum average (incoherent, 100 Hz/bin) ───────────
# FFT length for 100 Hz/bin at 50 MS/s:
N_FFT = int(SAMPLE_RATE / 100)   # 500,000 samples → 100 Hz per bin

N_WINDOWS = n_samples // N_FFT
if N_WINDOWS < 1:
    print(f"ERROR: File too short for 100 Hz resolution. Need ≥{N_FFT} samples.")
    sys.exit(1)

N_WINDOWS = min(N_WINDOWS, 50)   # cap at 50 windows (5 seconds of data)
print(f"Averaging {N_WINDOWS} × {N_FFT/1e6:.1f} M-sample windows (100 Hz/bin) ...")

power = np.zeros(N_FFT, dtype=np.float64)
for i in range(N_WINDOWS):
    chunk = iq[i * N_FFT : (i + 1) * N_FFT]
    power += np.abs(np.fft.fft(chunk)) ** 2

power /= N_WINDOWS
power  = np.fft.fftshift(power)
freqs_hz  = np.fft.fftshift(np.fft.fftfreq(N_FFT, d=1.0 / SAMPLE_RATE))
freqs_khz = freqs_hz / 1e3

# ── Normalize to noise floor ──────────────────────────────────
noise_mask = (np.abs(freqs_khz) > 50) & (np.abs(freqs_khz) < 450)
noise_mean = np.mean(power[noise_mask])
magnitude_db = 10 * np.log10(power / max(noise_mean, 1e-30))

# ── Find dominant peak (exclude DC ±5 kHz) ────────────────────
plot_mask = np.abs(freqs_khz) <= 400
no_dc     = plot_mask & (np.abs(freqs_khz) > 5)

peak_idx  = np.argmax(magnitude_db[no_dc])
peak_f_khz = freqs_khz[no_dc][peak_idx]
peak_db    = magnitude_db[no_dc][peak_idx]
v_radial   = peak_f_khz * 1e3 / HZ_PER_MPS   # m/s

# ── Plot ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor('#0d1b2a')
ax.set_facecolor('#0d1b2a')
ax.tick_params(colors='white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.title.set_color('white')
for spine in ax.spines.values():
    spine.set_edgecolor('#444')

ax.plot(freqs_khz[plot_mask], magnitude_db[plot_mask],
        color='steelblue', linewidth=0.5, alpha=0.85)

ax.axhline(0, color='#888', linestyle='--', linewidth=0.8, label='Noise floor (0 dB)')
ax.axhline(3, color='orange', linestyle='--', linewidth=0.8, label='3 dB threshold')

# Annotate dominant peak
sign = '+' if peak_f_khz >= 0 else ''
ax.annotate(
    f'{sign}{peak_f_khz:.0f} kHz\n({sign}{v_radial:.0f} m/s)',
    xy=(peak_f_khz, peak_db),
    xytext=(peak_f_khz + 25, peak_db + 0.8),
    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
    color='red', fontsize=11, fontweight='bold'
)

ax.set_xlabel('Frequency offset from IF center (kHz)', color='white')
ax.set_ylabel('Power above noise floor (dB)', color='white')
ax.set_title(
    f'Doppler-Separated Satellite Features — Raw IQ  ({fname})\n'
    f'Ch6 RF=11.575 GHz  |  50 MS/s  |  {N_WINDOWS}-window power average  |  100 Hz/bin\n'
    f'Doppler scale: {HZ_PER_MPS:.1f} Hz per m/s  →  1 km/s ≈ 38.6 kHz offset',
    color='white'
)
ax.legend(facecolor='#1a2a3a', edgecolor='#444', labelcolor='white')
ax.set_xlim(-400, 400)
ax.margins(y=0.20)
ax.grid(True, alpha=0.2, color='#555')

plt.tight_layout(pad=2.5)
plt.savefig(OUTPUT, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved: {OUTPUT}")

# ── Print result ──────────────────────────────────────────────
print()
print("=" * 60)
print("  KEY RESULT: DOPPLER SATELLITE ISOLATION")
print("=" * 60)
print(f"  Source file     : {fname}")
print(f"  Detection z     : {z_val:.1f}")
print(f"  FFT resolution  : 100 Hz/bin  ({N_WINDOWS} averaged windows)")
print()
print(f"  Dominant peak   : {sign}{peak_f_khz:.1f} kHz")
print(f"  Radial velocity : {sign}{v_radial:.0f} m/s")
print(f"    (At 11.575 GHz, Doppler scale = {HZ_PER_MPS:.1f} Hz per m/s)")
print()
print("  For localization:")
print("    Both receivers filter at this Doppler frequency →")
print("    both see only this satellite's signal → measure")
print("    differential phase → fit near-field model →")
print("    solve for satellite azimuth + range.")
print("=" * 60)
