#!/usr/bin/env python3
"""
Result 2: The 750 Hz Pilot Tone Fingerprint
============================================
Loads saved power-envelope files (.npy) from hard/soft detections
and plots the Welch PSD showing the 750 Hz OFDM frame repetition
rate peak — the core spectral fingerprint of Starlink downlink.

This reproduces the Key Result slide (Slide 7) of the presentation.

What you are looking at:
    The PSD is computed on |x(t)|^2 (the signal's power envelope),
    NOT on the raw IQ directly. Because the OFDM frame repeats every
    1.33 ms (= 750 Hz), the power envelope oscillates periodically
    at 750 Hz, producing a narrow spectral line in the PSD at 750 Hz
    plus harmonics at 1500 Hz, 2250 Hz, etc.

Usage:
    python3 scripts/2_plot_750hz_fingerprint.py

Output:
    750hz_fingerprint.png
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys

ENVELOPE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'envelopes')
OUTPUT       = os.path.join(os.path.dirname(__file__), '..', 'figures', '750hz_fingerprint.png')

# ── Load envelope files ────────────────────────────────────────
files = sorted(glob.glob(os.path.join(ENVELOPE_DIR, '*.npy')))
if not files:
    print(f"ERROR: No .npy files found in {ENVELOPE_DIR}")
    sys.exit(1)

print(f"Found {len(files)} envelope files — loading...")

detections = []
for f in files:
    d = np.load(f, allow_pickle=True).item()
    detections.append({
        'freqs'    : d['freqs'],
        'psd'      : d['psd'],
        'z_score'  : float(d['z_score']),
        'peak_freq': float(d['peak_freq_hz']),
        'timestamp': str(d['timestamp_utc']),
        'filename' : os.path.basename(f),
    })

detections.sort(key=lambda x: x['z_score'], reverse=True)

# ── Helper: normalize PSD to noise floor ──────────────────────
def normalize_psd(freqs, psd):
    """Express PSD in dB relative to the local noise floor."""
    bg_mask = ((freqs > 200) & (freqs < 600)) | ((freqs > 900) & (freqs < 2500))
    bg_mean = np.mean(psd[bg_mask])
    bg_mean = max(bg_mean, 1e-30)
    return 10 * np.log10(psd / bg_mean)

# ── Figure: 2 panels ──────────────────────────────────────────
fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(13, 12))
fig.patch.set_facecolor('#0d1b2a')
for ax in (ax_top, ax_bot):
    ax.set_facecolor('#0d1b2a')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(detections)))

# ── TOP PANEL: overlay all detections ─────────────────────────
peak_freqs = []
for i, det in enumerate(detections):
    freqs = det['freqs']
    psd_db = normalize_psd(freqs, det['psd'])
    mask = freqs <= 2000

    ts_short = det['timestamp'][11:16]
    label = f"z={det['z_score']:.2f}  {ts_short} UTC  peak={det['peak_freq']:.0f} Hz"
    ax_top.plot(freqs[mask], psd_db[mask], color=colors[i],
                alpha=0.85, linewidth=0.9, label=label)
    peak_freqs.append(det['peak_freq'])

ax_top.axhline(0, color='#888', linestyle='--', linewidth=0.8, label='Noise floor (0 dB)')
ax_top.set_xlim(0, 2000)
ax_top.set_xlabel('Frequency offset from IF center (Hz)')
ax_top.set_ylabel('PSD above noise floor (dB)')
ax_top.set_title(
    'Starlink Ku-band Pilot Tone Detections\n'
    'USRP X310  |  Ch6 IF=1825 MHz → RF=11.575 GHz  |  San Diego CA\n'
    'Pilot Tone PSD — Top Detections (noise-normalized, DC excluded)'
)
ax_top.legend(fontsize=8, loc='upper right', framealpha=0.85,
              facecolor='#1a2a3a', edgecolor='#444', labelcolor='white',
              borderpad=0.8, labelspacing=0.5)
ax_top.grid(True, alpha=0.2, color='#555')
ax_top.margins(y=0.15)

# ── BOTTOM PANEL: best single detection ───────────────────────
best = detections[0]
freqs = best['freqs']
psd_db = normalize_psd(freqs, best['psd'])
mask = freqs <= 2000

ax_bot.plot(freqs[mask], psd_db[mask], color='steelblue', linewidth=0.9)
ax_bot.axhline(0, color='#888', linestyle='--', linewidth=0.8, label='Noise floor')

# Annotate peak
search = (freqs > 700) & (freqs < 850) & mask
if np.any(search):
    pk_idx  = np.argmax(psd_db[search])
    pk_freq = freqs[search][pk_idx]
    pk_db   = psd_db[search][pk_idx]
    ax_bot.annotate(
        f'Pilot tone\n{pk_freq:.0f} Hz, z={best["z_score"]:.2f}',
        xy=(pk_freq, pk_db),
        xytext=(pk_freq + 120, pk_db + 0.25),
        arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
        color='red', fontsize=9, fontweight='bold'
    )

ts_best = best['timestamp']
ax_bot.set_xlim(0, 2000)
ax_bot.set_xlabel('Frequency offset from IF center (Hz)')
ax_bot.set_ylabel('PSD above noise floor (dB)')
ax_bot.set_title(
    f'Best Detection — z={best["z_score"]:.2f},  {ts_best[11:19]} UTC  (30-sec integration)'
)
ax_bot.grid(True, alpha=0.2, color='#555')
ax_bot.margins(y=0.20)

plt.tight_layout(pad=3.0, h_pad=4.0)
plt.savefig(OUTPUT, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved: {OUTPUT}")

# ── Print key result ──────────────────────────────────────────
median_f = np.median(peak_freqs)
print()
print("=" * 55)
print("  KEY RESULT: 750 Hz OFDM Frame Fingerprint")
print("=" * 55)
print(f"  Detections plotted  : {len(detections)}")
for det in detections:
    print(f"    z={det['z_score']:.2f}  peak={det['peak_freq']:.1f} Hz  ({det['timestamp'][:19]})")
print()
print(f"  Median peak frequency : {median_f:.1f} Hz")
print(f"  Expected              : 750.0 Hz  (1 / 1.33 ms OFDM frame period)")
print()
print("  Interpretation:")
print("    Each Starlink OFDM frame repeats every 1.33 ms (750 Hz).")
print("    The power envelope |x(t)|^2 oscillates at 750 Hz,")
print("    producing a spectral line at 750 Hz + harmonics.")
print("    This fingerprint is detectable without decoding any payload.")
print("=" * 55)
