#!/usr/bin/python3
"""
Starlink Sentinel — Always-On Detection Daemon
================================================
Continuously captures Ch6 (11.575 GHz) and accumulates a rolling 30-second
power envelope.  Detects the 750 Hz Starlink frame periodicity with far better
SNR than the old 1-second snapshot approach.

Usage:
    screen -S sentinel
    python3 ~/Starlink/starlink_sentinel.py
    # Ctrl-A, D  →  detach

Dependencies (all already installed except skyfield):
    pip install skyfield
"""

import os
import sys
import time
import logging
import sqlite3
import datetime
from collections import deque

import numpy as np
from scipy.signal import welch
import requests

# UHD is imported lazily so the file can be syntax-checked without hardware
try:
    import uhd
    UHD_AVAILABLE = True
except ImportError:
    UHD_AVAILABLE = False

try:
    from skyfield.api import load, wgs84, EarthSatellite
    from skyfield.timelib import Time
    SKYFIELD_AVAILABLE = True
except ImportError:
    SKYFIELD_AVAILABLE = False


# ===========================================================================
# CONFIGURATION
# ===========================================================================

class Config:
    # ---- Hardware ----
    USRP_ADDR   = "192.168.50.2"
    SAMPLE_RATE = 50e6          # 50 MS/s
    GAIN        = 50            # dB  (up from 35 in detector_lite)
    ANTENNA     = "TX/RX"
    LNBF_LO     = 9.75e9       # Standard Ku LNB local oscillator

    # ---- Target channel: Ch6 is the best Starlink candidate ----
    CHANNEL      = 6
    IF_FREQ_HZ   = 1825e6      # 11.575 GHz Ku − 9.75 GHz LO

    # ---- Power-envelope parameters ----
    ENVELOPE_WINDOW_US = 100   # 100 µs windows → 10 kHz envelope rate
    ENV_FS             = 10000.0  # Hz
    CHUNK_S            = 1.0   # Capture 1 s of IQ at a time
    SETTLE_MS          = 20    # Discard first 20 ms of each chunk (USRP LO settles in <10ms)
    INTEGRATION_S      = 30    # Accumulate 30 s of envelope before detection

    # ---- Detection thresholds ----
    SOFT_Z = 3.0   # save metadata + power envelope (.npy)
    HARD_Z = 4.0   # also save 500 ms IQ snippet (.cfile)
    SOFT_MIN_HARMONICS = 1   # envelope only saved if harmonics >= this

    # ---- Save cooldowns (prevent saving every second during a pass) ----
    ENVELOPE_COOLDOWN_S = 120   # save at most one envelope per 2 minutes
    SNIPPET_COOLDOWN_S  = 300   # save at most one snippet per 5 minutes

    # ---- TLE / satellite tracking ----
    LAT           = 32.88      # Dish latitude  (degrees N)  ← SET THIS
    LON           = -117.234     # Dish longitude (degrees E, negative = West)  ← SET THIS
    ALT_M         = 131        # Dish altitude in metres (approximate)
    MIN_ELEVATION = 15.0       # Only flag pass if satellite > 15° elevation
    TLE_URL       = ("https://celestrak.org/NORAD/elements/gp.php"
                     "?GROUP=starlink&FORMAT=tle")
    TLE_UPDATE_INTERVAL_H = 12  # Re-download TLEs every 12 hours

    # ---- Storage ----
    BASE_DIR      = os.path.expanduser("~/Starlink/sentinel")
    DB_PATH       = os.path.join(BASE_DIR, "detections.db")
    ENVELOPE_DIR  = os.path.join(BASE_DIR, "envelopes")
    SNIPPET_DIR   = os.path.join(BASE_DIR, "snippets")
    LOG_PATH      = os.path.join(BASE_DIR, "sentinel.log")

    SNIPPET_DURATION_S = 0.5   # 500 ms of IQ per snippet (~200 MB)


# ===========================================================================
# TLE PREDICTOR
# ===========================================================================

class TLEPredictor:
    """Download Starlink TLEs and predict overhead passes."""

    def __init__(self, config: Config, logger: logging.Logger):
        self.cfg    = config
        self.logger = logger
        self._ts    = None          # skyfield Timescale
        self._sats  = []            # list of EarthSatellite
        self._obs   = None          # observer (skyfield GeographicPosition)
        self._last_update = None    # datetime of last TLE fetch

        if SKYFIELD_AVAILABLE:
            self._ts  = load.timescale()
            self._obs = wgs84.latlon(config.LAT, config.LON,
                                     elevation_m=config.ALT_M)

    # ------------------------------------------------------------------
    def update_tles(self) -> bool:
        """Download TLEs from Celestrak and build EarthSatellite list."""
        if not SKYFIELD_AVAILABLE:
            self.logger.warning("skyfield not installed — TLE tracking disabled")
            return False

        self.logger.info(f"Downloading TLEs from {self.cfg.TLE_URL} …")
        try:
            resp = requests.get(self.cfg.TLE_URL, timeout=30)
            resp.raise_for_status()
            lines = resp.text.strip().splitlines()

            sats = []
            # TLE format: name line, line1, line2  (3-line elements)
            i = 0
            while i + 2 < len(lines):
                name = lines[i].strip()
                l1   = lines[i+1].strip()
                l2   = lines[i+2].strip()
                if l1.startswith('1 ') and l2.startswith('2 '):
                    sats.append(EarthSatellite(l1, l2, name, self._ts))
                    i += 3
                else:
                    i += 1

            self._sats        = sats
            self._last_update = datetime.datetime.utcnow()
            self.logger.info(f"  Loaded {len(sats)} Starlink satellites")
            return True

        except Exception as exc:
            self.logger.error(f"TLE update failed: {exc}")
            return False

    # ------------------------------------------------------------------
    def _refresh_if_stale(self):
        """Auto-refresh TLEs if older than TLE_UPDATE_INTERVAL_H hours."""
        if self._last_update is None:
            self.update_tles()
            return
        age = (datetime.datetime.utcnow() - self._last_update).total_seconds()
        if age > self.cfg.TLE_UPDATE_INTERVAL_H * 3600:
            self.logger.info("TLEs are stale — refreshing …")
            self.update_tles()

    # ------------------------------------------------------------------
    def get_visible(self) -> list:
        """Return list of dicts for currently visible Starlink satellites."""
        if not SKYFIELD_AVAILABLE or not self._sats:
            return []
        try:
            t = self._ts.now()
            visible = []
            for sat in self._sats:
                diff    = sat - self._obs
                topocentric = diff.at(t)
                alt, az, dist = topocentric.altaz()
                if alt.degrees >= self.cfg.MIN_ELEVATION:
                    visible.append({
                        'name':      sat.name,
                        'elevation': alt.degrees,
                        'azimuth':   az.degrees,
                        'distance_km': dist.km,
                    })
            # Sort highest elevation first
            visible.sort(key=lambda d: d['elevation'], reverse=True)
            return visible
        except Exception as exc:
            self.logger.warning(f"get_visible() error: {exc}")
            return []

    # ------------------------------------------------------------------
    def is_pass_active(self) -> bool:
        """True if any Starlink satellite is above MIN_ELEVATION right now."""
        return len(self.get_visible()) > 0

    # ------------------------------------------------------------------
    def get_next_pass_minutes(self) -> float:
        """Approximate minutes until the next overhead pass (best effort)."""
        if not SKYFIELD_AVAILABLE or not self._sats:
            return float('nan')
        try:
            t0 = self._ts.now()
            # Check in 1-minute increments for the next 120 minutes
            best = float('inf')
            for minute in range(1, 121):
                t = self._ts.utc(
                    *datetime.datetime.utcnow().timetuple()[:5],
                    minute * 60
                )
                for sat in self._sats[:50]:   # check first 50 to keep it fast
                    diff = sat - self._obs
                    topocentric = diff.at(t)
                    alt, _, _ = topocentric.altaz()
                    if alt.degrees >= self.cfg.MIN_ELEVATION:
                        best = min(best, minute)
                        break
                if best < float('inf'):
                    break
            return best if best < float('inf') else float('nan')
        except Exception:
            return float('nan')


# ===========================================================================
# USRP CAPTURE
# ===========================================================================

class USRPCapture:
    """Thin wrapper around uhd.  Tune once to Ch6 and stream IQ chunks."""

    def __init__(self, config: Config, logger: logging.Logger):
        self.cfg    = config
        self.logger = logger
        self.usrp   = None
        self.streamer = None

    # ------------------------------------------------------------------
    def init(self) -> bool:
        """Initialize USRP — returns True on success."""
        if not UHD_AVAILABLE:
            self.logger.error("uhd Python bindings not found.")
            return False

        self.logger.info(f"Connecting to USRP at {self.cfg.USRP_ADDR} …")
        try:
            self.usrp = uhd.usrp.MultiUSRP(f"addr={self.cfg.USRP_ADDR}")

            self.usrp.set_rx_rate(self.cfg.SAMPLE_RATE)
            self.usrp.set_rx_gain(self.cfg.GAIN)
            self.usrp.set_rx_antenna(self.cfg.ANTENNA)

            st_args          = uhd.usrp.StreamArgs("fc32", "sc16")
            st_args.channels = [0]
            self.streamer     = self.usrp.get_rx_stream(st_args)

            # Tune to Ch6 IF frequency — stay there forever
            tune_request = uhd.types.TuneRequest(self.cfg.IF_FREQ_HZ)
            self.usrp.set_rx_freq(tune_request)
            time.sleep(0.2)   # let PLL settle

            ku_ghz = (self.cfg.IF_FREQ_HZ + self.cfg.LNBF_LO) / 1e9
            self.logger.info(
                f"USRP ready — Ch{self.cfg.CHANNEL} | "
                f"IF {self.cfg.IF_FREQ_HZ/1e6:.1f} MHz | "
                f"RF {ku_ghz:.3f} GHz | "
                f"Rate {self.usrp.get_rx_rate()/1e6:.1f} MS/s | "
                f"Gain {self.usrp.get_rx_gain():.1f} dB"
            )
            return True

        except Exception as exc:
            self.logger.error(f"USRP init failed: {exc}")
            return False

    # ------------------------------------------------------------------
    def capture_chunk(self, duration_s: float = 1.0) -> np.ndarray:
        """Capture duration_s seconds of IQ using num_done burst mode."""
        num_samples = int(duration_s * self.cfg.SAMPLE_RATE)
        buffer      = np.zeros(num_samples, dtype=np.complex64)
        metadata    = uhd.types.RXMetadata()

        stream_cmd            = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
        stream_cmd.num_samps  = num_samples
        stream_cmd.stream_now = True
        self.streamer.issue_stream_cmd(stream_cmd)

        received = 0
        empty_streak = 0          # consecutive zero-sample recv() calls
        while received < num_samples:
            n = self.streamer.recv(buffer[received:], metadata)
            received += n
            err = metadata.error_code

            if err == uhd.types.RXMetadataErrorCode.overflow:
                if n == 0:
                    empty_streak += 1
                    if empty_streak > 50:
                        self.logger.debug("RX overflow stall — giving up on chunk")
                        break
                else:
                    empty_streak = 0
                # Don't break on overflow — data keeps coming after the notification
                continue
            elif err == uhd.types.RXMetadataErrorCode.timeout:
                break   # num_done finished
            elif err != uhd.types.RXMetadataErrorCode.none:
                self.logger.warning(f"RX error: {err}")
                break
            else:
                empty_streak = 0

        return buffer[:received]

    # ------------------------------------------------------------------
    def stop(self):
        """No-op for num_done mode (stream ends automatically)."""
        pass

    # ------------------------------------------------------------------
    def check_saturation(self, iq: np.ndarray) -> bool:
        """Warn if >1% of samples are clipping.  Returns True if clipping."""
        # fc32 from sc16 — max amplitude is ~1.0 after UHD normalisation
        clipped = np.mean(np.abs(iq) > 0.98)
        if clipped > 0.01:
            self.logger.warning(
                f"ADC saturation: {clipped*100:.1f}% of samples clipping — "
                "consider reducing GAIN"
            )
            return True
        return False


# ===========================================================================
# DETECTION ENGINE
# ===========================================================================

class DetectionEngine:
    """
    Stateful rolling 30-second power-envelope accumulator.

    Every ~1 second, feed in a new IQ chunk.  Once we have ≥5 seconds of
    envelope data, run the 750 Hz Welch periodicity test and the 1.33 ms
    autocorrelation check.
    """

    def __init__(self, config: Config):
        self.cfg = config
        max_env_samples = int(config.INTEGRATION_S * config.ENV_FS)
        self.envelope_buffer = deque(maxlen=max_env_samples)   # 300,000 floats
        self.iq_ring         = deque(maxlen=3)                 # last 3 chunks

    # ------------------------------------------------------------------
    def process_chunk(self, iq: np.ndarray) -> dict | None:
        """
        Process one IQ chunk.  Returns a result dict when enough data has
        accumulated, otherwise returns None.
        """
        # 1. Discard first SETTLE_MS to remove USRP LO transient
        settle_n = int(self.cfg.SETTLE_MS * 1e-3 * self.cfg.SAMPLE_RATE)
        iq_clean = iq[settle_n:]

        # 2. Compute 100 µs power-envelope windows
        win = int(self.cfg.SAMPLE_RATE * self.cfg.ENVELOPE_WINDOW_US * 1e-6)
        n   = len(iq_clean) // win
        if n == 0:
            return None
        pwr = np.mean(
            np.abs(iq_clean[:n * win].reshape(n, win)) ** 2, axis=1
        )

        # 3. Extend rolling buffer
        self.envelope_buffer.extend(pwr.tolist())

        # 4. Keep IQ for potential snippet saving
        self.iq_ring.append(iq_clean)

        # 5. Only run detection if we have ≥5 seconds of envelope
        min_env = int(5 * self.cfg.ENV_FS)
        if len(self.envelope_buffer) < min_env:
            return None

        env = np.array(self.envelope_buffer)
        integration_s = len(env) / self.cfg.ENV_FS

        # 6. 750 Hz Welch test
        prd = self._detect_750hz(env)

        # 7. 1.33 ms envelope autocorrelation (frame periodicity)
        autocorr_val = self._autocorr_1_33ms(env)

        # 8. OFDM CP autocorrelation on raw IQ (symbol-period structure)
        cp_peak, cp_snr, cp_lag = self._cp_autocorr(iq_clean)

        # 9. Bandwidth estimate from the most recent IQ chunk
        bw_mhz = self._estimate_bw(iq_clean)

        timestamp = datetime.datetime.utcnow()
        ku_ghz    = (self.cfg.IF_FREQ_HZ + self.cfg.LNBF_LO) / 1e9

        return {
            'timestamp_utc': timestamp.isoformat(),
            'unix_time':     timestamp.timestamp(),
            'channel':       self.cfg.CHANNEL,
            'ku_ghz':        ku_ghz,
            'z_score':       prd['z_score'],
            'peak_snr':      prd['peak_snr'],
            'peak_freq_hz':  prd['peak_freq'],
            'harmonics':     prd['harmonics'],
            'autocorr_1_33': autocorr_val,
            'cp_peak':       cp_peak,
            'cp_snr':        cp_snr,
            'cp_lag':        cp_lag,
            'bw_mhz':        bw_mhz,
            'integration_s': integration_s,
            # caller fills in satellite context:
            'in_pass':       0,
            'sat_name':      '',
            'sat_elevation': 0.0,
            # caller assigns confidence after threshold check:
            'confidence':    'BACKGROUND',
            # file paths filled in by PeakStore if saved:
            'envelope_file': '',
            'snippet_file':  '',
            # carry PSD data for .npy saving (not stored in DB):
            '_welch_freqs':  prd['welch_freqs'],
            '_welch_psd':    prd['welch_psd'],
            '_envelope':     env,
        }

    # ------------------------------------------------------------------
    def _detect_750hz(self, env: np.ndarray) -> dict:
        """Welch PSD of the power envelope, searching for the 750 Hz peak."""
        env_c = env - env.mean()

        nperseg  = int(self.cfg.ENV_FS * 0.5)   # 500 ms segments
        noverlap = int(self.cfg.ENV_FS * 0.25)  # 250 ms overlap
        freqs, psd = welch(
            env_c,
            fs=self.cfg.ENV_FS,
            nperseg=nperseg,
            noverlap=noverlap,
            window='hann',
            scaling='spectrum',
        )

        # Search band: 700–800 Hz
        search = (freqs >= 700) & (freqs <= 800)
        # Background band: 200–600 Hz and 900–2500 Hz (avoids 750 Hz and harmonics)
        bg     = ((freqs >= 200) & (freqs <= 600)) | \
                 ((freqs >= 900) & (freqs <= 2500))

        if not np.any(search) or not np.any(bg):
            return {'z_score': 0.0, 'peak_snr': 0.0, 'peak_freq': 0.0,
                    'harmonics': 0, 'welch_freqs': freqs, 'welch_psd': psd}

        search_psd  = psd[search]
        search_freq = freqs[search]
        bg_psd      = psd[bg]

        peak_idx  = np.argmax(search_psd)
        peak_freq = search_freq[peak_idx]
        peak_val  = search_psd[peak_idx]

        bg_mean = np.mean(bg_psd)
        bg_std  = np.std(bg_psd)

        z_score = (peak_val - bg_mean) / (bg_std + 1e-30)
        snr     = peak_val / (bg_mean + 1e-30)

        # Check harmonics: 1500, 2250, 3000 Hz
        harmonics = 0
        for h in [2, 3, 4]:
            h_mask = (freqs >= h * 750 - 50) & (freqs <= h * 750 + 50)
            if np.any(h_mask) and np.max(psd[h_mask]) > bg_mean + 2 * bg_std:
                harmonics += 1

        return {
            'z_score':     float(z_score),
            'peak_snr':    float(snr),
            'peak_freq':   float(peak_freq),
            'harmonics':   harmonics,
            'welch_freqs': freqs,
            'welch_psd':   psd,
        }

    # ------------------------------------------------------------------
    def _autocorr_1_33ms(self, env: np.ndarray) -> float:
        """Normalised autocorrelation of the envelope at lag = 1.33 ms."""
        lag = int(1.333e-3 * self.cfg.ENV_FS)   # = 13 samples at 10 kHz
        env_c = env - env.mean()

        # Use correlate only on a 5-second window to keep it fast
        window = int(5 * self.cfg.ENV_FS)
        x = env_c[-window:]

        autocorr = np.correlate(x, x, mode='full')
        mid = len(autocorr) // 2
        norm = autocorr[mid]
        if norm == 0:
            return 0.0
        return float(autocorr[mid + lag] / norm)

    # ------------------------------------------------------------------
    def _cp_autocorr(self, iq: np.ndarray) -> tuple:
        """
        OFDM cyclic-prefix autocorrelation on raw IQ samples.

        For an OFDM signal with CP length L and symbol period N, the signal
        satisfies x[n] ≈ x[n + N] over the CP region.  This creates a
        normalized autocorrelation peak at lag = N samples.

        Starlink expected symbol period: 4.267 μs → 213 samples at 50 MS/s.
        Expected CP contribution: ~0.1–0.25 (depending on CP fraction).
        Noise floor at this lag: ~1/sqrt(n_symbols) → small for many symbols.

        Returns (peak_val, cp_snr, peak_lag_samples)
          peak_val  — normalized |R(N)| / R(0),  ~0 for noise, ~0.1–0.25 for OFDM
          cp_snr    — peak_val / background_mean, large values = strong CP structure
          peak_lag  — actual sample lag of the peak (should be near 213)
        """
        # 5 ms of IQ = 250,000 samples → ~1170 OFDM symbols for good averaging
        n = min(len(iq), int(5e-3 * self.cfg.SAMPLE_RATE))
        if n < 500:
            return 0.0, 0.0, 0

        x     = iq[:n].astype(np.complex64)
        power = float(np.mean(np.abs(x) ** 2)) + 1e-30

        expected_lag = int(4.267e-6 * self.cfg.SAMPLE_RATE)  # 213 samples

        # Search lags: expected ± 30 samples
        s_lags = np.arange(max(1, expected_lag - 30),
                            min(n - 1, expected_lag + 31))
        # Background lags: well away from symbol period and its harmonics
        b_lags = np.arange(500, 600)

        def corr_mag(lag):
            return float(np.abs(np.dot(np.conj(x[:-lag]), x[lag:]))) / (n * power)

        s_vals = np.array([corr_mag(l) for l in s_lags])
        b_vals = np.array([corr_mag(l) for l in b_lags])

        peak_idx = int(np.argmax(s_vals))
        peak_val = float(s_vals[peak_idx])
        peak_lag = int(s_lags[peak_idx])
        bg_mean  = float(np.mean(b_vals))
        cp_snr   = peak_val / (bg_mean + 1e-30)

        return peak_val, cp_snr, peak_lag

    # ------------------------------------------------------------------
    def _estimate_bw(self, iq: np.ndarray) -> float:
        """Quick occupied-bandwidth estimate (MHz) from a short IQ segment."""
        try:
            n = min(len(iq), int(self.cfg.SAMPLE_RATE * 0.1))  # 100 ms
            freqs, psd = welch(iq[:n], fs=self.cfg.SAMPLE_RATE,
                               nperseg=4096, return_onesided=False)
            psd_db = 10 * np.log10(np.fft.fftshift(np.abs(psd)) + 1e-12)
            noise  = np.percentile(psd_db, 10)
            mask   = psd_db > (noise + 3)
            if not np.any(mask):
                return 0.0
            f_shift = np.fft.fftshift(freqs)
            return float((f_shift[mask].max() - f_shift[mask].min()) / 1e6)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    def get_iq_snippet(self) -> np.ndarray:
        """Return the last 500 ms of IQ from the ring buffer."""
        n = int(self.cfg.SNIPPET_DURATION_S * self.cfg.SAMPLE_RATE)
        if not self.iq_ring:
            return np.array([], dtype=np.complex64)
        combined = np.concatenate(list(self.iq_ring))
        return combined[-n:] if len(combined) >= n else combined

    # ------------------------------------------------------------------
    def reset_buffer(self):
        """Clear the envelope buffer at the end of a satellite pass."""
        self.envelope_buffer.clear()


# ===========================================================================
# PEAK STORE  (SQLite + file saving)
# ===========================================================================
# STORAGE MANAGER
# ===========================================================================

class StorageManager:
    """
    Keeps total file storage under MAX_GB by deleting the lowest-quality
    saved files (envelopes + snippets) when the limit is approached.

    Quality score = z_score × (1 + 0.5 × harmonics) × (2 if in_pass else 1)

    Called every CHECK_INTERVAL_S seconds from the main loop.
    SQLite rows are never deleted (trivially small).
    """

    CHECK_INTERVAL_S = 300     # check every 5 minutes
    MAX_GB           = 10.0
    TRIM_TO_GB       = 8.0     # trim down to this level when limit hit

    def __init__(self, config: Config, logger: logging.Logger):
        self.cfg    = config
        self.logger = logger
        self._last_check = 0.0

    # ------------------------------------------------------------------
    def check_and_trim(self, conn: sqlite3.Connection):
        """Call from main loop; does nothing unless CHECK_INTERVAL_S elapsed."""
        now = time.monotonic()
        if now - self._last_check < self.CHECK_INTERVAL_S:
            return
        self._last_check = now

        used_gb = self._get_used_gb()
        self.logger.info(f"Storage check: {used_gb:.2f} GB used / {self.MAX_GB} GB limit")

        if used_gb < self.MAX_GB:
            return

        self.logger.warning(
            f"Storage limit reached ({used_gb:.2f} GB) — pruning low-quality files …"
        )
        self._prune(conn)

    # ------------------------------------------------------------------
    def _get_used_gb(self) -> float:
        total = 0
        for d in (self.cfg.ENVELOPE_DIR, self.cfg.SNIPPET_DIR):
            for f in os.scandir(d):
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
        return total / 1e9

    # ------------------------------------------------------------------
    def _quality_score(self, row: dict) -> float:
        z      = row.get('z_score', 0) or 0
        harm   = row.get('harmonics', 0) or 0
        ip     = row.get('in_pass', 0) or 0
        cp_snr = row.get('cp_snr', 0) or 0
        # CP SNR > 3 is strong OFDM evidence — reward it significantly
        cp_bonus = 1.0 + min(cp_snr / 3.0, 2.0)   # caps at 3× bonus at cp_snr=6
        return z * (1 + 0.5 * harm) * (2.0 if ip else 1.0) * cp_bonus

    # ------------------------------------------------------------------
    def _prune(self, conn: sqlite3.Connection):
        # Fetch all rows that have a saved file, ordered by quality ascending
        rows = conn.execute("""
            SELECT id, z_score, harmonics, in_pass, envelope_file, snippet_file
            FROM detections
            WHERE envelope_file != '' OR snippet_file != ''
            ORDER BY (z_score * (1 + 0.5 * harmonics) * (1 + in_pass) * (1 + MIN(COALESCE(cp_snr,0)/3.0, 2.0))) ASC
        """).fetchall()

        freed = 0.0
        target_free = (self.MAX_GB - self.TRIM_TO_GB) * 1e9
        deleted_ids_env = []
        deleted_ids_snip = []

        for row in rows:
            if freed >= target_free:
                break
            rid, z, harm, ip, env_f, snip_f = row

            # Delete snippet first (larger, lowest quality first)
            if snip_f and os.path.exists(snip_f):
                size = os.path.getsize(snip_f)
                os.remove(snip_f)
                freed += size
                deleted_ids_snip.append(rid)
                self.logger.info(
                    f"  Pruned snippet z={z:.2f} harm={harm} pass={ip}: "
                    f"{os.path.basename(snip_f)} ({size/1e6:.0f} MB)"
                )

            if freed >= target_free:
                break

            # Then envelope
            if env_f and os.path.exists(env_f):
                size = os.path.getsize(env_f)
                os.remove(env_f)
                freed += size
                deleted_ids_env.append(rid)
                self.logger.info(
                    f"  Pruned envelope z={z:.2f} harm={harm} pass={ip}: "
                    f"{os.path.basename(env_f)} ({size/1e3:.0f} KB)"
                )

        # Clear file paths in DB for deleted files
        if deleted_ids_snip:
            conn.execute(
                f"UPDATE detections SET snippet_file='' WHERE id IN "
                f"({','.join('?'*len(deleted_ids_snip))})",
                deleted_ids_snip
            )
        if deleted_ids_env:
            conn.execute(
                f"UPDATE detections SET envelope_file='' WHERE id IN "
                f"({','.join('?'*len(deleted_ids_env))})",
                deleted_ids_env
            )
        conn.commit()

        self.logger.warning(
            f"Pruning complete — freed {freed/1e9:.2f} GB. "
            f"Removed {len(deleted_ids_snip)} snippets, {len(deleted_ids_env)} envelopes."
        )


# ===========================================================================

class PeakStore:
    """Log detection events to SQLite and save envelope / snippet files."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS detections (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp_utc  TEXT,
        unix_time      REAL,
        channel        INTEGER,
        ku_ghz         REAL,
        z_score        REAL,
        peak_snr       REAL,
        peak_freq_hz   REAL,
        harmonics      INTEGER,
        autocorr_1_33  REAL,
        cp_peak        REAL,
        cp_snr         REAL,
        cp_lag         INTEGER,
        bw_mhz         REAL,
        integration_s  REAL,
        in_pass        INTEGER,
        sat_name       TEXT,
        sat_elevation  REAL,
        confidence     TEXT,
        envelope_file  TEXT,
        snippet_file   TEXT
    );
    """

    def __init__(self, config: Config, logger: logging.Logger):
        self.cfg    = config
        self.logger = logger
        os.makedirs(config.ENVELOPE_DIR, exist_ok=True)
        os.makedirs(config.SNIPPET_DIR,  exist_ok=True)

        self._conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
        self._conn.execute(self._SCHEMA)
        self._conn.commit()
        self.logger.info(f"SQLite database: {config.DB_PATH}")

    # ------------------------------------------------------------------
    def log(self, result: dict):
        """Write one row to detections table."""
        cols = (
            'timestamp_utc', 'unix_time', 'channel', 'ku_ghz',
            'z_score', 'peak_snr', 'peak_freq_hz', 'harmonics',
            'autocorr_1_33', 'cp_peak', 'cp_snr', 'cp_lag', 'bw_mhz', 'integration_s',
            'in_pass', 'sat_name', 'sat_elevation', 'confidence',
            'envelope_file', 'snippet_file',
        )
        vals = tuple(result.get(c, None) for c in cols)
        sql  = (
            f"INSERT INTO detections ({', '.join(cols)}) "
            f"VALUES ({', '.join('?' * len(cols))})"
        )
        try:
            self._conn.execute(sql, vals)
            self._conn.commit()
        except Exception as exc:
            self.logger.error(f"DB insert error: {exc}")

    # ------------------------------------------------------------------
    def save_envelope(self, result: dict) -> str:
        """Save (freqs, psd, envelope) as .npy; return file path."""
        ts_safe = result['timestamp_utc'].replace(':', '-').replace('.', '-')
        fname   = f"env_z{result['z_score']:.1f}_{ts_safe}.npy"
        path    = os.path.join(self.cfg.ENVELOPE_DIR, fname)
        try:
            np.save(path, {
                'freqs':         result.get('_welch_freqs'),
                'psd':           result.get('_welch_psd'),
                'envelope':      result.get('_envelope'),
                'timestamp_utc': result['timestamp_utc'],
                'z_score':       result['z_score'],
                'peak_freq_hz':  result['peak_freq_hz'],
            }, allow_pickle=True)
            self.logger.debug(f"Envelope saved: {path}")
        except Exception as exc:
            self.logger.error(f"Failed to save envelope: {exc}")
            return ''
        return path

    # ------------------------------------------------------------------
    def save_snippet(self, iq_snippet: np.ndarray, result: dict) -> str:
        """Save IQ snippet as binary complex64 .cfile; return file path."""
        ts_safe = result['timestamp_utc'].replace(':', '-').replace('.', '-')
        fname   = f"snip_z{result['z_score']:.1f}_{ts_safe}.cfile"
        path    = os.path.join(self.cfg.SNIPPET_DIR, fname)
        try:
            iq_snippet.astype(np.complex64).tofile(path)
            self.logger.debug(
                f"Snippet saved: {path} ({iq_snippet.nbytes/1e6:.0f} MB)"
            )
        except Exception as exc:
            self.logger.error(f"Failed to save snippet: {exc}")
            return ''
        return path

    # ------------------------------------------------------------------
    def close(self):
        self._conn.close()


# ===========================================================================
# LOGGING SETUP
# ===========================================================================

def setup_logging(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fmt = logging.Formatter(
        '%(asctime)s  %(levelname)-8s  %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger('sentinel')
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ===========================================================================
# MAIN LOOP
# ===========================================================================

def main():
    cfg    = Config()
    logger = setup_logging(cfg.LOG_PATH)

    logger.info("=" * 60)
    logger.info("Starlink Sentinel starting up")
    logger.info(f"  Ch{cfg.CHANNEL}  IF={cfg.IF_FREQ_HZ/1e6:.0f} MHz  "
                f"RF={(cfg.IF_FREQ_HZ+cfg.LNBF_LO)/1e9:.3f} GHz")
    logger.info(f"  Integration window: {cfg.INTEGRATION_S} s")
    logger.info(f"  Thresholds: soft z>{cfg.SOFT_Z}  hard z>{cfg.HARD_Z}")
    logger.info(f"  Storage: {cfg.BASE_DIR}")
    logger.info("=" * 60)

    # ---- Initialise subsystems ----
    tle    = TLEPredictor(cfg, logger)
    usrp   = USRPCapture(cfg, logger)
    engine  = DetectionEngine(cfg)
    store   = PeakStore(cfg, logger)
    storage = StorageManager(cfg, logger)

    # Download TLEs at startup (non-fatal if no network)
    tle.update_tles()

    # Start USRP (exit on failure)
    if not usrp.init():
        logger.error("Cannot start without USRP — exiting")
        sys.exit(1)

    # ---- State for pass-transition detection ----
    was_in_pass = False
    chunk_count = 0
    last_envelope_save = 0.0   # unix time of last envelope save
    last_snippet_save  = 0.0   # unix time of last snippet save

    logger.info("Entering main capture loop (Ctrl-C to stop) …")

    try:
        while True:
            t_loop_start = time.monotonic()

            # 1. Capture one IQ chunk
            iq = usrp.capture_chunk(cfg.CHUNK_S)
            usrp.check_saturation(iq)

            # 2. Run detection engine
            result = engine.process_chunk(iq)

            if result is not None:
                # 3. Add satellite context
                visible            = tle.get_visible()
                in_pass            = bool(visible)
                result['in_pass']  = int(in_pass)
                if visible:
                    result['sat_name']      = visible[0]['name']
                    result['sat_elevation'] = visible[0]['elevation']

                # 4. Assign confidence label
                z = result['z_score']
                if z >= cfg.HARD_Z:
                    result['confidence'] = 'HARD'
                elif z >= cfg.SOFT_Z:
                    result['confidence'] = 'SOFT'
                else:
                    result['confidence'] = 'BACKGROUND'

                now = result['unix_time']

                # 5. Soft detection: save envelope (cooldown prevents saving every second)
                if (z >= cfg.SOFT_Z
                        and result['harmonics'] >= cfg.SOFT_MIN_HARMONICS
                        and now - last_envelope_save >= cfg.ENVELOPE_COOLDOWN_S):
                    env_path = store.save_envelope(result)
                    result['envelope_file'] = env_path
                    last_envelope_save = now
                    logger.info(
                        f"SOFT DETECTION  z={z:.2f}  peak={result['peak_freq_hz']:.1f} Hz  "
                        f"SNR={result['peak_snr']:.2f}x  harm={result['harmonics']}  "
                        f"autocorr={result['autocorr_1_33']:.4f}  "
                        f"pass={result['sat_name'] or 'none'}"
                    )

                # 6. Hard detection: save IQ snippet (cooldown prevents saving every second)
                if z >= cfg.HARD_Z and now - last_snippet_save >= cfg.SNIPPET_COOLDOWN_S:
                    snip      = engine.get_iq_snippet()
                    snip_path = store.save_snippet(snip, result)
                    result['snippet_file'] = snip_path
                    last_snippet_save = now
                    logger.warning(
                        f"*** HARD DETECTION  z={z:.2f} — IQ snippet saved: {snip_path}"
                    )

                # 7. Log to SQLite (always — even BACKGROUND rows for baseline)
                store.log(result)

                # 8. Periodic status line (every ~30 s when no detection)
                if chunk_count % 30 == 0:
                    pass_str = (f"SAT={visible[0]['name']} el={visible[0]['elevation']:.1f}°"
                                if visible else "no pass")
                    logger.info(
                        f"[tick {chunk_count:5d}]  z={z:.2f}  "
                        f"cp_snr={result['cp_snr']:.2f}  cp_lag={result['cp_lag']}samp  "
                        f"buf={result['integration_s']:.0f}s  {pass_str}"
                    )

                # 9. Detect end of satellite pass → reset envelope buffer
                if was_in_pass and not in_pass:
                    logger.info("Pass ended — clearing envelope buffer")
                    engine.reset_buffer()

                was_in_pass = in_pass

            # 10. Refresh TLEs if stale
            tle._refresh_if_stale()

            # 11. Storage management — prune worst files if > 10 GB
            storage.check_and_trim(store._conn)

            chunk_count += 1

            # 11. Sleep for any remaining time in the ~1 s cycle
            elapsed = time.monotonic() - t_loop_start
            sleep_s = max(0.0, cfg.CHUNK_S - elapsed - 0.05)
            if sleep_s > 0:
                time.sleep(sleep_s)

    except KeyboardInterrupt:
        logger.info("Interrupted by user — shutting down")
    finally:
        usrp.stop()
        store.close()
        logger.info("Sentinel stopped.")


if __name__ == "__main__":
    main()
