# Antarctica level 2 - "Listen to the bay" - baseline
# Reads bay.wav, flags every loud moment in the whale band, splices those moments into a reel,
# and draws the reel's spectrogram. It is a starting point, not a solution.

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import soundfile as sf
from scipy import signal

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SR = 250
WHALE_BAND = (15.0, 30.0)      # blue whale song tones and fin whale pulses both live here
FRAME_S = 0.5
MIN_EVENT_S = 3.0


def band_energy(x, lo, hi):
    """Energy per FRAME_S frame inside one band, in dB."""
    sos = signal.butter(4, (lo, hi), btype="bandpass", fs=SR, output="sos")
    y = signal.sosfiltfilt(sos, x)
    n = int(FRAME_S * SR)
    e = np.array([np.mean(y[i:i + n] ** 2) for i in range(0, len(y) - n, n)])
    return 10 * np.log10(e + 1e-12)


def loud_events(e_db, thresh_db):
    """Stretches of frames above the threshold that last at least MIN_EVENT_S."""
    above = e_db > thresh_db
    edges = np.flatnonzero(np.diff(np.r_[0, above.astype(int), 0])).reshape(-1, 2)
    return [(a * FRAME_S, b * FRAME_S) for a, b in edges if (b - a) * FRAME_S >= MIN_EVENT_S]


def splice(x, events):
    """The reel: the events in order, and where each starts inside the reel."""
    parts, bounds, pos = [], [], 0
    for a, b in events:
        seg = x[int(a * SR):int(b * SR)]
        parts.append(seg)
        bounds.append((pos, pos + len(seg)))
        pos += len(seg)
    return np.concatenate(parts), bounds


def spectrogram_png(x, bounds, out_png):
    """Log spectrogram of the reel with a line at every splice."""
    f, t, z = signal.stft(x, fs=SR, nperseg=500, noverlap=375)
    db = 10 * np.log10(np.abs(z) ** 2 + 1e-12)
    fig, ax = plt.subplots(figsize=(20, 4))
    ax.pcolormesh(t, f, db, shading="auto")
    for a, _ in bounds:
        ax.axvline(a / SR, c="white", lw=0.4)
    ax.set_xlabel("reel time, s")
    ax.set_ylabel("Hz")
    ax.set_title("the baseline reel: everything loud in %d-%d Hz" % WHALE_BAND)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main():
    """Energy above the median in 15-30 Hz is treated as a call. The reel this gives is short and
    mixed: most of what is loud down there is not a blue whale, and the faint tones are missing."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--bay", type=Path, default=Path("bay.wav"))
    ap.add_argument("--out", type=Path, default=Path("reel_baseline.wav"))
    args = ap.parse_args()
    x, sr = sf.read(str(args.bay), dtype="float32")
    assert sr == SR
    e = band_energy(x, *WHALE_BAND)
    events = loud_events(e, np.median(e) + 6.0)
    print("%d loud events in %.1f h" % (len(events), len(x) / SR / 3600))
    reel, bounds = splice(x, events)
    sf.write(str(args.out), reel, SR, subtype="PCM_16")
    print("reel of %.1f min written to %s" % (len(reel) / SR / 60, args.out))
    spectrogram_png(reel, bounds, args.out.with_suffix(".png"))
    print("spectrogram written to", args.out.with_suffix(".png"))


if __name__ == "__main__":
    main()
