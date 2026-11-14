import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import median_filter

def load(path):
    x, sr = sf.read(path)
    return (x.mean(axis=1) if x.ndim > 1 else x), sr

def find_tones(x, sr, thresh_db, nperseg=2**17):
    f, p = signal.welch(x, sr, nperseg=nperseg, noverlap=nperseg // 2)
    db = 10 * np.log10(p + 1e-18)
    base = median_filter(db, size=1501, mode="nearest")
    excess = db - base
    idx, _ = signal.find_peaks(excess, height=thresh_db, distance=8)
    idx = idx[(f[idx] > 60) & (f[idx] < sr / 2 - 100)]
    return f[idx], excess[idx]

def notch_all(x, sr, freqs, width_hz):
    y = x.copy()
    for fc in freqs:
        q = fc / width_hz
        b, a = signal.iirnotch(fc, q, fs=sr)
        y = signal.sosfiltfilt(signal.tf2sos(b, a), y)
    return y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--thresh", type=float, default=6.0, help="peak height over baseline, dB")
    ap.add_argument("--width", type=float, default=6.0, help="notch width in Hz")
    ap.add_argument("--passes", type=int, default=2, help="repeat tone detection")
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--out", default="out3")
    args = ap.parse_args()

    x, sr = load(args.wav)
    y = x
    for n in range(args.passes):
        freqs, ex = find_tones(y, sr, args.thresh)
        print(f"pass {n+1}: {len(freqs)} tones", np.round(freqs[:12], 1), "...")
        if len(freqs) == 0:
            break
        y = notch_all(y, sr, freqs, args.width)

    sos = signal.butter(4, [100, 4500], btype="band", fs=sr, output="sos")
    y = signal.sosfiltfilt(sos, y)
    y = y / (np.max(np.abs(y)) + 1e-9) * 0.9

    out = Path(args.out); out.mkdir(exist_ok=True)
    sf.write(out / "clean_full.wav", y, sr)
    step = args.chunk * sr
    for i in range(0, len(y), step):
        sf.write(out / f"chunk_{i // step:03d}.wav", y[i:i + step], sr)
    print("done ->", out)

if __name__ == "__main__":
    main()