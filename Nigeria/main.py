import matplotlib.pyplot as plt
from scipy.io import wavfile

sample_rate, samples = wavfile.read("challenge.wav")

plt.figure(figsize=(12, 6))
plt.specgram(samples, Fs=sample_rate, NFFT=1024, noverlap=512, cmap="inferno")
plt.xlabel("Timp (s)")
plt.ylabel("Frecvență (Hz)")
plt.colorbar(label="Intensitate (dB)")
plt.tight_layout()
plt.savefig("spectrogram.png", dpi=300)
plt.show()
