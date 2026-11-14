Each WAV sample is a 16-bit signed integer between -32,768 and 32,767, recorded mono at 22,050 Hz over ~10 seconds. While the raw waveform sounds like a flat, slightly buzzing test tone, the engineer embedded a visual message directly into the frequency spectrum. Using a Short-Time Fourier Transform (STFT), the script slices the audio into overlapping time windows and maps the signal energy across frequencies from 0 Hz up to the Nyquist limit (11,025 Hz).

To reveal what is hidden inside challenge.wav, the solver performs three steps:
1. Load audio: Read the sample rate and 16-bit PCM array from challenge.wav.
2. Compute spectrogram: Apply an STFT (NFFT=1024, noverlap=512) to convert the time-domain signal into a 2D time-frequency matrix.
3. Render spectrum: Plot frequency (y-axis) against time (x-axis) in decibels, exposing the high-frequency carrier tones spelling out the hidden text.

Because the signal energy is concentrated in the upper frequency bands rather than audible speech or drum rhythms, standard playback hides the message while a spectrogram renders it clearly.

Result on challenge.wav (22,050 Hz, 10.0s):
Carrier band: ~6,000 Hz – 10,500 Hz
Decoded text: TALKINGDRUM