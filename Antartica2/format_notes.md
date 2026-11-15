# The recordings

All audio is mono 16-bit WAV at **250 Hz**. A sample rate that low is normal for this kind of
work: the sounds of interest live below 100 Hz, far under what a human ear picks out, so the
hydrophone data are decimated and analysed as a time-frequency picture rather than listened
to. (Speeding a file up twenty times moves the calls into the audible range, if you want to
hear them.)

## `train_clips/` and `train_manifest.csv`

Clips of exactly 30 s cut from recordings of three moorings in the Southern Ocean. The
manifest has one row per clip:

| Column | Meaning |
|---|---|
| `clip` | file name in `train_clips/` |
| `site` | mooring the clip comes from (`greenwich2015`, `maudrise2014`, `kerguelen2005`) |
| `label` | what an expert annotator marked in the clip, one of the four values below |

| Label | What is in the clip |
|---|---|
| `bma` | an Antarctic blue whale **A-call**: the first unit of the species' song, a tone near 28 Hz held for about ten seconds, sometimes followed by the lower second unit |
| `fin` | fin whale sound: short pulses near 20 Hz, often in long regular trains, sometimes with a higher component |
| `dcall` | a blue whale **D-call**: a sweep that falls from roughly 90 Hz to 30 Hz within a second or two; a social call, not the song |
| `none` | nothing annotated in or around the clip: sea noise, ice, distant ships |

Clips of one label may contain other sounds in the background. Sites differ in noise level,
instrument gain and how often each sound occurs.

## `bay.wav`

A continuous recording of about two hours from a fourth mooring, not present in the
training set. There are no labels.

## `reel.wav`

Same format as the input (mono 16-bit, 250 Hz). The blue whale song of the bay, in the order
it was sung, and nothing else.
