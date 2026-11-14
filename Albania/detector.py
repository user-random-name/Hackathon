import math
from collections import deque
MU = {
    "temperature_c": 13.00, "humidity_pct": 72.02, "vibration_g": 0.1199,
    "acoustic_db": 32.10, "air_pressure_kpa": 101.503, "co2_ppm": 510.35,
}
SD = {
    "temperature_c": 0.372, "humidity_pct": 0.580, "vibration_g": 0.01005,
    "acoustic_db": 0.350, "air_pressure_kpa": 0.0992, "co2_ppm": 10.5,
}
PHI = {"temperature_c": 0.715, "air_pressure_kpa": 0.79}

POINT_Z = 6.0
COLL_STEP = 4.0
COLL_WIN = 3.0
EW_LAM = 0.12
EW_THR = 1.0
EW_SHORT = 0.6
SLOW_THR = 3.0


def _finite(v, default):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


class Detector:
    def __init__(self):
        self.prev = {}
        self.joint = deque(maxlen=4)
        self.ew = {"humidity_pct": 0.0, "co2_ppm": 0.0}
        self.short = {"humidity_pct": deque(maxlen=4), "co2_ppm": deque(maxlen=4)}
        self.slow_recent = {c: deque(maxlen=8) for c in PHI}
        self.slow_base = {c: deque(maxlen=130) for c in PHI}
        self.last_flag = False

    def update(self, reading: dict) -> float:
        z = {}
        for c in MU:
            z[c] = (_finite(reading.get(c), MU[c]) - MU[c]) / SD[c]

        score = 0.0
        pmax = max(abs(z["vibration_g"]), abs(z["acoustic_db"]),
                   abs(z["co2_ppm"]), abs(z["humidity_pct"]))
        for c, phi in PHI.items():
            if c in self.prev:
                e = (z[c] - phi * self.prev[c]) / math.sqrt(1 - phi * phi)
                pmax = max(pmax, abs(e))
            else:
                pmax = max(pmax, abs(z[c]) / 1.5)
        if pmax > POINT_Z:
            score = max(score, 1.0)
        s = (z["vibration_g"] + z["acoustic_db"] + z["co2_ppm"]) / math.sqrt(3)
        self.joint.append(min(s, 8.0))
        wmean = sum(self.joint) / len(self.joint)
        if s > COLL_STEP or (len(self.joint) >= 3 and wmean > COLL_WIN):
            score = max(score, 0.9)
        for c in self.ew:
            self.ew[c] = (1 - EW_LAM) * self.ew[c] + EW_LAM * z[c]
            self.short[c].append(max(-6.0, min(6.0, z[c])))
            sm = sum(self.short[c]) / len(self.short[c])
            if abs(self.ew[c]) > EW_THR and abs(sm) > EW_SHORT and \
                    self.ew[c] * sm > 0 and len(self.short[c]) >= 4:
                score = max(score, 0.8)
        for c in PHI:
            self.slow_recent[c].append(z[c])
            base = self.slow_base[c]
            if len(base) >= 130 and len(self.slow_recent[c]) == 8:
                old = list(base)[:100]
                b = sum(old) / len(old)
                dev = sum(self.slow_recent[c]) / 8.0 - b
                if abs(dev) > SLOW_THR:
                    score = max(score, 0.7)
            if score < 0.5:
                base.append(z[c])

        self.prev = {c: max(-4.0, min(4.0, z[c])) for c in PHI}
        self.last_flag = score >= 0.5
        return float(min(1.0, max(0.0, score)))