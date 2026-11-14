import csv
from decimal import Decimal, ROUND_HALF_UP

# 1. Read file, drop the preamble (blank lines and # comment lines)
with open("test-input.csv", encoding="utf-8-sig") as f:
    lines = [l for l in f.read().splitlines() if l.strip() and not l.lstrip().startswith("#")]
rows = csv.DictReader(lines, delimiter=";")  # first remaining line = header


def fix_name(name):
    """Undo mojibake: UTF-8 text wrongly decoded as cp1252/latin-1."""
    name = name.strip()
    for _ in range(3):
        fixed = None
        for enc in ("cp1252", "latin-1"):
            try:
                fixed = name.encode(enc).decode("utf-8")
                break
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
        if fixed is None or fixed == name:
            break
        name = fixed
    return name


def to_celsius(raw, unit):
    t = Decimal(raw.strip().replace(",", "."))  # 54,6 -> 54.6
    is_c = unit.strip().upper() != "F"
    if not is_c:
        t = (t - 32) * 5 / 9                    # F -> C
    t = t.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    if t == 0:
        t = abs(t)
    return f"{t:.1f}", is_c


# 2. Remove duplicates by reading_id (prefer the original Celsius row)
seen = {}
for r in rows:
    rid = int(r["reading_id"])
    temp, is_c = to_celsius(r["temperature_raw"], r["unit"])
    rec = (fix_name(r["station_name"]), temp, r["reading_date"].strip(), is_c)
    if rid not in seen or (is_c and not seen[rid][3]):
        seen[rid] = rec

# 3. Write sorted by reading_id
with open("test-output.csv", "w", encoding="utf-8", newline="") as f:
    f.write("reading_id,station_name,temperature_c,reading_date\n")
    for rid in sorted(seen):
        n, t, d, _ = seen[rid]
        f.write(f"{rid},{n},{t},{d}\n")