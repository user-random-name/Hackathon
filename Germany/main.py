"""Recover a five-variable function with linear effects, one interaction and a step."""
import itertools
import math
import numpy as np
from scipy.optimize import linprog

NAMES = [f'x{i}' for i in range(1, 6)]

def solve(oracle, points):
    center = [0.2] * 5
    probes = [center[:]]
    for i in range(5):
        p = center[:]; p[i] = 0.4; probes.append(p)
    for i, j in itertools.combinations(range(5), 2):
        p = center[:]; p[i] = p[j] = 0.4; probes.append(p)
    values = oracle.ask_many(probes, phase='low-region polynomial')
    base = values[0]
    slopes = [(values[i + 1] - base) / .2 for i in range(5)]
    pairs = list(itertools.combinations(range(5), 2))
    interactions = [(values[6 + k] - values[i + 1] - values[j + 1] + base) / .04
                    for k, (i, j) in enumerate(pairs)]
    k = max(range(10), key=lambda t: abs(interactions[t]))
    a, b = pairs[k]
    coef = interactions[k]
    def smooth(x):
        return base + sum(slopes[i] * (x[i] - .2) for i in range(5)) + coef * (x[a] - .2) * (x[b] - .2)

    # Discover a step from corners and interior points, using the smooth model.
    corners = [[float((mask >> i) & 1) for i in range(5)] for mask in range(32)]
    rng = np.random.default_rng(12345)
    random_points = rng.uniform(0, 1, (48, 5)).tolist()
    samples = corners + random_points
    ys = oracle.ask_many(samples, phase='step detection')
    residuals = np.array([y - smooth(x) for x, y in zip(samples, ys)])
    # The most common residual is zero; the other cluster is the jump.
    jumps = [float(v) for v in residuals if abs(v) > .08]
    jump = float(np.median(jumps)) if jumps else 0.0
    labels = np.array([abs(v - jump) < abs(v) for v in residuals], dtype=int) if jumps else np.zeros(len(samples), dtype=int)

    gate = None
    if jumps and 0 < sum(labels) < len(labels):
        # Find the pair whose 2D coordinates can linearly separate the labels.
        for i, j in pairs:
            X = np.array([[p[i], p[j]] for p in samples])
            sign = 2 * labels - 1
            # Margin-maximising linear separator (L1 regularised).
            A = np.column_stack((-sign[:, None] * X, sign, np.ones((len(X), 1))))
            # Variables are w1,w2,threshold,margin; maximise margin with bounded w.
            res = linprog([0, 0, 0, -1], A_ub=A, b_ub=np.zeros(len(X)),
                          bounds=[(-1, 1), (-1, 1), (-2, 2), (None, None)], method='highs')
            if res.success and res.x[3] > 1e-5:
                gate = (i, j, res.x[:3]); break

    if gate:
        i, j, initial = gate
        # Locate actual step crossings with bisection, then fit the boundary.
        boundary_points = []
        w1, w2, threshold = initial
        for varying, fixed in [(i, j), (j, i)]:
            for fixed_value in (0., .35, .7, 1.):
                def point(t):
                    p = [.2] * 5; p[fixed] = fixed_value; p[varying] = t
                    return p
                p0, p1 = point(0.), point(1.)
                y0, y1 = oracle.ask_many([p0, p1], phase='boundary endpoints')
                c0 = abs((y0 - smooth(p0)) - jump) < abs(y0 - smooth(p0))
                c1 = abs((y1 - smooth(p1)) - jump) < abs(y1 - smooth(p1))
                if c0 == c1: continue
                lo, hi = 0., 1.
                for _ in range(9):
                    mid = (lo + hi) / 2
                    pm = point(mid)
                    ym = oracle.ask(pm, phase='boundary bisection')
                    cm = abs((ym - smooth(pm)) - jump) < abs(ym - smooth(pm))
                    if cm == c0: lo = mid
                    else: hi = mid
                bp = point((lo + hi) / 2)
                boundary_points.append([bp[i], bp[j]])
                if oracle.remaining < 8: break
            if oracle.remaining < 8: break
        if len(boundary_points) >= 2:
            P = np.array(boundary_points)
            _, _, vt = np.linalg.svd(P - P.mean(axis=0))
            normal = vt[-1]
            if np.dot(normal, [w1, w2]) < 0: normal = -normal
            w1, w2 = normal.tolist()
            threshold = float(np.dot(P.mean(axis=0), normal))
        gate = (i, j, (float(w1), float(w2), float(threshold)))

    relevant = {idx for idx, s in enumerate(slopes) if abs(s) > .03}
    if abs(coef) > .03: relevant.update([a, b])
    if gate: relevant.update(gate[:2])
    def predict(x):
        y = smooth(x)
        if gate:
            i, j, (w1, w2, t) = gate
            if w1 * x[i] + w2 * x[j] >= t: y += jump
        return float(y)

    result = {
        'relevant_variables': [NAMES[i] for i in sorted(relevant)],
        'irrelevant_variables': [NAMES[i] for i in range(5) if i not in relevant],
        'interactions': [{'variables': [NAMES[a], NAMES[b]], 'type': 'product',
                          'coefficient': float(coef)}] if abs(coef) > .03 else [],
        'discontinuities': [],
        'predictions': [predict(p) for p in points],
        'notes': 'Low-region finite differences; corner and interior step detection; linear separator and binary-search boundary.'
    }
    if gate:
        i, j, (w1, w2, t) = gate
        result['discontinuities'] = [{
            'gate_variables': [NAMES[i], NAMES[j]],
            'boundary': {'form': 'linear', 'coefficients': {NAMES[i]: w1, NAMES[j]: w2},
                         'threshold': t, 'active_when': 'greater_equal'},
            'jump': float(jump)
        }]
    return result
