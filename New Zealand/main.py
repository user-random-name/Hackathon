import heapq
from environment import load_map, _step

def solve(path):
    v = load_map(path)
    start = (v.start[0], v.start[1], v.initial_open, 0)
    key = lambda s: (s[0], s[1], s[2], s[3] % v.tide_period)
    best = {key(start): 0}
    parent = {key(start): None}
    pq = [(0, 0, start)]
    cnt = 0
    while pq:
        t, _, s = heapq.heappop(pq)
        k = key(s)
        if best.get(k, 1 << 30) < t:
            continue
        if (s[0], s[1]) == v.goal:
            out = []
            while parent[k] is not None:
                pk, a = parent[k]
                out.append(a)
                k = pk
            return "".join(reversed(out)), t
        for a in "UDLRW":
            r, c, down, nt, bad = _step(v, s[0], s[1], s[2], s[3], a)
            if bad:
                continue
            ns = (r, c, down, nt)
            nk = key(ns)
            if nt < best.get(nk, 1 << 30):
                best[nk] = nt
                parent[nk] = (k, a)
                cnt += 1
                heapq.heappush(pq, (nt, cnt, ns))
    return None, None

if __name__ == "__main__":
    route, t = solve("test-input.txt")
    with open("solution.txt", "w") as f:
        f.write(route + "\n")