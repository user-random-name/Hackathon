import lagoon as L

V = [dict() for _ in range(L.STEPS_PER_DAY + 1)]
policy = {}
cells = [(r, c) for r in range(L.ROWS) for c in range(L.COLS) if L.is_water(r, c)]

for t in range(1, L.STEPS_PER_DAY + 1):
    for (r, c) in cells:
        pb = L.BITE_CHANCE.get(L.GRID[r][c], 0.0)
        for h in range(L.HOLD_CAPACITY + 1):
            best, best_a = -1.0, "WAIT"
            for a in L.ACTIONS:
                q = 0.0
                for bite, pbite in ((True, pb), (False, 1 - pb)):
                    for tide, ptide in ((True, L.TIDE_CHANCE), (False, 1 - L.TIDE_CHANCE)):
                        p = pbite * ptide
                        if p == 0:
                            continue
                        nr, nc, nh, landed = L._advance(r, c, h, a, bite, tide)
                        q += p * (landed + V[t - 1].get((nr, nc, nh), 0.0))
                if q > best + 1e-12:
                    best, best_a = q, a
            V[t][(r, c, h)] = best
            policy[(r, c, h, t)] = best_a

print("Expected fish per day from the start:", round(V[L.STEPS_PER_DAY][(*L.START, 0)], 3))
L.export_main(policy, "main.py")
print("main.py written")