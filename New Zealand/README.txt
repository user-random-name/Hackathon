main.py finds the fastest valid route from S to G in the 47 x 61 valley
(test-input.txt) and writes it to solution.txt as moves U D L R W.

How it works:
The problem is a shortest-path search where the cost is time in minutes.
A state is (row, column, set of bridges that are down, minute mod 16).
The minute mod 16 is needed because the tide repeats every 16 minutes, so
two visits to the same cell at different tide phases are different states.
The search is Dijkstra's algorithm with a priority queue ordered by the
real elapsed minute:
1. Start at S, minute 0, with the bridges from the map's open line down.
2. From each state try U, D, L, R and W (wait).
3. Every action goes through environment._step, the simulator's own rule
   function. It handles move time (1 minute, 3 onto a swamp, +1 for each
   height level climbed), raised bridges, winches that flip bridges, rock,
   deep water and fords that may only be ended on at minutes 0-4 mod 16.
   Illegal actions are skipped. Because the simulator's rules are reused,
   the solver cannot disagree with the checker.
4. A state is only kept if it is reached earlier than any previous visit
   to the same state.
5. The first time G is popped from the queue, the route is optimal. It is
   rebuilt by following the stored parent links back to the start.
Waiting is one of the actions, so the route can wait for low tide.
Nothing is random, so the same moves always give the same result.

Results:
Practice valley (train-input.txt): 114 minutes, same as train-output.txt.
Test valley (test-input.txt): 293 minutes (244 actions), valid.
Park ranger's best time is 297 minutes, so the route beats it.

Usage:
Put main.py, environment.py and test-input.txt in one folder, then run:
python main.py
It writes solution.txt and exits with code 0 in a few seconds.
To check the route:
python environment.py --map test-input.txt --replay-file solution.txt
This prints: valid run: 293 minutes (244 actions)
To draw it on the map:
python environment.py --map test-input.txt --replay-file solution.txt --render route.svg

Needs Python 3.10-3.12, standard library only (heapq).

Flag: shown by the checker after upload of a valid route under 297 minutes.