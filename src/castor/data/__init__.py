"""Phase 0: corpus generation. Everything downstream trains on this.

Three datasets come out of this package:

  1. transitions   - state, structured action, next state, rho vector,
                     overflow timers, trip events
  2. cascade_logs  - for every blackout, the ordered trip chain with rho
                     values and timings (raw material for Phase 5)
  3. oracle        - for sampled states, brute force obs.simulate over the
                     reduced action set, recording the best bus configuration

The oracle is expensive and runs for months in the background. Start it first.
Every dataset must regenerate from a script, and the near-congestion fraction
is tracked from day one.
"""
