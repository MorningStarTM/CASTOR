"""Phase 4: cascade dynamics. This is Paper 1.

Encoder and cross attention unfrozen; language model stays on LoRA.

Input is a short state history plus a candidate action in structured bus
configuration form. Output is the rho vector over the next steps, overflow
timers, trip events and episode termination.

Targets are the physics, not the reward: reward is a design choice someone
made, rho is real.

Headline numbers, all on held-out grids: rho prediction accuracy against
obs.simulate, cascade onset accuracy (does it call the right first trip), and
speedup factor over the solver.
"""
