"""Phases 5 and 6: explanation.

Phase 5, reasoning corpus and supervised training. The simulator supplies
facts, Claude supplies prose, never the other way round. Templates render the
bulk of the corpus from the Phase 0 cascade logs; a few thousand hand-written
examples cover variety and register. Counterfactual pairs are included: same
state, action A fails, action B survives, explain the difference.

Exit test: on held-out blackouts, names the correct first trip and correct trip
order above a stated threshold, scored automatically against the logs and not
by reading samples.

Phase 6, verified reasoning, is a stretch: the model makes a checkable claim,
simulate scores it, correct claims are rewarded. Novel if reached, and the
first phase to cut. Bank Phases 1 to 5 first.
"""
