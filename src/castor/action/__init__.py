"""Phase 7: action head.

Bus configuration head, behaviour cloned from the Phase 0 oracle actions with
Phase 5 reasoning in the context. Decoding to the gym discrete id happens only
at the environment boundary, never inside the model.

Exit test: survival steps on the official tracks against do nothing, greedy
simulate, earlier agents, and the LLM guided Safety SAC baseline.
"""
