# Runners

Every dataset must regenerate from a script here. That is the Phase 0 exit
test, so nothing is generated from a notebook or an interactive session.

Order of operations for Phase 0:

    gen_oracle.py          # start this FIRST, it runs for months
    gen_transitions.py     # parallel across cores
    gen_cascades.py        # derived from the same rollouts
    report_corpus.py       # near-congestion fraction, counts, held-out audit

Later phases:

    train_encoder.py       # phase 1
    adapt_language.py      # phase 2
    train_alignment.py     # phase 3
    train_dynamics.py      # phase 4
    build_reasoning_corpus.py / train_reasoning.py   # phase 5
    train_verified.py      # phase 6, stretch
    train_action_head.py   # phase 7
    run_gridstar.py        # phase 8
    run_exit_test.py       # any phase, held-out only
    run_ablation.py        # both arms, all three sizes
