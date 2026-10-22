"""Exit tests and the two cross-cutting tracks.

Every exit test in this package runs on grids the model never saw. That is the
only number that supports the word "foundation", so the held-out guard lives
here and is not optional.

The ablation harness also lives here: at 50M, 100M and 300M, one arm from the
language checkpoint and one from random initialisation, with the same encoder,
the same data and the same budget. Reported at Phase 4 and again at Phase 7.
If the language arm does not win, that is a result worth publishing.
"""
