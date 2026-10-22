"""Phase 2: language side preparation. Runs in parallel with Phase 1.

Three size arms at 50M, 100M and 300M. Light domain adaptation on blackout
investigation reports, grid codes and topology control literature: LoRA, low
learning rate, small corpus. Full finetuning here destroys the general ability
the checkpoint is being paid for.

Exit test: perplexity drops on held-out domain text without wrecking general
benchmarks.
"""
