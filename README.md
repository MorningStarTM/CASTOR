# CASTOR

**CAS**cade aware **TO**pology **R**easoning.

A grid2op foundation model that learns cascade dynamics from sensor data,
explains why a topology fails, and outputs bus configurations that keep the
grid alive. The simulator is ground truth throughout.

The baseline to beat is the LLM guided Safety SAC work, which uses a frozen
Qwen 2.5 7B to clean the replay buffer every 200 training steps. It has no
world model and no transfer across grids. Both of those are where CASTOR lives.

The research plan, the module layout and the risk list live in `docs/`, which
is kept local rather than committed. Ask for a copy if you are picking this up.

## Layout

    configs/        one directory per phase, plus envs/ and experiment/
      envs/         train.yaml and heldout.yaml, the single source of the split
      data/         the three Phase 0 dataset configs
      experiment/   the ablation that runs through every phase
    src/castor/
      naming.py     stable element names, shared by both channels
      envs/         grid2op construction, opponent, near-congestion chronics
      data/         phase 0: transitions, cascade logs, oracle actions
      encoder/      phase 1: element encoder
      language/     phase 2: size arms and domain adaptation
      align/        phase 3: cross attention and the grounding tasks
      dynamics/     phase 4: cascade surrogate -> paper 1
      reasoning/    phases 5 and 6: explanation, then verification
      action/       phase 7: bus configuration head
      deploy/       phase 8: GridStar integration -> paper 2
      eval/         exit tests, held-out guard, ablation harness
    scripts/        every dataset and number regenerates from here
    tests/          unit tests, plus the held-out guard as an integration test

Not committed: data/ and experiments/ (generated), docs/, papers/ and
notebooks/ (local working material).

## Two tracks run through every phase

**The ablation.** At 50M, 100M and 300M: one arm from the language checkpoint,
one from random initialisation. Same encoder, same data, same budget. Reported
at Phase 4 and again at Phase 7. If the language arm does not win, that is a
result worth publishing and it saves a year.

**The held-out grids.** Every exit test runs on grids the model never saw.
That is the only number that supports the word foundation.

## Install

    pip install -e ".[train,dev]"

## Where to start

Phase 0. Start `scripts/gen_oracle.py` first, because it runs for months and
Phase 7 blocks on it. Before generating a single transition, name the two
held-out environments in `configs/envs/heldout.yaml`: changing that split
later invalidates every transfer number.
