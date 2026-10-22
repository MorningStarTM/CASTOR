"""grid2op environment construction and sampling.

Backend is lightsim2grid throughout. Two things here are not defaults and
matter for Phase 0 data quality:

  * the opponent module is enabled
  * chronics start near congestion, otherwise most transitions are states
    where nothing is near a limit

Held-out environments are declared in configs/envs/heldout.yaml and this
package refuses to build them outside of evaluation.
"""
