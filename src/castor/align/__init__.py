"""Phase 3: alignment. Language model frozen, encoder frozen.

Trains only the cross attention and projection layers, on three grounding
tasks, none solvable from one channel alone:

  1. from sensors, name the three most loaded lines in words
  2. from a text instruction, output the bus assignment
  3. given a sensor state and a text claim about it, judge true or false,
     scored by the simulator

Element names come from castor.naming, identically in both channels.
Exit test: names the three most loaded lines correctly on a held-out grid.
"""
