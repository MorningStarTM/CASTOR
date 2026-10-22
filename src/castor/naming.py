"""Stable element names, shared by the sensor channel and the text channel.

Phase 3 depends on this module being the single source of truth: "line 12" in
prose and the embedding at the index this module assigns to line 12 must be the
same object. Both channels import from here; neither invents its own names.

Names must be stable across grids of different sizes, so they are derived from
grid2op's own element ordering plus the element type, never from a position in
a flattened observation vector.
"""
