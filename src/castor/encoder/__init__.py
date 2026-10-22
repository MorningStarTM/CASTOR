"""Phase 1: element encoder. Trained alone, with no language model attached.

One token per line, generator, load and substation, with learned type
embeddings and grid adjacency carried in the attention. Objective is masked
element reconstruction plus a power balance consistency loss at each bus.

Exit test: reconstruction error on the held-out grids within a reasonable
factor of the training grids. This runs before anything expensive, precisely
to catch an encoder that overfits to one grid shape.
"""
