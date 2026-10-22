"""Phase 2: the language arms, pretrained here rather than downloaded.

The plan said "pick the three checkpoints". Pretraining our own instead costs
a few extra weeks but buys a cleaner ablation: off-the-shelf checkpoints at
different sizes differ in architecture, tokenizer and training data, so
"language init beats random init" would be confounded three ways. Arms built
here are identical but for the weights.

    config.py       GPTConfig and PretrainConfig; parameter maths per arm
    tokenizer.py    the pretrained tokenizer and the grid-name audit
    corpus.py       stream the mixture, tokenize, pack to uint16 shards
    dataset.py      memory-mapped batching over those shards
    model.py        builds gpt.py into a configured arm
    schedule.py     warmup then cosine decay
    checkpoint.py   atomic saves, resume, run manifests
    pretrain.py     the training loop
    perplexity.py   held-out loss, and later the Phase 2 exit test

    gpt.py          the model itself

Two arms, both under 100M parameters:

    50m    d=512, L=8,  H=8    42.1M params, ~2.5B tokens
    100m   d=768, L=10, H=12   96.1M params, ~5B tokens

The random-initialisation arm of the ablation is not pretrained. It is the
same GPTConfig handed to build_model and trained from scratch in Phase 4.

Note on model.py: gpt.py holds its hyperparameters in module-level globals,
so build_model sets them immediately before constructing the model. That shim
is not thread safe and exists only so gpt.py stays unmodified. Giving
GPTLanguageModel a GPTConfig argument would remove it.
"""
