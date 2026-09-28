"""A LeJEPA-style visual encoder for per-object crops of the head camera.

The entity tokens a conditioned policy reads carry geometry and a type one-hot,
but nothing about what an object *looks like*. This package learns that part
without labels: crops of each tracked object are cut from the head image using
the recorded pose and the static head camera (`crops`), and a small ViT is
trained on them with the LeJEPA objective -- a view-prediction loss plus
SIGReg, which keeps the embedding distribution isotropic Gaussian and so
cannot collapse without paying for it (`model`, `sigreg`, `train`).

Needs numpy and torch, so it is imported only by the policy environment.
"""
