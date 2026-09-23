"""The object-conditioning method, independent of any backbone.

Four files hold the whole method; a backbone adapter (``control_*``) only says
where in its network each piece attaches.

========== ============================================================
entity.py  what an entity is, and its embedding (one shared contract)
kv.py      KV: ControlVLA's added, zero-initialised K/V attention term
adaln.py   AdaLN: a pooled scene vector modulating the host's blocks
tokens.py  tokens: entity tokens placed in the host's own sequence
hosts.py   how KV attaches to each host attention implementation
========== ============================================================

The three arms a run can select (``object_conditioning``) are nested, so each
isolates one addition: ``kv``, ``kv_adaln`` = kv + AdaLN, ``kv_tokens`` = kv +
tokens.
"""
