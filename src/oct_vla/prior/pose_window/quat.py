"""Moved to ``oct_vla.bpfn.quat``; this name still resolves (pickled packs refer to it)."""
import sys

from oct_vla.bpfn import quat as _moved

sys.modules[__name__] = _moved
