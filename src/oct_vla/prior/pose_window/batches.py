"""Moved to ``oct_vla.bpfn.data.batches``; this name still resolves (pickled packs refer to it)."""
import sys

from oct_vla.bpfn.data import batches as _moved

sys.modules[__name__] = _moved
