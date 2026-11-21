"""Moved to ``oct_vla.bpfn.data.packs``; this name still resolves (pickled packs refer to it)."""
import sys

from oct_vla.bpfn.data import packs as _moved

sys.modules[__name__] = _moved
