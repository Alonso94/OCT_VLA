"""Moved to ``oct_vla.bpfn.data.bank``; this name still resolves (pickled packs refer to it)."""
import sys

from oct_vla.bpfn.data import bank as _moved

sys.modules[__name__] = _moved
