"""Moved to ``oct_vla.bpfn.data.syn_data.branches``; this name still resolves (pickled packs refer to it)."""
import sys

from oct_vla.bpfn.data.syn_data import branches as _moved

sys.modules[__name__] = _moved
