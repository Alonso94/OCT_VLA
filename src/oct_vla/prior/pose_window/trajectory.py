"""Moved to ``oct_vla.bpfn.data.syn_data.trajectory``; this name still resolves (pickled packs refer to it)."""
import sys

from oct_vla.bpfn.data.syn_data import trajectory as _moved

sys.modules[__name__] = _moved
