"""Moved to ``oct_vla.bpfn.schema``; this name still resolves (pickled packs refer to it)."""
import sys

from oct_vla.bpfn import schema as _moved

sys.modules[__name__] = _moved
