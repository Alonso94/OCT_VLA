"""Moved to ``oct_vla.bpfn.benchmark_tools.metrics``; this name still resolves."""
import sys

from oct_vla.bpfn.benchmark_tools import metrics as _moved

sys.modules[__name__] = _moved
