"""BimanualPFN: one completed demonstration and a scene history -> both wrists' next poses.

``paper/BimanualPFN.tex``; package layout and stage runners: ``docs/bpfn.md``.
Every pose is a seven-value cell ``[x, y, z, qx, qy, qz, qw]`` (metres, unit xyzw
quaternions, workcell frame). Everything below ``models/`` is numpy-only, so the data code
runs in both the policy and the RoboTwin interpreter.
"""
