"""BimanualPFN: one completed demonstration and a scene history -> both wrists' next poses.

``paper/BimanualPFN.tex``. Every pose is a seven-value cell ``[x, y, z, qx, qy, qz, qw]``
(metres, active unit xyzw quaternions, workcell frame). The model predicts the next H
poses of the left and right wrist, ``[H, 2, 7]``.

This package is numpy-only below the model, so the generator, fixtures, scorer and
renderer run in either interpreter (the policy venv and the RoboTwin venv).
"""
