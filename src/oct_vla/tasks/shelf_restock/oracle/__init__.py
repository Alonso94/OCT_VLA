"""Simulation-only expert generation: world model, grasps, motion, placement.

Uses privileged simulator state (exact poses, cuRobo's collision world). Never
imported by policy, training, or runtime code -- see docs/architecture.md.
"""
