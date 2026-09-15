"""Process bridge between the simulator and policy environments.

The two environments cannot be merged: RoboTwin pins Python 3.10, NumPy 1.26
and Torch 2.4 for SAPIEN/cuRobo's compiled extensions, while LeRobot's pi0.5
stack wants Python 3.12, NumPy 2.x and Torch 2.11. Closed-loop evaluation needs
both at once, so the simulator runs as a server in its own interpreter and the
policy drives it as a client from the other.

`protocol` is the only module both sides import, so it stays pure standard
library -- no NumPy, no Torch, nothing that could pull one environment's ABI
into the other. `server` imports RoboTwin; `client` never does.
"""
