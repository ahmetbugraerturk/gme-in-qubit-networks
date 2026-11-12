"""qgpu: array-module-generic (NumPy / CuPy) versions of the atlas entanglement stages plus the
batched GMN ADMM solver.  Every function takes `xp` (numpy or cupy) as its first argument."""
from . import core, gmn_admm  # noqa: F401
