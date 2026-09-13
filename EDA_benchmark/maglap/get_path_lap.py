"""Re-exports the path-complex Laplacian PE transform from the repo-root
``utils/get_path_lap.py`` (and, transitively, ``pathhomology/``), so EDA_benchmark
doesn't need its own copy of that math.

This can't be a plain ``from utils.get_path_lap import ...``: EDA_benchmark already
has its own top-level ``utils.py`` (imported everywhere as ``from utils import ...``),
which would shadow or be shadowed by the root ``utils/`` package depending on
``sys.path`` order. Loading the target file directly by path sidesteps the name
collision entirely.
"""
import importlib.util
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

_spec = importlib.util.spec_from_file_location(
    "_shared_get_path_lap", os.path.join(_ROOT, "utils", "get_path_lap.py")
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

AddPathLaplacianEigenvectorPE3 = _module.AddPathLaplacianEigenvectorPE3
build_two_path_index = _module.build_two_path_index
realign_edge_pe = _module.realign_edge_pe
