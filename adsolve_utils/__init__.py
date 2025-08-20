"""
Compatibility shim so `adsolve_utils.models` and `adsolve_utils.evaluation_bundles`
work while the real packages live at top level as `models` and `evaluation_bundles`.
"""
from importlib import import_module as _imp
import sys as _sys

_sys.modules[__name__ + ".models"] = _imp("models")
_sys.modules[__name__ + ".evaluation_bundles"] = _imp("evaluation_bundles")
