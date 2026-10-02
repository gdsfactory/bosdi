"""circulax integration layer for bosdi.

Provides :func:`osdi_component`, :func:`va_component`, and supporting classes
for use with :func:`circulax.compiler.compile_netlist`.

Install via::

    pip install circulax[verilog-a]

which pulls in ``bosdi`` as a dependency.  Import directly from either namespace::

    from bosdi.circulax import osdi_component          # bosdi-first style
    from circulax import osdi_component                # circulax-first style (after install)
"""

from importlib import import_module
from typing import TYPE_CHECKING

from bosdi.circulax.osdi_component import (
    OsdiComponentGroup,
    OsdiModelDescriptor,
    _BOSDI_AVAILABLE,
    _BOSDI_ERR,
    osdi_component,
)

if TYPE_CHECKING:
    from bosdi.circulax.va_component import JacobianReturn, va_component


def __getattr__(name: str):
    """Load the decorator only when requested; native OSDI needs no circulax."""
    if name in {"JacobianReturn", "va_component"}:
        module = import_module("bosdi.circulax.va_component")
        # Importing the submodule sets the package attribute to that module.
        # Restore the public function export and cache both lazy exports.
        globals().update(
            JacobianReturn=module.JacobianReturn, va_component=module.va_component
        )
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "JacobianReturn",
    "OsdiComponentGroup",
    "OsdiModelDescriptor",
    "osdi_component",
    "va_component",
    "_BOSDI_AVAILABLE",
    "_BOSDI_ERR",
]
