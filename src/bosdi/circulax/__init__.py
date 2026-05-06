"""circulax integration layer for bosdi.

Provides :func:`osdi_component` and :class:`OsdiComponentGroup` / :class:`OsdiModelDescriptor`
for use with :func:`circulax.compiler.compile_netlist`.

Install via::

    pip install circulax[verilog-a]

which pulls in ``bosdi`` as a dependency.  Import directly from either namespace::

    from bosdi.circulax import osdi_component          # bosdi-first style
    from circulax import osdi_component                # circulax-first style (after install)
"""

from bosdi.circulax.osdi_component import (
    OsdiComponentGroup,
    OsdiModelDescriptor,
    _BOSDI_AVAILABLE,
    _BOSDI_ERR,
    osdi_component,
)

__all__ = [
    "OsdiComponentGroup",
    "OsdiModelDescriptor",
    "osdi_component",
    "_BOSDI_AVAILABLE",
    "_BOSDI_ERR",
]
