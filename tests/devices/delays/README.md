# Compiled delay fixtures

`fixed_delay.va`, `variable_delay.va`, `mode_delay.va` and `reciprocal_delay.va` are vendored OSDI 0.5 fixtures captured
on 2026-11-24. The scalar-port waveguide and its discipline include are vendored optical fixtures; their Apache-2.0 SPDX
notices are preserved. The fixture sources are maintained locally for reproducible qualification.

The supplied `variable_delay` has an explicit maxdelay and a parameter-only duration; `mode_delay` changes its signal by
analysis mode. `duration_voltage.va` and `duration_mode.va` add the corresponding voltage-dependent and
analysis-dependent duration cases. These differences are intentional and tested explicitly.

Tests compile all fixtures with `BOSDI_OPENVAF_05`, falling back to `openvaf-r` on PATH. Use a compiler supporting OSDI
0.5 and `absdelay`. The qualification compiler SHA-256 and commands are in the checked-groups benchmark report.
