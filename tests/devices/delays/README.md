# Compiled delay fixtures

`fixed_delay.va`, `variable_delay.va`, `mode_delay.va` and `reciprocal_delay.va` are copied unchanged from
`circulox/circulox/crates/engine/tests/fixtures/osdi5/` on 2026-11-24. `waveguide.va` and its discipline include are
copied from Circulox's canonical `circulox/model-libraries/circulox-base_models-va/src/photonics/` directory; the
waveguide's Apache-2.0 SPDX notice is preserved. This test suite does not modify the Circulox checkout.

The supplied `variable_delay` has an explicit maxdelay and a parameter-only duration; `mode_delay` changes its signal by
analysis mode. `duration_voltage.va` and `duration_mode.va` add the corresponding voltage-dependent and
analysis-dependent duration cases. These differences are intentional and tested explicitly.

Tests compile all fixtures with `CIRCULOX_OPENVAF`, falling back to `openvaf-r` on PATH. Use a compiler supporting OSDI
0.5 and `absdelay`. The qualification compiler SHA-256 and commands are in the checked-groups benchmark report.
