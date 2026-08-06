# Evaporator Capacity

## Scope

Add one missing P5 source channel and one derived scientific channel. Do not change Dataset structure, cycle processing, plots, analysis policy, or existing formulas.

## Channels

`compressor_power` reads `p5__压机功率`. The raw value is in W and is converted to kW with `scale: 0.001`.

`evaporator_capacity` is a derived performance channel in kW:

```text
evaporator_capacity = heating_capacity - compressor_power
```

If either dependency is unavailable, `evaporator_capacity` is null. Its `__imputed` flag follows the existing dependency-imputation rule.

## Implementation

Extend the existing channel YAML and named-formula dispatch. Do not add a generic arithmetic abstraction or special Dataset code.

## Verification

Add one focused formula test proving that `10.0 kW - 2.428 kW = 7.572 kW` and that a missing dependency produces null. Existing channel/config and Process tests must continue to pass.
