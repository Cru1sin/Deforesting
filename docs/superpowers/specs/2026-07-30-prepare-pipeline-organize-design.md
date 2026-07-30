# Prepare Pipeline Refactor Design

## Goal

Refactor the `organize` branch prepare pipeline so that its orchestration
names data contracts clearly, each date owns its camera mapping, cycle quality
is validated in `data/cycles.py`, image wide-table construction lives in
`data/alignment.py`, and the final output is explicit and auditable.

## Decisions

- `timestamp` is the canonical time column for `cycles.py`, `alignment.py`,
  and `build_cycle_summary()`.
- Registry output contains `operating_mode` as the numeric/source value and
  `is_heating` as the derived boolean value. The configured heating value is
  normally `3` but is not hard-coded in cycle validation.
- `data/<MMDD>/IPlocation.yaml` takes precedence over the legacy global camera
  mapping. The legacy mapping remains a compatibility fallback when the local
  file is absent.
- Image output columns are keyed by configured `camera_role`, never by IP or
  device directory. Role values used for output columns are stable schema keys
  such as `front`, `left`, and `top`.
- Cycle status has exactly three top-level values: `valid`, `incomplete`, and
  `invalid`. Details stay in `cycle_summary.csv` as `cycle_status_reason`.
- Sensor coverage is the valid sensor observation time span divided by the
  cycle span. RGB coverage is the valid multiview-group time span divided by
  the cycle span. Joint coverage is the smaller of the two.
- Prepare never interpolates. Any `__interpolated` column reaching the output
  boundary raises an error; raw audit columns are excluded by explicit output
  selection rather than suffix deletion.
- State timestamps are UTC ISO-8601 strings and include output paths plus
  configuration and registry fingerprints.

## Data flow

```text
raw directory
  -> file_inventory / source_field_inventory
  -> sensor_load_result
  -> registry_specs / registry_result
  -> prepared_data
  -> cycle_segmentation / cycle_validation
  -> image_records / multiview_index / image_alignment
  -> cycle_summary
  -> warnings / metrics / PrepareResult
```

## Interfaces

The pipeline will expose the following readable boundaries:

```python
cycle_validation = validate_cycles(
    cycle_segmentation,
    config.prepare.cycle_validation,
)

prepared_data = attach_cycle_fields(prepared_data, cycle_validation)
prepared_data = attach_image_paths(prepared_data, image_alignment)

return PrepareResult(
    prepared_data=prepared_data,
    cycle_summary=cycle_summary,
    warnings=tuple(warnings),
    metrics=metrics,
)
```

The implementation keeps existing public stage entry points and downstream
processing boundaries. Internal processing modules may continue to use their
own `sensor_time` representation after the prepare artifact has been written;
prepare itself will no longer use an anonymous `internal` rename.

## Validation and publishing

`validate_prepare_result()` checks the prepared schema, timestamp ordering,
absence of interpolation artifacts, cycle coverage ranges, and required image
behavior. `publish_prepare_result()` writes the prepared table, cycle summary,
and state through atomic temporary-file replacement. The refactor does not
silently repair malformed inputs.

## Testing

Tests will cover the new names and contracts, local mapping precedence, stable
role-based image columns, required alignment columns, three-state status
normalization, NaN-safe issue appending, sampling interval inference, coverage
fractions, interpolation rejection, ISO state output, and end-to-end prepare
orchestration with a small fixture.
