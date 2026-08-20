# Figure 1 QA contract

- Core conclusion: under the observed fixed-duration defrost policy, raw-data renewal cost identifies an empirical optimum before the observed defrost in many complete cycles, but the broad near-optimal regions limit point-label precision.
- Archetype: quantitative grid with one representative raw-data example and cross-cycle validation.
- Backend: Python/matplotlib only.
- Final size: 183 mm wide; 7.2 × 6.2 in working canvas.
- n definition: 47 complete observed-policy cycles in the primary figure; 12 sensor-end right-censored cycles are sensitivity-only; 50 valid defrost/recovery tickets.
- Center/spread: panel d reports all cycle values and a box plot (median, interquartile range, 1.5×IQR whiskers); no hypothesis test is claimed.
- Source data: `source_data/cycle_optimal_points.csv`, `source_data/candidate_cost_curves.parquet`, and `source_data/defrost_ticket_events.csv`.
- Editable exports: SVG text preserved; PDF uses TrueType fonts; TIFF is 600 dpi; PNG is 300 dpi.
- Image integrity: no microscopy or image manipulation in this figure; panel a uses unsmoothed original sensor points.
- Representative cycle: frost_cycle_000070, selected as the interior-minimum cycle nearest the cohort median advance.
