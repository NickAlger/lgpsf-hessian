# Column-owner phases (2026-10-09)

The plan for moving the row fit's owner-side phases (coarsening, baseline, re-score, assembly) to the ranks
that own the window's columns lives with the SPMD fit in lgpsf: `lgpsf/dev/column-owner-phases-plan.md`.
This bridge's part: build the reverse plan, pass the row's local spacing with the window, reduce the row
energies before the weighted symmetrization and feed its weights explicitly, keep the column layout of the
symmetric B as the row layout it already is, print the new phase timers; a flag `--pencil-column-phases`.
