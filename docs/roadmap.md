The requested outcome remains a complete, behavior-preserving Rust translation
of local lavaan. This implementation is the first increment. The inventory is
the backlog: a nearby formula, happy-path test or mathematical specification does
not establish completion of an R component.

1. Extend foundations: general symmetric inversion/preconditioning, inverse
   downdates, numerical derivatives, sample Gamma and regression kernels;
   conditioning/stress fixtures and proofs for indexing loops and allocation.
2. Complete syntax, parameter tables/options, defaults, constraints,
   identification, starts/bounds and continuous ML/GLS variants. Extend inference
   to observed/robust information, tests, residuals, standardization, modification
   indices and prediction; match R warnings and failures.
3. Complete missing-data and sampling-weight inference, conditional/fixed-x,
   multigroup/multilevel pipelines, categorical moments, polychorics, thresholds,
   WLS/DWLS/DLS/PML/MML and correction families.
4. Port EFA/rotations, growth/SEM convenience APIs, SAM/MIIV, bootstrap, simulation,
   lists/multiple imputation, inspection/export and remaining utilities.
5. Close numerical/proof gaps, expand cross-platform CI and corpus/fuzz coverage,
   then optimize measured bottlenecks and compare complete fitting workflows
   with equivalent preprocessing and inference costs.

Each component needs a source mapping, explicit domain, R equivalence including
branches/failures, appropriate invariant checks, relevant completed Lean/Aeneas
proofs where supported, and reproducible benchmark evidence. Unsupported work
stays visible in `reports/coverage.json`; no unavailable estimator is advertised
as working.
