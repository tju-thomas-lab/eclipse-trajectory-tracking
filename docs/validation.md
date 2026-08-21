# Validation plan

Automated tests cover rolling-window timestamps, event merging, normalized change boxes, schemas,
localhost endpoint enforcement, caching signatures, and a synthetic video end-to-end run. CI is not
yet configured because recordings and models must remain local.

Before using output as research labels:

1. Sample events across plan creation, geometry, optimization, dose, DVH, context switches, and idle
   periods.
2. Have qualified reviewers independently mark action type, target, values, support, and boundaries.
3. Report agreement and precision/recall separately for proposal detection and semantic inference.
4. Measure false clinical-rationale claims and identifier leakage explicitly.
5. Treat Eclipse performance as unknown until this validation is completed.

