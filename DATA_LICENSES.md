# Data licences and citation duties

This repository contains **no benchmark data**. It contains pointers, hashes and
index files. The data stay with their publishers.

| Resource | Publisher | Licence (as stated by us) | Status |
|---|---|---|---|
| PolyBench (39 tasks, 42,169 records) | Fudan | **Paper: CC BY-NC-ND 4.0. Data deposit (figshare, DOI 10.6084/m9.figshare.30917717): CC BY 4.0** | one of the two must be checked against the deposit page on D0 |
| PolyBench26 (8 properties, ~250k datapoints) | LLNL | TBD | **verify on D0 before redistributing anything** |
| OpenPoly (741 structures, 3,985 labels, 26 properties) | repo release | TBD | verify on D0 |
| MoleculeNet / PCQM4Mv2 | OGB / DeepChem | per-dataset | cross-hierarchy context only, never redistributed |

Rules we hold ourselves to:

1. **Non-commercial clause.** PolyBench's paper licence is CC BY-NC-ND. A course
   project is not commercial, but any figure we publish outside the course must
   check this first.
2. **No redistribution.** We do not commit raw CSVs, only sha256 digests plus a
   fetch script. Anyone cloning this repo downloads from the publisher.
3. **Derived artefacts are ours.** The split index files and the lambda / kappa
   tables are our own work and are released under MIT with the code.
4. **Attribute per release, never per family.** Four resources with similar names
   exist (OpenPoly / OPoly26 / PolyBench26 / an unrelated LLM PolyBench). Cite by
   DOI and scale, never by name alone.
