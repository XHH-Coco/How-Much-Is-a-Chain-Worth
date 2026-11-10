# Appendix A 鈥?the 39 tasks as released

Names and record counts read from the platform's `/api/data-counts` endpoint on
2026-11-11; they sum to **42,169**, reproducing the published total. The `Tier`
column is filled **only where the benchmark's main text names that property or its
family**. Anything else is `pending S27` and is **not eligible** for the main table.

| # | Task ID (dataset name) | n | Tier | Role |
|---|---|---|---|---|
| 1 | Glass transition temperature | 8,055 | 1 | A#1 |
| 2 | Thermal decomposition temperature | 4,898 | 1 | reserve |
| 3 | Melting temperature | 3,796 | 1 | A#2 |
| 4 | bandgap chain_Egc | 3,380 | 1 | reserve |
| 5 | Intrinsic viscosity | 1,936 | pending S27 |  |
| 6 | Density | 1,699 | pending S27 |  |
| 7 | Specific volume | 1,699 | pending S27 |  |
| 8 | PCE_OPV_aug | 1,203 | 2 |  |
| 9 | Elongation at break | 1,113 | 2 |  |
| 10 | Tensile stress strength at break | 1,034 | 2 | reserve |
| 11 | Tensile modulus | 1,034 | 2 | A#3 |
| 12 | Electric conductivity | 982 | 3 |  |
| 13 | Lc phase transition temperature | 941 | pending S27 |  |
| 14 | Volume resistivity | 934 | pending S27 |  |
| 15 | Softening temperature | 763 | pending S27 |  |
| 16 | Gas permeability coefficient | 711 | 2 | A#4 |
| 17 | Water absorption | 680 | pending S27 |  |
| 18 | Heat of fusion | 588 | pending S27 |  |
| 19 | bandgap bulk_Egb | 561 | 1 |  |
| 20 | Crystallization temperature | 456 | pending S27 |  |
| 21 | Gas diffusion coefficient | 433 | 2 |  |
| 22 | Dynamic mechanical properties storage modulus | 387 | 3 | reserve |
| 23 | Dielectric Constant_EPS | 382 | 1 |  |
| 24 | Refractive Index_Nc | 382 | pending S27 |  |
| 25 | Ionization Energy_Ei | 370 | 1 |  |
| 26 | Electron Affinity_Eea | 368 | 1 |  |
| 27 | Surface tension | 346 | pending S27 |  |
| 28 | Solubility parameter | 316 | pending S27 |  |
| 29 | Cohesive energy density | 307 | pending S27 |  |
| 30 | Dynamic mechanical properties loss tangent | 300 | 3 |  |
| 31 | Dielectric loss factor | 279 | 3 | A#5 |
| 32 | Contact angle | 255 | pending S27 |  |
| 33 | Tensile stress strength at yield | 253 | 2 |  |
| 34 | Gas solubility coefficient | 249 | 2 |  |
| 35 | Dielectric loss tangent | 238 | 3 | reserve |
| 36 | Isothermal weight loss time | 228 | 3 | A#6 |
| 37 | Heat of fusion mol conversion | 213 | pending S27 |  |
| 38 | Specific heat capacity | 207 | pending S27 |  |
| 39 | Dynamic mechanical properties loss modulus | 193 | 3 |  |

Totals: **39 tasks, 42,169 records.** Named tiers: 8 in Tier 1, 8 in Tier 2, 7 in
Tier 3; **16 tasks remain `pending S27`** 鈥?enough headroom that no design decision
depends on the missing table.

## The honest note for a reviewer

Tier *sizes* (14/15/10) are verified from the main text. The per-task assignment is
not: it lives in Table S27 of a Supporting Information file that ships as a single
156 MB PDF behind a bot wall. Two ways to close the gap on D0, in order of cost:

1. download the SI once and transcribe 39 rows (~30 min of human work);
2. if the SI is unobtainable, recompute the tiers from the four published criteria
   and submit that as a derived artefact with its provenance line.

Never silently adopt a different tiering. Either way the main table does not change,
because every task in it is already named in the main text.
