# Speech script — How Much Is a Chain Worth? (Group 32, English)

**Deck:** `ChainWorth_Group32.pptx` · 18 presented pages + 3 backup pages · **Length:** 1,351 spoken words ≈ **10 min 00 s** at 135 wpm (≈ 9 min 00 s at 150). **The eight-minute version is this same script minus the eleven passages listed at the end — 2 min 04 s of cuts, landing just under 8 min.** Nothing needs rewriting on the day; only whole sentences are dropped, and they are marked there.
**How to read this:** bold = stress and slow down · « » = stage direction, not spoken · each block names the slide **and the part of the slide** it talks about.

Every block has three parts:
① **Script** — what you actually say out loud. **All timing is measured on this part only.**
② **Plain version** — the same page in everyday words. **Not read aloud** — for Q&A, for rehearsing your own understanding, and for anyone who reads the deck without you.
③ **Likely question → one-line answer** — the ten or so questions we are most afraid of, parked on the page that can answer them. Flip, read, stop talking.

**Hardware (applies to every GPU number):** one **NVIDIA GeForce RTX 4070**, 12 GB (12,282 MiB), 46 SMs, compute capability 8.9, driver 591.86, CUDA 12.1, torch 2.3.1+cu121. Total measured GPU time **5.34 GPU-h** (A3 2.66 + A4 fine-tune 2.68) against a 19–36 GPU-h plan. The A2 rows in the shared cost table are CPU hours and are not in that sum.

| Block | Slides | Plan |
|:--|:--|--:|
| Frame | 1–3 | 1 min 21 s |
| Question and design | 4–7 | 1 min 58 s |
| Results | 8–13 | 3 min 59 s |
| Close | 14–18 | 2 min 42 s |
| **Total** | **1–18** | **10 min 00 s** |

---

## Slide 1 · Title — badge, title, the amber line, four statistics

**On slide:** top-right `GROUP 32` → title → the amber "short version" band → the four numbers → member line.
**≈ 21 s** — spoken part only

**① Script — read this aloud**
> Good morning. Group 32 — Yihui, Xingshuai, Peng, Yixuan. Our question is the title: **how much is a chain worth?** Here is the answer up front. **In twelve of nineteen cells the chain does no better than the unit arm, and tenfold data has not fixed that.**

**② Plain version**
The cover. It names us, the question, and — new this version — **the answer**, before the evidence. Putting the conclusion first is deliberate: an audience that knows where you are going can spend the talk checking your reasoning instead of guessing it.

**③ Likely question → one-line answer**
- *"Nineteen cells of what?"* → One task × one chain-aware arm; two arms, eleven and eight cells.

---

## Slide 2 · Motivation — left column (crystal) → right column (polymer)

**On slide:** three-level stack on the left; four-level stack on the right, orange box = the chain.
**≈ 32 s** — spoken part only

**① Script — read this aloud**
> A crystal gives composition and a unit cell; the structure settles. A polymer adds a third level: the chain. Recipe and dish — for a crystal the recipe is the dish; for a polymer, two cooks follow the same recipe and get different dinners. Our benchmarks keep the recipe, throw away the dish, and ask a model how it tastes. The chain isn't badly modelled. It is **deleted**, and PolyBench says so itself.

**② Plain version**
The orange box is what gets thrown away: molar-mass distribution, sequence, tacticity, morphology, processing history. The bottom band paraphrases a limitation the benchmark states about itself, so the gap is acknowledged upstream — we did not invent it.

**③ Likely question → one-line answer**
- *"If the chain is deleted, why call any arm 'chain-aware'?"* → Because a richer encoding of the repeat unit is not the same as chain information — backup page B1.

---

## Slide 3 · Quantities — Λ card, then n card

**On slide:** left card (Λ) → right card (κ) → the two lower cards (n, n\*).
**≈ 28 s** — spoken part only

**① Script — read this aloud**
> Two symbols. **Λ** is credit for the gap: what share of what the unit arm left on the table does the chain arm pick up. We **don't clip** it — negative is a finding. **n** is training rows, not total rows, and that bites in a moment. κ and n\* are its slope and its crossing; both have guards, neither is a headline.

**② Plain version**
Λ is a credit score for the remaining gap: grade the chain arm only on what the unit-only model left unexplained. A denominator below 0.05 is flagged unstable, because there the ratio amplifies noise rather than measuring recovery.

---

## Slide 4 · The question — quote box, three numbered reasons

**On slide:** large quote on top; the three columns below.
**≈ 24 s** — spoken part only

**① Script — read this aloud**
> The project in one sentence: compress a chain into a repeat unit — **how much variance becomes unexplainable, and does that share shrink when data grows tenfold?** Both answers are useful. If it shrinks we are short of data. If it doesn't, the representation is **missing a coordinate**, and data never puts a coordinate back.

**② Plain version**
The second case is what "worth" means in the title: a gap that survives a tenfold increase behaves like a missing coordinate, so more data is the wrong remedy.

---

## Slide 5 · Benchmarks — table, tier table, ρ warning

**On slide:** three-row table on top; bottom-left the tiers; bottom-right the ρ warning.
**≈ 32 s** — spoken part only

**① Script — read this aloud**
> Three benchmarks, three jobs. PolyBench carries H1, six tasks on a fixed split. PB26 carries H2, five official folds plus transfer. OpenPoly carries H3 and H4, three properties where **we** set the sample sizes. One warning, on this slide rather than in the discussion: **here tier and sample size are nearly the same variable — ρ = −0.96.** Read "needs chain information" and you also read "has thirty times less data".

**② Plain version**
The warning is the catch of the whole design: the tier label proxies two things at once, so any tier-based claim has to be checked against a matched-n experiment — which is exactly slide 10.

**③ Likely question → one-line answer**
- *"Then why use tiers at all?"* → Because the benchmark publishes them; we report the confound with the tiers, not after the results.

---

## Slide 6 · Hypotheses — four-row table, two notes

**On slide:** last column = refutation criterion; bottom-right = the H2′ note.
**≈ 29 s** — spoken part only

**① Script — read this aloud**
> Four hypotheses. The last column is the one that matters — **the number that would kill each.** H1 and H2 are the proposal's words; H3 and H4 say plainly what they are — untested, and descriptive. No criterion moved afterwards. The pass mark agreed before anyone sees the exam paper. The corner note: **H2′ isn't pre-registered**; it sits beside H2 and cannot change its verdict.

**② Plain version**
A contract with ourselves. H2′ exists because the absolute ruler turned out to disagree with the ratio; reporting it beside H2 rather than inside it is the honest way to add a metric after the fact.

---

## Slide 7 · Method — pipeline left, arm table right

**On slide:** five-step pipeline; five-row arm table; footer = provenance gate.
**Hardware (not spoken):** single **RTX 4070**, 12 GB (12,282 MiB), 46 SMs, cc 8.9, driver 591.86, CUDA 12.1, torch 2.3.1+cu121. A3 = chemprop 2.1.0 official MPNN — `d_h 300`, depth 3, dropout 0, batch 64, max lr 1e-3, ≤100 epochs, patience 15; A4 = frozen polyBERT (600-d) + ridge head, plus a fine-tuned control. 538 A3 + 48 A4 records, all `device=cuda`; VRAM ≈ 1.4 GiB.
**≈ 33 s** — spoken part only

**① Script — read this aloud**
> Every number goes down one pipe: licensed data, frozen hashed splits, one arm at a time on the same rows and seeds. **A2** descriptors into XGBoost, **A3** a chemprop net, **A4** a frozen polyBERT — those two on one **RTX 4070, twelve gigabytes, five and a third GPU-hours**. **A1** predicts the mean. **A5 has zero runs** because its inputs don't exist. Behind it, 962 records and **4,897 provenance checks — zero failures, 65 exempt**.

**② Plain version**
One production line, five machines, same rows and seeds — otherwise a difference could be data, or luck, or tuning. A5 sits on the board with zero runs so nobody can read it as "we tried and it didn't work".

**③ Likely question → one-line answer**
- *"A3 was never tuned — isn't that unfair?"* → Yes, and it is the first next step: 30 trials for A3, about 2–3 GPU-hours.

---

## Slide 8 · Result 1 — left KPI column, right cell table

**On slide:** the three numbers on the left, then the table; the amber line under the table is the caveat.
**≈ 36 s** — spoken part only

**① Script — read this aloud**
> First result, and it's uncomfortable. **Twelve of nineteen cells have a negative Λ** — same test rows, same splits, the chain-aware arm does worse. Median cell minus 0.01, median reading minus 0.098; two units, so both. The bottom row is the worst cell — there the chain arm does worse than predicting the mean: 0.59 without it, minus 0.01 with it. The largest positive, plus 0.64, rests on six rows of fifteen. We print it; we don't lean on it.

**② Plain version**
A cell is one task × one arm; its value is the mean of that cell's usable runs. A reading is one run. Mixing the two units is how a "10% shortfall" gets written into a summary when the typical cell is near zero.

**③ Likely question → one-line answer**
- *"Is 12 of 19 just your sign convention?"* → No — Λ is unclipped, so a negative value means the chain arm scored below the unit arm on identical test rows.

---

## Slide 9 · H1 — leave-one-task-out chart, then right column

**On slide:** each bar = ρ after deleting one task; dashed lines = ±0.5; right column = the two ρ values.
**≈ 35 s** — spoken part only

**① Script — read this aloud**
> Does Λ order tasks the way the tiers do? It doesn't. **ρ is essentially zero with A3, 0.24 with A4**, and each arm has two cross-layer inversions. Either one refutes. But only one deletion pushes ρ past half, and it's dielectric loss — interval twice the effect, R² of minus 0.009, worse than the mean. So H1 is **refuted as pre-registered, by a point we cannot decide**. Removing it gives plus 0.53; we report that and don't remove it.

**② Plain version**
Like a class average dragged down by one student who never sat the exam: remove them and the average moves, but it is no longer the average of the class. Choosing the analysis after seeing the result is the one thing pre-registration exists to prevent.

**③ Likely question → one-line answer**
- *"So is H1 refuted or not?"* → Refuted by the rule we wrote down; the honest caveat is that we had no power to test it cleanly, with six tasks.

---

## Slide 10 · H1 re-tested — intro line, table, two readings

**On slide:** top line = the matching; table = the answer; bottom-left and bottom-right = two readings.
**≈ 35 s** — spoken part only

**① Script — read this aloud**
> The objection you're all thinking: tier one has more data. So we matched it — tier one downsampled to tier two's **training counts, 497 and 723**, training rows not totals, test rows untouched. **The gap does not go away**: tier one stays below tier two by **−0.033 and −0.087**, both intervals clear of zero. The confound takes −0.086 to −0.033: part of the size, none of the sign. The break is inside tier three — **+0.068 against −1.479**.

**② Plain version**
Two students with different study hours: we equalised the hours, and one is still ahead. The confound explains magnitude, not direction.

**③ Likely question → one-line answer**
- *"Why 497 and 723, not the row totals?"* → Because the tier label is about training rows; using totals would have aligned the test sets and mismatched the training sets.

---

## Slide 11 · H2 — Λ-versus-rows chart, two KPIs, sensitivity strip

**On slide:** line chart on the left; then κ and the 0.650 ratio; four grey boxes = sensitivity checks.
**≈ 45 s** — spoken part only

**① Script — read this aloud**
> H2 needs two things at once. **κ is −0.086, interval entirely below zero** — least squares through five points, so the slope is real. But Λ at the largest n is still **0.650 of** the smallest; halving means 0.5. So H2 is **not refuted**, and not refuted is not support. Look at the curve: it's a **hump**. Up from 0.295 to 0.503, then down to 0.192. The climb to the peak runs at **+0.398 a decade**; the fall to 3,000 at **−0.284**; the long tail at **−0.202**. The negative sign lives in the tail. Start at the peak and you refute yourself.

**② Plain version**
The four boxes are the sensitivity checks: drop the smallest level and the sign holds; keep only the two stable levels, 300 and 1,000 rows, and it flips. We report all four, including the one that would have helped us.

κ itself is one straight line fitted by least squares to Λ against log10 of the training rows — one line per fold over the same five levels, then averaged. Not a piecewise average of the segments, and not a chord between the ends: the four segment numbers on the chart are read off the curve, they are not what κ is.

**③ Likely question → one-line answer**
- *"Isn't κ just an artefact of a bad denominator?"* → Partly — which is why the sign-flip after the small levels are dropped is on the page, not in a footnote.

---

## Slide 12 · H2′ — definition band, two curves, mechanism note

**On slide:** amber definition on top; two curves normalised to the smallest n; mechanism paragraph right.
**≈ 40 s** — spoken part only

**① Script — read this aloud**
> So we changed the ruler. H2′ asks the same thing absolutely: does tenfold data halve **ΔR²**? Same folds. The ratio ends at **0.650**, the absolute gain at **0.167** — halved, and the intervals **don't overlap**. Why both? Arithmetic. At 1,000 rows the denominator has already halved while ΔR² barely moved — so the ratio rises. End to end ΔR² falls six-fold and the denominator four, so Λ only falls to 0.65. Divide two collapsing numbers and the quotient can stop collapsing. **Two rulers, two questions.** Report both, merge neither.

**② Plain version**
A discount that shrinks in money can look flat as a percentage if the price itself is falling. That is the whole mechanism, and it is why the hump on the previous slide is arithmetic rather than physics.

**③ Likely question → one-line answer**
- *"So which ruler is right?"* → Both answer different questions; that is why H2′ is reported beside H2 and never merged into its verdict.

---

## Slide 13 · H3 and H4 — left card, then right card

**On slide:** left = H3, the three-input table and the floor; right = H4, three values of n\*.
**≈ 48 s** — spoken part only

**① Script — read this aloud**
> H3 was the clean experiment: same architecture, only the input changes. We never ran it — **the input can't be built**. No repeat-count column among 18,152 structures; a weight block identical in all 18,036 strings; one row per repeat unit, so no sequence. Its status is **not constructible** — there is no labelled chain-level target to fit. **So H3 has no verdict, and we do not report one.** In its place, a floor: median 0.0046, cleared by five of eight tasks. **H4** is description: below about **56 and 59** rows the graph arm leads, above it the classical one does. For Tm the sign never flips, so no n\*.

**② Plain version**
The three-row table is the whole argument: input, whether it exists, verdict. "Zero gain" is explicitly forbidden in the ledger for A5, because a zero we cannot measure is not a zero we measured.

**③ Likely question → one-line answer**
- *"Couldn't you synthesise the missing inputs?"* → A row can be assembled into a pentamer with RDKit, but that pentamer has no label here — we would be inventing the target.

---

## Slide 14 · Conclusions — four claims, then the closing band

**On slide:** claims 01–04; the sentence in the band at the bottom is the one to land.
**≈ 52 s** — spoken part only

**① Script — read this aloud**
> Four claims, each with its strength attached. **Chain-aware arms mostly do not pay** — twelve of nineteen. **Stratification is refuted**, not cleanly tested. **On the ratio ruler more data has not halved it, on the absolute ruler it has** — to 0.0066, which is the same order as the floor. **And information versus architecture stays unanswerable**, because A5 cannot exist here. So the line to take away — largest on the slide for a reason: **a richer encoding of the repeat unit does not recover it, and the chain itself stayed untestable here.** Which means the honest title of this talk is not how much a chain is worth; it is **how much a repeat unit is worth**.

**② Plain version**
Note the verb: *has not* bought it back, not *does not*. We never upgrade "could not test" into "no". The qualifier under the big line lists exactly what bounds it.

---

## Slide 15 · Falsifiability — what would change our mind

**On slide:** left = what would overturn each claim; right = what would not.
**≈ 39 s** — spoken part only

**① Script — read this aloud**
> One more page, because a project that lives on pre-registration should show its tripwires. **Build A5 and have it pay, and claim one dies. Match tier one until it's no longer lower, and claim two dies.** On the right, what would *not* move us: a bigger sample of the same kind, or quietly deleting the deepest cell. A third architecture we simply cannot exclude — both arms are negative overall, yet they disagree in sign in half the shared cells, which is exactly why we never average them.

**② Plain version**
These criteria went into the proposal before the numbers existed — that is the point. Every row on the left is a measurement someone else could run.

**③ Likely question → one-line answer**
- *"What result would make you abandon the whole framing?"* → A5 built and positive at matched tuning — that would make the gap architectural, not informational.

---

## Slide 16 · Limits and next steps — left list, right list

**On slide:** four limits on the left; four next steps on the right, already ordered.
**≈ 29 s** — spoken part only

**① Script — read this aloud**
> Limits, quickly. **A2 got thirty trials, A3 got none.** κ partly lives where the denominator is amplified. Tier three can't be matched upward. All six tasks share one label source. The fixes are ordered by benefit over cost: **two and three are analysis only; item one is a real experiment** — thirty trials for A3, two to three GPU-hours. Our cheapest test of the weakest assumption.

**② Plain version**
Correcting an earlier version of this script: the A3 tuning run is *not* free. It is 30 trials × folds × tasks, which is a few hundred fits. It is cheap in GPU-hours, not free in effort.

**③ Likely question → one-line answer**
- *"Why didn't you just do it?"* → Because changing A3's tuning budget after the freeze would have broken comparability — that is the real reason. It is cheap in GPU-hours (five and a third in total), not free in effort.

---

## Slide 17 · Contributions — four rows

**On slide:** one row per member; footer = the shared protocol.
**≈ 24 s** — spoken part only

**① Script — read this aloud**
> Four people, four workstreams, one protocol. Yihui framed the question and ran the GPU arms. Yixuan built the classical family. Xingshuai ran the scaling sweep and set up the transfer. Peng ran the transfer arm and wrote the audit. The numbers compare because **none of us used our own splits, features or seeds**.

**② Plain version**
Read the rows as who did what, not who did more. The footer is what makes the table mean anything.

---

## Slide 18 · References — the list and the repository

**On slide:** seven references, then the repository band.
**≈ 18 s** — spoken part only

**① Script — read this aloud**
> Every dataset, model and tool is here with a link. And the line at the bottom is the one to photograph: **github.com/XHH-Coco/How-Much-Is-a-Chain-Worth**. Every number in this deck is regenerated by a script in that repository. Thank you — questions welcome.

**② Plain version**
The repository is the answer to "where did that number come from". Appendix B of the report lists the tables those scripts write.

**③ Likely question → one-line answer**
- *"Can I reproduce this on my machine?"* → Yes if `env_core.json` matches; a different RDKit build changes the descriptor arm, and we flag that as host-sensitive.

---

## Backup pages (not presented)

| Page | What it answers |
|:--|:--|
| **B1** | What each arm actually sees — one source string, three encoders, no chain-level field |
| **B2** | Why Λ is a ratio and cannot be added across cells; the 0.135 → 0.034 mechanism |
| **B3** | 12 of 19 taken apart: 7 of 11 for A3, 5 of 8 for A4, and the aggregation rules |

---

## Appendix — GPU model, parameters, cost

| Item | Value | Recorded in |
|:--|:--|:--|
| GPU | NVIDIA GeForce RTX 4070 · 12 GB (12,282 MiB) · 46 SM · cc 8.9 | `results/env_fingerprints/env_gnn.json` |
| Driver / CUDA / torch | 591.86 · CUDA 12.1 · torch 2.3.1+cu121 | same file |
| A3 model | chemprop 2.1.0 official MPNN (nothing tuned by us) | `configs/arms/a3_graph.yaml` |
| A3 hyper-parameters | `d_h 300` · depth 3 · dropout 0 · batch 64 · max lr 1e-3 · ≤100 epochs · patience 15 · Noam 1e-4→1e-3→1e-4, warm-up 2 | `docs/交付_P2_B3_A3图神经网络主扫.md` §3 |
| A3 runs / cost | **538 records**, 2.661 GPU-h | `results/aggregate/gpu_hours.csv` |
| A4 runs / cost | 48 frozen (0.002 GPU-h) + 48 fine-tuned (**2.684 GPU-h**) | `results/aggregate/a4_finetune/gpu_hours.csv` |
| VRAM sampled | ≈1.37–1.41 GiB during A3 (whole-card sample, not a peak record) | `docs/交付_P2_B3_A3图神经网络主扫.md` §5 |
| Total GPU | **5.35 GPU-h** — inside the 19–36 GPU-h plan envelope (script says "about five and a third") | derived from the two tables above |
| Determinism | `cudnn.deterministic` **and** `use_deterministic_algorithms`; same seed twice = **bit-identical** (max Δ 0.0) | `results/aggregate/a3_repeatability.json` |

CPU-only arms (A1, A2) are not counted above; their ledger entry is 0.26 h on `device=cpu`.

---

### If you run long

For an eight-minute slot: 10 min 00 s minus **2 min 04 s** of cuts. Drop these eleven passages, in this order, and nothing else. Times are measured on the words, not guessed.

| # | Cut | Where | ≈ |
|:--|:--|:--|--:|
| 1 | the crystal aside ("A crystal gives composition…") | Slide 2, opening sentence | 5 s |
| 2 | the guard sentence on κ and n\* | Slide 3, last sentence | 7 s |
| 3 | the second half (why the question is worth answering) | Slide 4, from "Both answers" | 12 s |
| 4 | the three benchmark assignments read aloud | Slide 5, first three sentences | 13 s |
| 5 | "A1 predicts the mean" | Slide 7 | 2 s |
| 6 | the objection lead-in (why matched n) | Slide 10, opening sentence | 15 s |
| 7 | the sensitivity strip | Slide 11, four small cards | 18 s |
| 8 | the per-decade detail of the hump | Slide 12, after "it's a hump" | 17 s |
| 9 | "For Tm the sign never flips, so no n\*" | Slide 13 | 4 s |
| 10 | the ordering sentence (which fixes are cheap) | Slide 16, second half | 16 s |
| 11 | the four names | Slide 17, roster | 15 s |

**Keep Slide 15 whole** — it is the page that pre-empts the questions. It is deliberately absent from this list.

**Never cut** the ρ = −0.96 warning, the one-task refutation (Slide 9), the matched-n table (Slide 10), or 0.167-against-0.650 (Slide 12) — those four carry the argument.

The **plain versions** cost no time — they are not spoken — so when you trim, trim the script only.
