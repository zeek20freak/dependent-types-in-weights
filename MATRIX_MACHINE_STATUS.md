# DT weights kernel v0.1 — status (2026-11-25)

## Done: the plain-logic DT reference machine (RefMachine)

`weights_kernel_dt.py` implements the sealed DT checker (Pi + universes +
conversion + Nat + eliminator) as a plain-logic reference machine, the
lockstep partner the matrix machine must agree with step for step.

- **41/41 passed** against `kernel_dt.py` on the re-declared suites
  (27 check + 6 infer + 8 defeq), with **exact verdicts AND exact
  failure paths/kinds** for every rejection — including both
  motive-codomain regressions at `("motive",)` / `"not-a-type"`.
- Run: `python3 weights_kernel_dt.py`.
- The plain suites still pass on import (20/20 DT, 21/21 Nat).
- Sealed files untouched: `kernel_dt.py`, `test_kernel_dt.py`,
  `test_kernel_dt_nat.py`, `RULES_DT.md`, `WEIGHTS.md`.

### Design decisions beyond the spec (with reasons)

- **Monotonic register allocation.** Registers are allocated and never
  freed within a run, so every `(sel, ptr)` reference stays valid for
  the whole run. This trades register count for the absence of any
  freeing/aliasing bug. (Cost: 180 registers allocated at most;
  see the live-register measurement below.)
- **Var lookup shifts on read.** A `Var(n)` returns `ctx[n]` copied
  with shift `n+1`, because the stored type's free variables are
  relative to the definition context, not the current context. This is
  the de Bruijn form of the a2 capture class.
- **Failure paths from the frame stack.** `_error` computes the path
  from the `FRAME_SEG` of each frame on the control stack, plus the
  segment of the frame whose handler is currently running (it was
  popped before dispatch). This makes nested rejection paths faithful
  (e.g. `("body", "func")`, `("arg",)`, `("motive",)`).
- **Frame carries the whnf'd motive type.** The `REC_M_D` frame stores
  the whnf'd motive type in its 4th ref, because a later `COMPARE`
  overwrites `res` — recovering it from `res` was a real bug
  (it read an EMPTY slot).

### Measured bounds (from the sealed suites)

| quantity | max | notes |
|---|---|---|
| term depth (binary layout, Rec chain) | 10 | `App(App(add, 2), 3)`; 2047 slots |
| whnf output depth | 9 | |
| subst output depth | 9 | |
| de Bruijn index | 3 | plan index field over 0–7 |
| universe level | 1 | level field one-hot over 0–3 (3 is headroom) |
| context entries | 8 | dynamic (7 static) |
| control depth | 14 | frames |
| walk stack depth | 7 | |
| steps per case | 1050 | worst case (induction principle) |
| registers allocated (monotonic) | 180 | |
| registers **live at once** | **14** | worst case: "eta through check" |

Bounds measured by `measure_bounds.py` (term/depth/index/level/ctx)
and by the RefMachine harness (ctrl/reg/ctx/wstack/steps), with the
live-register count by `measure_live_regs.py`.

## Done: the MatrixMachine (explicit matrices) — 2026-11-26

`matrix_machine_dt.py` transliterates the RefMachine's ~100 handlers
into explicit matrix ops: one control matrix `W_CTRL` (156 features x
116 actions, threshold discipline — an action fires iff all its
required feature bits are 1), 116 straight-line actions (11 INFER,
3 WHNF, 1 COMPARE, 15 WALK, 86 RETURN), each a fixed sequence of
sparse-matrix ops over the bounded 16-register file, the frame stack
(14), the walk stack (7), and the context (8). Run:
`python3 matrix_machine_dt.py`.

- **Lockstep 41/41 passed**: RefMachine vs MatrixMachine agree step
  for step (full canonical snapshot after every step, including the
  whole 16-register file with stale trees in free registers) on all
  41 cases — 27 check + 6 infer + 8 defeq — with verdicts matching
  `kernel_dt.py` (DT 20/20, Nat 21/21, plain suites re-verified
  20/20 + 21/21), including both motive-codomain regressions at
  `("motive",)` / `"not-a-type"` and both capture cases (a2, family).
- Measured maxima in the lockstep run: ctrl=14, reg=16, ctx=8,
  wstack=7, steps=1050 — identical to the RefMachine's.
- **Weight count: 1,338,752 nonzero parameters** (registry discipline,
  same as STLC's 4,849; `count_params()` in the file, 26 entries).
  Dominant: register-file per-register gating 786,048 (the 16 x
  49,128-dim register file); ctrl shift-down/up 135,421 each;
  node-write 98,256; node-read slot extraction 49,128; wstack
  shift-down/up 37,326 each; ctx shift-down/up 14,462 each; frame
  write routing 10,417; walk-item write routing 6,221; eta
  error-catch 2,334; ctx slot write routing 2,066; ptr permutations
  2,046/1,023/1,023/511/255/255; liveness OR 1,664; fail-path 1,344;
  comparison bits 512; node-read sel mux 432; control matrix +
  bias 430; derived-bit threshold layers 337; free-stack routing 32.
  (The whole-tree substitution circuit's 730,003 weights / 65,504
  gates are a separate component measurement, NOT in this count —
  see the deviation below.)

### Design decisions beyond the spec (with reasons)

- **Walk is node-at-a-time, not whole-subtree.** The RefMachine's
  `_step_walk` pops one item and processes one node, and lockstep
  is step-for-step, so the MatrixMachine's WALK does the same.
  Consequence: the whole-tree `subst_circuit` (730,003 weights) is
  NOT the SUBST work-item — applying it whole-subtree in one step
  would finish a walk in 1 step while the RefMachine takes N,
  breaking lockstep. The SUBST/COPY work-items use the circuit's
  *per-node* machinery (VAR threshold comparisons, shift-down
  permutation, graft detection as an equality). The whole-tree
  circuit stands as a separate component measurement.
- **Only the selected action is evaluated per step.** STLC computes
  all 8 candidates and gates them; with 116 actions that is 100x
  the work for identical values (the action vector is one-hot, so
  the gated sum equals the selected candidate). The control matrix
  still selects the action; each action is a fixed sequence of
  matrix ops.
- **The free list is a stack, not a set.** The RefMachine's `_free`
  is LIFO (initially [15..0], top 0; a sweep pushes freed registers
  in ascending order, so the next alloc pops the highest freed).
  The MatrixMachine keeps the free stack faithfully (16 slots +
  depth) because the register index a ref points to is part of the
  state.
- **A sweep does NOT clear trees; an alloc DOES** (`regs[r] :=
  empty` on alloc, exactly as the RefMachine). The lockstep decode
  therefore compares the full 16-register file, including stale
  trees in free registers.
- **Counting discipline (STLC's):** a matrix construction is
  counted once by its nnz however often applied; gating
  (multiplying by a 0/1 selector) is free — the matrices being
  gated are counted. Bilinear selections (node read/write by
  data-dependent pointers) are evaluated in numpy via the gated-sum
  form and counted analytically, the same move `subst_circuit.py`
  makes for index arrays.

### Bugs the lockstep caught (both fixed in the file)

- **CMP_W_R's right operand is `res`, not the frame's r2.**
  In the RefMachine's `_ret_cmp_w_r`, `rw = (res_sel, res_ptr)` —
  the whnf'd right from the walk that just returned — while the
  frame's r2 is the pre-whnf right. All nine CMP_W_R actions that
  use the right operand (`lam_lam`, `eta_l`, `eta_r`, `var_var`,
  `type_type`, `pi_pi`, `app_app`, `succ_succ`, `rec_rec`) read
  `fr[2]`; they now read `self._res_ref()`. (First lockstep run:
  8/41; the eta case diverged at step 15 on the CMP_ETA frame.)
- **`W.N_SRC` is 10, but the live sel encoding needs 18.**
  `weights_kernel_dt`'s `SOURCES` list (INPUT, RES, WORK, AUX,
  R0..R5) is dead code — never used in that file. The live encoding
  is `_sel_of_reg(r) = 2+r` over the 16-register pool, i.e. sel
  0..17 (0=INPUT, 1=AUX, 2+r=register r). The MatrixMachine took
  `N_SRC = W.N_SRC` (=10), so every sel >= 10 one-hotted to an
  all-zero vector (argmax 0) — register 8+ read back as the input
  tree. Now `N_SRC = 2 + POOL_SIZE` (=18), `RES_SEL_DIM = 19`.
  (Second lockstep run: still 8/41, every failure a sel-10+
  truncation; fixed, then 41/41.)
- Minor, fixed before the first run: `_error` collected frame
  segments bottom-to-top then wrongly reversed them (the
  `reversed()` call was removed); two CMP_W_R actions referenced a
  never-assigned `_last_popped` (now use the popped frame directly).

### Performance note

The lockstep compares a full canonical snapshot after every step;
`decode()` originally looped over all 16x2047 register slots in
Python (3 argmax calls per slot, ~0.2s per decode). It is now
vectorized (per-register argmax over the slot axis, 0.022s per
decode). Full 41-case lockstep runs in ~170s. This is a test-only
speedup, not a semantic change.

## Path forward

- The MatrixMachine is built and sealed by lockstep (41/41).
- Then the Still measures the floor on the new registry.

## Parent assessment (2026-11-25 ~16:50 PT, verified by Quine's own run)

`python3 weights_kernel_dt.py` re-run by Quine: **RefMachine 41/41
passed**, exact verdicts + failure paths, both motive-codomain
regressions at `("motive",)` / `"not-a-type"`. The RefMachine half
of the bar is done and verified.

The MatrixMachine blocker as stated above is overstated on point 1.
The naive construction (a lookup table over all (source, `T`)
pairs) explodes, but the structured construction does not:
for each output slot, the content is a fixed function of the
source's path slots (nearest target-variable ancestor = a priority
encoder over a fixed path; the binder count on the path = a
threshold over fixed slots) and the replacement tree's slots
(mapped from that ancestor, shifted by the binder count — a fixed
permutation per shift value). That is O(N·depth²) gates,
polynomial in the layout — large at the measured layout
(term depth 10, 2047 slots; roughly ~200k gates, an estimate to
be measured, not taken on trust), not an explosion. It is the
same move as the STLC equality circuit (XNOR per entry, AND over
entries) versus a lookup table. Point 2 stands: the bounded
register file (~16 registers, free list; 14 live at once says it
fits) is real work, and the transliteration is a big job. The
blocker is a workload, not a refutation of the bet.

Path forward (revised): (a) bounded-register RefMachine redesign,
re-verify 41/41; (b) the substitution circuit as a fixed
construction, tested against the plain kernel's `subst` on the
suite + fuzz, count measured; (c) transliterate + lockstep.

## Cruxes solved (2026-11-25)

### Crux 1 — Bounded-register RefMachine: DONE, 41/41

The RefMachine now allocates from a **bounded pool of 16 registers**
with a free list and a mark-and-sweep (`POOL_SIZE = 16` in
`weights_kernel_dt.py`). `_alloc` pops a free register; when the
free list empties it sweeps, freeing every register not in
`_live_regs()` (plus the registers allocated in the current step,
which may not be stored in state yet). A freed register's tree is
reset to empty on re-allocation.

Two corrections to the earlier numbers: the complete liveness scan
(`_live_regs`, in the file) covers frames (r1..r4, saved dest, walk
dest, smalls refs), ctx, dest, res, focus (fsel/asel), and walk
items — the old `measure_live_regs.py` missed fsel/asel and the
smalls refs (the motive ref rides in frame smalls on REC_*/WHNF_REC
frames), so its 14 was an undercount. **True max live is 15**
(worst case "eta through check", step 64), so the 16-register pool
fits with exactly one spare. Measured by `measure_live_regs2.py`.

Verification (all by the harness + a debug invariant run):
- **RefMachine 41/41 passed**, exact verdicts AND exact failure
  paths/kinds (the harness checks both) — including both
  motive-codomain regressions at `("motive",)` / `"not-a-type"`.
- **Invariant holds over 6,893 steps** (`check_pool_invariant.py`):
  no freed register is ever referenced in state — the liveness
  function is complete.
- **105 sweeps fired** across the 41 cases: the pool genuinely
  reuses registers, it is not just large enough to never sweep.
- New maxima: registers allocated simultaneously **16** (was 180
  monotonic), live at once **15**, control depth 14, ctx 8,
  walk stack 7, steps 1,050. Register file as a flat state vector:
  **16 x 49,128 = 786,048 dims** (was 8,843,040).

### Crux 2 — Substitution as a fixed circuit: DONE, exact agreement

`subst_circuit.py` builds de Bruijn substitution as an explicit
fixed construction (matrices + threshold gates, the STLC equality
circuit's discipline). Per output slot q: the binder count on the
path root -> q is a fixed sparse matrix (M_B); the graft flag is an
integer equality (two thresholds + AND); the no-graft candidate is
S with a VAR adjustment (one threshold + a shift-down permutation);
the grafted content routes T through a fixed index array REL (one
entry per (q, ancestor) pair) with the selection by graft(a) as the
bilinear gating primitive (STLC's action gating — the matrices
being gated are counted, the gating itself is not). A VAR in T
shifts iff its index >= its binder depth within T (the cutoff the
plain COPY carries); slots below a graft point but beyond T's own
depth are EMPTY (a present-bit gates the grafted idx one-hot —
that was a real bug found by the test, fixed before the final run).

Structural facts that keep it polynomial: a VAR node is a leaf, so
at most ONE graft ancestor lies on any root -> q path (no priority
encoder needed — the selection is a unique-select); grafted regions
below distinct VAR leaves are disjoint. The at-most-one fact is
asserted empirically: worst grafts/path = 1 over the whole fuzz.

Test results (`test_subst_circuit.py`, all exact):
- (A) `subst_db` (the independent recursive de Bruijn spec) vs
  `kernel_dt.subst` on suite terms: **294 cases, 0 failures**.
- (B) circuit vs `subst_db`, de Bruijn fuzz, general (k, s):
  **321 cases, 0 failures** (79 skipped: expected result outside
  the layout — depth > 10 or index > 7 — out of the circuit's
  contract, as the machine's slot arrays cannot represent them
  either). 138 of the 321 exercise at least one graft.
- (C) circuit vs `kernel_dt.subst`, named suite terms + named
  fuzz: **381 cases, 0 failures** (3 more skipped out of layout).
- Output one-hot validity asserted on every circuit run (each
  field row sums to exactly 1, or the slot is EMPTY).

**Measured count: 730,003 nonzero parameters (weights) and 65,504
threshold gates** (counted programmatically from the built
matrices and the threshold layers' fan-ins; breakdown in
`test_subst_circuit.py` output / the file's Counter). The
~200k-gates estimate is corrected, not confirmed: the weight
count is ~3.6x it, the threshold-gate count ~3x below it. The
dominant cost is the REL routing (368,658 weights, ~50%): 18
scalars routed per (slot, ancestor) pair over 20,481 pairs — the
estimate counted a slot-route as one gate and did not separately
account for the 18-scalar width or the idx one-hot encode
(143,290). The structural claim holds (polynomial in the layout,
O(N x depth) routing + O(N) threshold layers, not an explosion);
the constant was underestimated. The count is a component
measurement for the substitution circuit alone, not the machine's.

Files: `subst_circuit.py` (circuit + `subst_db` + count registry),
`test_subst_circuit.py` (layers A/B/C + validity + graft-path
assertions), `measure_live_regs2.py` (complete liveness),
`check_pool_invariant.py` (debug invariant + sweep count).
Sealed files untouched: `kernel_dt.py`, `test_kernel_dt.py`,
`test_kernel_dt_nat.py`, `RULES_DT.md`, `WEIGHTS.md`.

## Independent check (sealing pass, part 1) — DONE 2026-11-26, PASS with a qualification

Independent checker (agent ce9a2647, completed ~18:16 PT). Harness `/tmp/check_mm_independent.py` (log `/tmp/check_mm_independent.log`), fresh snapshot/decode code written from the two machines' state layouts — NOT reusing `run_lockstep()` or `MatrixMachine.decode()`. Two measures the builder's decode lacks: every one-hot field strictly validated (sum must be 1, exactly one entry 1 — no silent argmax of an all-zero vector, the sel≥10 failure mode), and every node-row field validated on its own terms.

- **Independent lockstep: 41/41 passed** step-for-step (27 check + 6 infer + 8 defeq), full canonical snapshot after every step including the whole 16-register file with stale trees. **Total steps compared: 6,893** — matches the invariant run's 6,893 exactly, a cross-check on the step count. Maxima identical (ctrl=14, reg=16, ctx=8, wstack=7, steps=1050). Verdicts checked against `want` in build_cases (the kernel_dt.py expectations), not just against the RefMachine — including both motive-codomain regressions and both capture cases. Builder-decode diagnostic: 0 mismatches.
- **Weight count reproduces: 1,338,752, 26 entries, sum==total.** 24 of 26 entries recompute exactly from dimensions/formulas (register-file gating 786,048 = 16×2047×24; all six shift matrices; node-write 98,256; node-read 49,128 + sel mux 432; ptr permutations; frame/walk/ctx routing; free-stack 32; liveness 1,664; fail-path 1,344; eta 2,334; W_CTRL + bias 430 = sum(len(req)) + 116).
- **Hand-derived actions: MATCH (10 across all 5 classes)** — INFER (inf_var_ok, inf_lam), WHNF (whnf_app, whnf_rec), COMPARE (cmp_start), WALK (walk_copy_var, walk_subst_var_graft), RETURN (ret_lam_ty, ret_cmp_w_r_lam_lam, ret_entry_valid) plus extras. The fixed CMP_W_R bug is confirmed fixed in all nine actions (every one reads `rw = self._res_ref()`, never `fr[2]`).
- **Sealed files untouched: PASS** — suites 20/20 + 21/21 re-run; SHA-256 hashes unchanged (kernel_dt.py 2cf4156b…, test_kernel_dt.py a21afd3e…, test_kernel_dt_nat.py d4096872…, RULES_DT.md 3404e76d…, WEIGHTS.md dc29dda6…); independent-kernel comparison re-verified 67/67 verdict + 67/67 failure-path; both seal receipts intact (re-run preserved `seal_verdict`).
- Checker's own error, caught and corrected: its first harness wrongly required register-row sums in {0, 3} and flagged 32/41 cases; it investigated instead of reporting a machine failure — the machine's node convention is per-field (kind always written, level/idx only when the kind carries them), so an all-zero field decodes to 0. Harness corrected, re-ran clean. The check caught its assumption, not a machine bug.

**Qualification (verified by Quine's own reading of the registry): the count is soft by ~400–500 weights (~0.04%).** Two entries are analytic, not built circuits: `"comparison bits (idx/depth/level)" 512` is a bare literal (line ~372) whose comment names the same three comparisons the 337 entry already itemizes — likely over by ~300–500 (double-count or a separate allowance for action-internal comparisons, estimated actual ~120–200). `"control derived-bit threshold layers" 337` is itemized and arithmetically correct, but its w_gt component is written `"~130"` (explicitly approximate) — a tight implementation is ~30 sparse or ~89 dense, so likely over by ~40–100. The fail-path 98 ("~98") and eta 2,334 carry the same "~" softness at smaller scale. **Call: disclose, don't tighten now.** The headline 1,338,752 is sound to within ~500. The Still floor will be measured against this registry with the same ±500 caveat — negligible against 1.3M and against any floor the Still finds. Tightening (building the actual comparison/derived-bit circuitry to count exactly) is real work for 0.04%; revisit only if a floor ever lands close enough for 500 to matter.

## Path forward (updated)

- Independent check: PASS (with the ~500-weight softness noted above).
- Then the Still measures the floor on the new registry (main-session work — the Still's rounds are played by the parent, not delegated).
- Then fold into the held package.

## Still floor — MEASURED 2026-11-25 ~18:48 PT

Still subagent (17d28b73) measured the floor on the DT registry. Files: still_checker_dt.py, still-checker-dt-result.json, still-checker-dt-map.json, STILL-CHECKER-DT.md (all new; no sealed file modified). Bar re-verified first: baseline lockstep 41/41 in 91s, verdicts 20/20 + 21/21.

| | weights | share of baseline (1,338,752) |
|---|---|---|
| Coverage floor, pruning only | 49,204 | 3.7% |
| **Floor, pruning only (eta folded in)** | **49,196** | **3.68%** |
| **Floor, pruning + sharing** | **47,687** | **3.56%** |

The machine is **96.3% never-read** — far sparser than STLC (whose sealed floor was 16.6% of its baseline). Phase 1 coverage: one instrumented run of all 41 cases (6,893 steps). Phase 2 whole-entry prunes (all 26): 24 break at the screen, 2 hold. Phase 3: zeroing all never-read stored weights as one batch **holds at lockstep**; three transpose pairs survive (ctx, ctrl, wstack shift down/up), sharing saves **1,509**.

- **Eta error-catch (2,334) — HOLD, genuine.** `_try_catch_error` never returns True in the bar; its 8 read encoder weights are removable, floor 49,204 → 49,196.
- **Register-file gating (786,048) — HOLD, but vacuous.** The prune no-ops `write_tree_reg`, which is never called (dead code, definition only). It does not test the `__init__` ctx writes, which bypass it via direct assignment. The gating's floor share (43,039) stands in the primary floor.

**The gating layers (the 43,039 question — a fork, Zack 2026-11-25; resolved as layers, 2026-11-26).** The gating is 58.7% of the baseline; 743,009 of its 786,048 weights are never read. Its 43,039 "read" weights come entirely from the `__init__` ctx writes (each ctx type a dense tree, ~6,141 gating positions per register, 7 registers written). Two mechanisms, not two readings of one number: Layer 1 — as shipped (registry model) — the gating covers dest writes, `__init__` writes are dest writes → 43,039 counts; floor **47,687** shared / **49,196** pruning-only. Layer 0 — machine proper (scaffolding) — if `__init__` writes are setup, not machine ops, the gating contributes 0 → floor **4,648** shared / **6,157** pruning-only (0.35% / 0.46%), the machine proper with setup excluded. The difference is exactly the 43,039. The boundary is the `__init__` writes (direct assignment, bypassing the write path); both layers are always reported together, and the next rung reports both layers the same way.

Other coverage notes: shift matrices almost entirely unread (ctrl shifts: 1,026 of 135,421 each); node-write erase set only 43 positions; only 96 tree slots ever read; only 4 of 16 level-max pairs ever occur; 100 of 116 actions fire (16 never-fired are error variants the bar's cases do not reach); W_CTRL 429/430 read. Caveats carried: ±500 softness inherited; comparison bits and derived-bit layers appear to overlap (baseline may be overstated by up to ~512); analytic entries are cost models of inline code, not stored matrices; the floor is against the 41-case bar, not a universal minimum.

**Note (2026-11-25, fold): the independent-check harness script (`/tmp/check_mm_independent.py`) was lost to /tmp cleanup before the fold — its numbers and hashes are recorded here, not re-runnable from the package. The re-runnable seal is the lockstep + suites + Still receipts in the package.**
