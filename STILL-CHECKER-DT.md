# STILL-CHECKER-DT.md — The Still on the DT checker

**Date:** 2026-11-25. **Net:** the DT MatrixMachine
(`matrix_machine_dt.py`, W_CTRL 156×116, 116 actions).
**Bar:** full lockstep 41/41 (RefMachine vs MatrixMachine,
step-for-step) AND verdicts matching `kernel_dt.py` (DT 20/20,
Nat 21/21). A prune holds only if both pass.
**Script:** `still_checker_dt.py` (new file; no sealed file touched).

## Method

Four phases, one script. Results in `still-checker-dt-result.json`
(floor, twins, batch confirm, per-entry results, coverage);
per-entry ledger in `still-checker-dt-map.json`.

- **Phase 0 — baseline lockstep.** Re-ran the bar unpatched:
  41/41 in 91s (maxima ctrl=14 reg=16 ctx=8 wstack=7 steps=1050).
- **Phase 1 — coverage.** One instrumented run of all 41 cases
  (MatrixMachine alone, 6,893 steps, verdicts 41/41). Method
  wrappers + CountingMat record, per registry entry, which
  weights are ever read. The never-read set gives the
  pruning-only floor. Two instrumentation fixes worth noting:
  action methods are dispatched through the module-level
  `ACTION_FNS` list (captured at import), so patching the class
  attribute does not intercept them — the level-max wrapper
  replaces the `ACTION_FNS` entry; and `__init__` materialises
  ctx types into registers by direct assignment, bypassing
  `write_tree_reg`, so the gating read set is recorded by
  wrapping `__init__` (see the gating note below).
- **Phase 2 — whole-entry prunes.** Disable each of the 26
  registry entries (zero a stored matrix, or no-op the code
  path), staged screen (4 heavy cases first, then all 41
  verdicts) then the full lockstep if the screen passes.
  24 break at the screen (0–2s each); 2 hold at lockstep.
- **Phase 3 — batch confirm + transpose twins.** Zero all
  never-read stored weights as one batch; screen, then the full
  lockstep. Then check which stored-matrix pairs survive as
  exact transposes after the batch prune; a surviving pair
  stores one copy.

## Result

| | weights | share of baseline |
|---|---|---|
| Baseline | 1,338,752 | 100% |
| Coverage floor, pruning only | 49,204 | 3.7% |
| Floor, pruning only (eta folded in) | **49,196** | **3.68%** |
| Floor, pruning + sharing | **47,687** | **3.56%** |

The coverage floor counts the weights ever read. The eta entry
prune genuinely holds (see below), so its 8 read weights are
removable and the pruning-only floor drops 49,204 → 49,196.
Sharing the three surviving transpose pairs saves 1,509.

## The two holds

- **eta error-catch construction (2,334) — HOLD at lockstep
  (97s).** Genuine: `_try_catch_error` is called on errors but
  never returns True in the bar — no catch ever fires (the defeq
  False cases return False directly via `ret_cmp_w_r_else`, they
  do not error into the catch). Forcing it to always return
  False changes nothing. The whole construction, including its
  8 read encoder weights, is dead for this bar.
- **register-file per-register gating (786,048) — HOLD at
  lockstep (98s), but VACUOUS.** The prune replaces
  `write_tree_reg` with a no-op, and `write_tree_reg` is never
  called — it is dead code (definition only,
  `matrix_machine_dt.py:496`, no call sites). The hold confirms
  that and nothing more. It does **not** test the `__init__`
  ctx writes, which bypass `write_tree_reg` via direct
  assignment (`self.regs[r] = ...`). The gating's floor share
  (43,039) comes entirely from those `__init__` writes, so it
  stands in the primary floor under the registry model (see
  the gating note).

## The gating note (the 43,039 question)

The gating is the machine's dominant entry (786,048 = 58.7% of
the baseline). Coverage says 43,039 of it is read and 743,009
never read. The 43,039 is entirely the `__init__` ctx-type
writes: each ctx type is a dense tree (2,047 slots × 3 one-hots
— `slot_onehot` sets kind, level, idx even for EMPTY nodes),
so one ctx register costs ~6,141 gating positions, and 7
registers are written. `write_tree_reg` itself contributes 0
(dead code). Two mechanisms — first named a fork, not two
readings of one number (Zack, 2026-11-25: "fork is the right
word"; the 4,648 floor is a separate mechanism):

- **Layer 0 — machine proper (scaffolding):** if the `__init__`
  writes are setup scaffolding (direct assignment, not a matrix
  op), the gating contributes 0 and the floor is **4,648**
  shared / **6,157** pruning-only (0.35% / 0.46%). Recorded in
  the result JSON as `floor_alt_gating_as_scaffolding`.
- **Layer 1 — as shipped (registry model):** the gating covers
  dest writes, and the `__init__` ctx writes are dest writes,
  so the 43,039 counts. Floor 47,687 shared / 49,196
  pruning-only.

The difference is exactly the 43,039. Resolved 2026-11-26
(Zack: keep as layers): the fork is a layered measurement, not
a choice. Layer 0 is the machine as an instrument, setup
excluded; Layer 1 is the machine plus its setup. The boundary
is the `__init__` writes (direct assignment, bypassing the
write path) — checkable, not a judgment call. Both layers are
always reported together; neither is quoted alone as "the
floor." The next rung reports both layers the same way.

## The twins

Three transpose pairs survive the batch prune (exact transposes
after zeroing never-read weights):

- `ctx shift-down` / `ctx shift-up` — saves 81
- `ctrl shift-down` / `ctrl shift-up` — saves 1,026
- `wstack shift-down` / `wstack shift-up` — saves 402

Sharing saves **1,509** total. The other predicted pairs
(ptr perms, node-read) are not transposes of each other.

## Per-entry coverage (read / total)

| entry | read | total |
|---|---|---|
| register-file per-register gating | 43,039 | 786,048 |
| ctrl shift-down | 1,026 | 135,421 |
| ctrl shift-up | 1,026 | 135,421 |
| node-write (write + erase) | 102 | 98,256 |
| node-read slot extraction (E_p) | 451 | 49,128 |
| wstack shift-down | 402 | 37,326 |
| wstack shift-up | 402 | 37,326 |
| ctx shift-down | 81 | 14,462 |
| ctx shift-up | 81 | 14,462 |
| frame write routing | 257 | 10,417 |
| walk-item write routing | 246 | 6,221 |
| eta error-catch construction | 8 | 2,334 |
| ctx slot write routing | 33 | 2,066 |
| ptr parent permutation | 9 | 2,046 |
| liveness OR (live bits) | 237 | 1,664 |
| fail-path construction | 112 | 1,344 |
| ptr left-child permutation | 55 | 1,023 |
| ptr right-child permutation | 55 | 1,023 |
| ptr rec-base permutation | 8 | 511 |
| comparison bits (idx/depth/level) | 512 | 512 |
| node-read sel mux | 301 | 432 |
| control matrix W_CTRL + bias | 429 | 430 |
| control derived-bit threshold layers | 284 | 337 |
| ptr rec-step permutation | 8 | 255 |
| ptr rec-target permutation | 8 | 255 |
| free-stack push/pop routing | 32 | 32 |

Why the shifts are so sparse: a frame is ~10–14 nonzero
positions out of 10,417, so a shift matrix copies almost
nothing — only 1,026 of 135,421 ctrl-shift weights are ever
read. The node-write erase set is only 43 positions: writes
land on freshly-allocated (near-empty) registers. Only 96
tree slots are ever read, and only 4 of 16 level-max pairs
((0,0), (0,1), (1,0), (1,1)) ever occur. All 100 distinct
actions fire except 16 (the never-fired are the error variants
that the bar's cases do not reach).

## Per-entry prune results (all 26)

24 break at the screen; the failing case and symptom are in
`still-checker-dt-result.json` under `entry_results`. The
pattern: every load-bearing entry breaks on the first heavy
case ("check induction principle" or "check Type 0 : Type 0
(Girard)") within 0–2s, usually as mode ERROR, a register-pool
exhaustion, or "control fired total 0.0". The two holds are
the gating (vacuous, above) and eta (genuine, above).

## Caveats (read before quoting the floor)

- **The floor is against the 41-case bar, not a universal
  minimum.** A case outside the bar could read weights this
  floor prunes.
- **±500 softness, inherited.** Comparison bits (512) is a bare
  literal; the derived-bit threshold layers (337) have a ~130
  estimated component (w_gt). The floor inherits that ±500.
- **Possible double count (up to ~512).** Comparison bits and
  the derived-bit layers appear to overlap (both cover the
  VAR-idx / SUBST-idx / level-max threshold layers), so the
  baseline may be overstated by up to ~512 and the floor by
  the same.
- **Analytic entries are cost models of inline code, not
  stored matrices.** Their floor share is coverage-measured,
  not code-pruned — except eta and the gating, which were
  code-pruned (eta genuinely, gating vacuously).
- **Eta truncation gating (2,278) is a coarse analytic count**
  (ctrl stack // 64).

## Files

- `still_checker_dt.py` — the checker (this run).
- `still-checker-dt-result.json` — floor, twins, batch
  confirm, per-entry results, coverage, caveats, corrections.
- `still-checker-dt-map.json` — per-entry prune ledger
  (also the resume file for the entry sweep).
- `STILL-CHECKER-DT.md` — this file.
