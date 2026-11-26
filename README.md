# Dependent Types, Rung 1 — a minimal kernel, sealed

A reference implementation of a minimal dependent type theory:
**Π-types + universes + conversion (alpha / beta / eta-for-functions).**
Plain Python, no dependencies, no build step. The spec is
[`RULES_DT.md`](RULES_DT.md); the code is the spec in executable form.

## What it is

- **One idea:** a type is a term. Once a type can mention a term,
  checking must evaluate — a type like `(λA:Type 0. A -> A) B` is a
  program the checker has to run to `B -> B` before it can compare.
  Conversion is the whole of this rung.
- **Five rules, nothing else:** Var, Type (`Type i : Type (i+1)`),
  Lam, Pi (at `max` of the levels), App (with substitution).
  No elaborator, no tactics, no `sorry`, no axioms.
- **The cut, stated plainly:** no Nat (rung 2), no Prop, no
  cumulativity, no universe polymorphism. See "The cut" in
  `RULES_DT.md` for why each is deferred rather than missing.

## How to run it (stranger test)

Requires only Python 3.10+ (developed on 3.12). From this directory:

```sh
./run_tests.sh
```

or by hand:

```sh
python3 test_kernel_dt.py            # reference suite: 20/20
python3 test_kernel_dt_independent.py # independent suite: 37/37
python3 compare_dt.py                # both kernels, 43 cases: 43/43 verdict + failure-path agreement
python3 hostile_dt.py                # 10 hostile probes: 10/10
python3 battery_dt.py                # adversarial battery incl. 500-case fuzz: 500/500
python3 merge_seal.py                # merges battery + hostile into sealing-pass-result.json
python3 examples_dt.py               # 20 guided examples: 20/20
python3 test_repl.py                 # REPL parser verified against the kernel: 29/29
```

To play with it live, run `python3 repl_dt.py` — type a term like
`\A:Type 0. \x:A. x` and get its type back, or `<term> : <type>` for
an ACCEPT/REJECT with the failure path. `:defeq t == u`, `:decl`,
`:ctx`, `:help`, `:quit` inside the loop. The parser is a front end,
not part of the sealed kernel: it is verified in `test_repl.py`,
which re-expresses every example in surface syntax and checks the
verdicts match (29/29, including the a2 capture case).

`compare_dt.py`, `battery_dt.py` and `hostile_dt.py` rewrite the
`*.json` result files in place; the committed copies are the sealed
receipts from 2026-11-25. Re-running them should reproduce the same
verdicts. Note the pipeline order matters: `compare_dt.py` writes a
two-key `sealing-pass-result.json` (comparison + observer
measurement only); `merge_seal.py` rebuilds the full merged receipt
from it plus the battery and hostile files, and `run_tests.sh`
asserts the merged receipt is actually sealed.

Repair note (2026-11-25, found by this package's own stranger
test): the original `merge_seal.py` hardcoded the *pre-fix* state —
sealed False, battery capture "4/5 (a2 disagrees)", the a2
disagreement live — because it was written before the rename-on-entry
fix and never updated. Re-running it regenerated a NOT-sealed receipt
that contradicted the sealed file sitting next to it. It now reads
the agreement numbers from the battery/hostile files, keeps the a2
story as a `resolved_disagreements` entry (adjudication + fix
preserved, not erased), and refuses to present an unsealed result as
sealed. `sealing-pass-result-v1-not-sealed.json` is the pre-fix
record, kept for provenance.

## How to play

`examples_dt.py` is the guided tour: 8 proofs that pass, 5 terms
that fail (each showing the failure path), 4 "break it yourself"
proofs with a TRY line naming the one change that flips the verdict,
and 3 definitional-equality checks. Every example asserts its
expected verdict, so the file fails loudly if the kernel ever changes
under it — a readable second test suite, not just prints. The REPL
is the same loop made immediate: try a term, see the verdict, adjust,
try again.

## Why two kernels

`kernel_dt.py` is the reference kernel (named variables,
head-beta `whnf`, fn-first traversal). `kernel_dt_independent.py`
was derived from `RULES_DT.md` alone and is deliberately different
in three ways — de Bruijn indices (capture structurally impossible),
arg-first traversal, full beta-normalization instead of `whnf` — so
that agreement between the two is evidence about the rules, not
about shared code. The sealing pass ran both on the same 43 cases
plus a 500-case fuzz: **43/43 and 500/500 verdict agreement**.

The sealing pass caught a real bug this way: a capture bug in the
reference kernel's Lam→Pi construction (case `a2`), fixed by
rename-on-entry — freshen the binder before extending the context,
never a blind rename (which would regress the dependent-family
case). Both regression tests are in `test_kernel_dt.py`. The
pre-fix not-sealed record is kept in this package as
`sealing-pass-result-v1-not-sealed.json`, so the before/after is
auditable from the repo alone.

## What it proves (and what it doesn't)

- It proves the fragment checks: dependent identity, dependent
  const, type-level functions, conversion through a beta-redex, S
  with universe variables — and rejects Girard's `Type 0 : Type 0`
  and four other bad terms, each with a recorded failure path.
- It does **not** prove anything about Prop, or a real proof
  (the Riemann Zero Lean proof is the north star, several rungs
  away). Nat is sealed (rung 2, in this package). The weights
  kernel — this checker as matrices — is now in this package too:
  design in `WEIGHTS.md`, RefMachine in `weights_kernel_dt.py`,
  MatrixMachine in `matrix_machine_dt.py` (lockstep 41/41, count
  1,338,752), Still floor 47,687 shared weights
  (`STILL-CHECKER-DT.md`).

## Handoff

- **Receiver:** any stranger who wants to check the fragment
  unaided — the plain kernel, the weights kernel, and the Still
  floor are all in this package. The next rung (Prop, then a real
  proof term) is the open work.
- **Maintainer:** zeeker__, with Quine (Muse) as the
  continuity layer. Rung 1 sealed 2026-11-25; rung 2 Nat sealed
  the same day; the weights kernel sealed the same evening.
  The next rung (Prop) is the open decision.
- **Status:** sealed, held — weights kernel built and sealed
  2026-11-25. MatrixMachine lockstep 41/41 (count 1,338,752
  nonzero parameters, ±500 soft — two registry entries are
  estimates, disclosed in `MATRIX_MACHINE_STATUS.md`); independent
  check 41/41 under a fresh harness (its script was lost to /tmp
  cleanup — the numbers and hashes are recorded in
  `MATRIX_MACHINE_STATUS.md`, not re-runnable from this package);
  Still floor 47,687 shared weights (49,196 pruning-only), 3.56%
  of the registry — the machine is 96.3% never-read against the
  41-case bar. One open fork, named in `STILL-CHECKER-DT.md`: if
  `__init__` ctx writes count as scaffolding rather than machine
  ops, the floor is 4,648 shared. The publish gate is met — the
  weights kernel exists. Publishing (a public repo under
  zeeker__'s account) is zeeker__'s call.
