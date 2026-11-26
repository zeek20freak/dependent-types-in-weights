# The Checker As Weights

## The two questions, answered

**Can the checker be a fixed circuit?** As a pure feedforward circuit,
only for bounded terms — the circuit grows with the bound. As a
*recurrent* circuit, yes, in general: fixed weights reused across
steps, the state as memory. That is what is built here, and it is the
same shape as the Qwenputer's in-weight CPU — the weights are the
transition function, the state is the RAM.

**What is the smallest weight configuration that checks STLC proofs?**
First measurement, not the floor: **4,849 nonzero parameters** for this
fragment and encoding (4,833 + 16 for the LAM_ABC kind added 2026-11-24
so the gate can judge S). The Still ran on the 4,833-weight registry
and found a 602-weight floor on the 11-term bar; that floor needs
re-measuring on the new registry and the 12-term bar (S included).

## What it is

A CEK-style inference machine, running entirely on explicit matrices:

- **Types** are fixed-layout binary trees: 15 slots, one-hot over
  {ARR, A, B, C, EMPTY} — a 75-dim vector. Every tree operation is a
  fixed permutation or a small threshold circuit, because a subtree's
  slots are fixed positions. Arrow construction is one 75×150 matrix
  plus a bias; splitting is a permutation; equality is a two-layer
  threshold net (XNOR per entry, AND over entries).
- **Terms** are fixed-layout binary trees: 63 slots, one-hot over 17
  node kinds — a 63×17 constant input matrix. A Lam's body is its
  left child; an App's function is its left child, its argument its
  right child. Variables are de Bruijn indices (no names).
- **State** is a 1,748-dim vector: node pointer (63), mode (4:
  INFER/RETURN/HALT/ERROR), context stack (4 slots × 76), result
  register (75), control stack (6 frames × 217). Stack push/pop are
  shift matrices. Reading a node is a matmul: pointer times the term
  matrix.
- **Control** is one 25×8 matrix: it reads (mode, node kind, top
  frame kind) and fires exactly one of 8 actions (Lam, App, Var,
  return-from-Lam, return-from-AppFun, return-from-AppArg, Halt,
  Error). The only bilinear step is the action gating — one action's
  matrices applied, the rest multiplied by zero.

## The count (nonzero parameters, 4,849 total)

The control-stack shift matrices dominate: 2 × 1,085 = 2,170 — the
price of a 6-frame stack with 217-dim frames. The equality circuit is
450. The context shifts are 456. Everything else is small
(arrow 70, splits 35 each, control 29). The term input (63×17) is
not counted — it is the program being checked, not the checker.

## Verification

- A plain-logic reference machine runs side by side and must agree
  **step for step** on every term. It does, on all 11.
- Verdicts match `kernel.py` 11/11: 5 proofs accepted, 6 terms
  rejected (wrong type, self-application, unbound variable, wrong
  argument type, applying a non-function, wrong return type).
- Three bugs found and fixed during the build, all in the encoding,
  none in the logic: a hardcoded 4-token width after adding the third
  base type; split outputs missing the EMPTY one-hot in unused slots,
  which broke equality; error branches that kept the frame the
  reference pops. The lockstep check caught all three.

## Boundaries (honest)

- **Fragment bounds** (parameters, not a redesign): terms depth ≤ 5,
  binder types depth ≤ 1 over A/B/C, plus the one nested-arrow binder
  `A→(B→C)` (LAM_ABC, added 2026-11-24 for S), inferred types depth ≤ 3,
  context depth ≤ 4, control depth ≤ 6. Wider bounds mean wider
  vectors and a bigger count, same design.
- **The gating** is bilinear. In a strict feedforward net it would be
  one more layer of AND gates; it is kept as gating here and stated
  as such.
- **4,849 is a first measurement, not the floor.** The obvious
  compressions: the shift matrices are pure permutations (routing,
  not learned weights); the 63-dim frame pointer is the biggest
  single field; the T/U field matrices are shared. The Still's floor
  (602) was measured on the 4,833-weight registry and the 11-term bar;
  it needs re-measuring on the new registry and the 12-term bar.
- The machine checks; it does not *find* proofs. The proposer is
  still missing — that is the other half of the loop.

## Dependent types — rung 1 (design + plain kernel)

*This section lives here rather than in GRADED-PROPOSER.md because
this step is the gate side, not the proposer side: the dependent
kernel is the checker's next form, and this file is the gate's
design document. GRADED-PROPOSER.md is the proposer-side ledger,
and this step changes nothing about the proposer — the proposer
half is closed on STLC at 10/10 on all nine arms.*

### The fragment

`RULES_DT.md` holds the full design. Rung 1 is **Π + universes +
conversion** — genuinely dependent, not STLC in disguise: types
are terms, a type-level function computes a type from a term
(`λA:Type 0. A -> A : Π A:Type 0. Type 0`), and checking must
evaluate. In: Π-types, λ with dependent binders, application,
conversion via whnf + definitional equality (alpha, beta, eta
for functions), universes `Type i : Type (i+1)` with Pi formation
at `max`. Out, with reasons stated in RULES_DT.md: Nat +
eliminator (rung 2 — inductive elimination is a second mechanism,
not more of the first), Prop (proof irrelevance +
impredicativity is a second axis), cumulativity (strict levels
keep defeq a pure equality), universe polymorphism (levels are
concrete small integers), inference/tactics/`sorry`/axioms
(carried over from STLC).

### The kernel, and the hard parts stated honestly

`kernel_dt.py` — a separate file, not an extension of `kernel.py`,
so the sealed STLC artifact stays sealed. Five rules (Var, Type,
Lam, Pi, App). The hard parts:

- **whnf is pure head-beta.** There is nothing else to reduce: no
  definitions to unfold (no delta — no environment, no `let`, no
  constants) and no recursor to compute (no iota — Nat is rung 2).
  It needs no context and carries a fuel bound (1000 beta steps)
  as a safety net, not a load-bearing part: every well-typed term
  in this fragment normalizes, so whnf terminates on well-typed
  terms.
- **defeq is untyped except where eta forces a type.** Eta for
  functions needs a type to expand against, so when exactly one
  side is a Lam the other side's type is inferred, and if it
  whnf's to a Pi the neutral is eta-expanded. Everything else is
  structural on whnf'd forms. Eta is included for Π-types only —
  the only types whose inhabitants are determined by observation
  (application). A subtlety worth stating: eta can never fire
  when comparing two *types* (a Lam is never a type, so two
  types can never differ by eta) — it fires in defeq's structural
  recursion, on function terms (App funcs, Lam bodies). The test
  suite proves it there, not at the top level.
- **Universes are strict.** `Type i : Type (i+1)`, Pi at `max`,
  no `Type : Type` (Girard), no cumulativity. Levels are small
  ints; `max`/`+1` is the whole of level arithmetic.
- **Failure paths keep kernel.py's shape exactly**
  (path/kind/expected/actual/message) — the graded proposer reads
  failure depth off that shape. Implementation differs
  (exception-carried paths instead of duplicated `_or_fail`
  logic); the shape is identical. `check_or_fail` (the public
  entry) verifies the expected type is itself a type; internal
  `check` trusts it, since every type reaching it is validated
  at the top or computed by `infer`.

Two test drafts failed before the suite went green, both times
the kernel being right and the test being wrong: the first
eta-through-check draft used `λy:B. H a y` as an expected type
(a function `B -> B`, not a type), the second used
`λA:Type 0. F A` (a type-level function, not a type). Both were
caught by the expected-type validation — the validation doing
real work. The passing form puts the Lam under a neutral (`H`),
so whnf cannot beta-reduce it away and only eta closes it.

### Test results

`test_kernel_dt.py`: **18/18 passed.**

- 10 proofs: dependent identity, dependent const, the type-level
  function, conversion through a beta-redex in the expected type,
  the redex itself as a type, eta through check, `Type 0 : Type 1`,
  `Π A:Type 0. A -> A : Type 1`, S (with `A, B, C` as universe
  variables), the applied dependent identity.
- 3 direct defeq checks: eta (`f ≡ λx:A. f x`), distinct
  variables unequal, alpha-equivalence.
- 5 rejections with recorded failure paths: `Type 0 : Type 0`
  (Girard, path `()`, mismatch); dependent application with the
  wrong argument (path `("arg",)`, mismatch); binder annotation
  that is not a type (path `("binder",)`, not-a-type); type-level
  self-application (path `("body", "func")`, not-pi); the
  polymorphic identity type against `Type 0` (path `()`,
  mismatch — it lives in `Type 1`).

Regression gate: `test_kernel.py` **11/11** (STLC, `kernel.py`
untouched), S via `kernel.py` PASS, S via `kernel_dt.py` PASS.

### Observer-view prediction

The view earns promotion only by predicting something new that
is then measured — and the stated example is which quantities in
the dependent-types checker are observer-dependent vs
independent. This checker makes it precise (full statement in
RULES_DT.md):

- **Predicted independent: verdicts, and universe levels.**
  Any two sound implementations agree on accept/reject (soundness
  + confluence); levels are computed by `max`, deterministically.
- **Predicted dependent: whnf step counts, defeq unfolding
  counts, and first-failure paths.** They depend on reduction
  strategy and traversal order — the observer's frame. Sharpest
  form: **the graded proposer's grade is frame-dependent** —
  failure depth changes under a different traversal order.

Measurement (named, not claimed): a second whnf strategy and an
arg-first traversal variant of `infer_or_fail`, same suite —
verdicts must agree 100%, step counts and failure paths must
differ. If verdicts differ, the prediction is wrong and the view
takes the hit.

### Next rung: the weights kernel — open design questions

Not solved here; stated honestly, grounded in what
`weights_kernel.py` actually is (CEK machine, 8 actions, fixed
15-slot type trees over `{ARR, A, B, C, EMPTY}`, 63-slot term
trees, de Bruijn, context stack 4, control stack 6):

1. **Types are terms now.** The 15-slot type encoding cannot
   represent Pi binders or `Type i`. The type encoding becomes a
   term-tree encoding (Pi nodes, Type nodes with a level field).
   And the term encoding's kind-per-Lam-type table breaks: STLC
   binder types were depth ≤ 1 over 3 bases (13 kinds); in DT a
   binder's type is an arbitrary type term — kinds cannot
   enumerate types anymore. The machine needs types-as-vectors
   (a type is a tree in a register), not kind tags. Open: the
   smallest fixed layout holding Pi types and universe levels,
   and how big the vectors get.
2. **Conversion needs evaluation.** whnf is a head-beta loop.
   The STLC machine never substitutes — it uses a context stack
   (CEK). whnf needs substitution or closures: a CEK machine
   with explicit environments avoids capture-avoiding
   substitution, but then defeq must compare closures, not
   trees. Open: whnf-as-CEK-with-closures vs
   whnf-as-substitution-on-trees — which maps to matrices, and
   what does the equality circuit compare?
3. **defeq is a two-term comparison with a type.** The STLC
   machine has one result register and one control stack; defeq
   needs whnf on two terms (and their types, for eta) and a
   comparison — a comparison mode with two term pointers, two
   whnf states, and a type register. The control matrix (8
   actions) grows; the action set and frame kinds multiply.
   Open: how many actions, and does the control matrix stay
   small?
4. **Eta needs types during comparison.** Type-directed defeq
   must know the type of a neutral while comparing — the machine
   must carry types alongside terms in comparison mode. The
   STLC machine's frames carry `(T, U)` for APPARG; DT frames
   need more.
5. **Universe levels are small ints, but they live inside
   types.** `max`/`+1` are small threshold circuits — but levels
   appear inside type trees (`Type i` as a term), so the type
   encoding needs a level field per Type node. Open: level field
   width (how many levels?) and whether level arithmetic stays a
   small circuit.
6. **Fuel becomes load-bearing.** The plain kernel's whnf fuel
   becomes the machine's step budget; DT checks take many more
   steps (whnf inside defeq inside check), so the budget matters
   in a way it didn't for STLC. Open: the step bound for the DT
   test suite, and whether the control-stack depth bounds it.
7. **Type validation is part of the machine.** The plain kernel
   validates expected types (public entry) and binder
   annotations (infer) — the weights machine must do the same;
   validation becomes actions, not an assumption. Open: which
   actions validate, and at what cost?

Files: `RULES_DT.md` (design), `kernel_dt.py` (plain kernel),
`test_kernel_dt.py` (18/18).

## Dependent types — rung 1 sealing pass (2026-11-25)

The DT rung 1 kernel (`kernel_dt.py`, 18/18 + STLC regression 11/11) was
sealed by a second, independent derivation. A fresh kernel
(`kernel_dt_independent.py`) was built from `RULES_DT.md` alone — never
reading `kernel_dt.py` or the DT section of this file — with three forced
structural differences: de Bruijn indices (capture structurally
impossible), arg-first App traversal (argument inferred before the
function), and full beta-normalization as the conversion strategy
(instead of head-beta whnf). Its own rule-derived suite
(`test_kernel_dt_independent.py`) is 37/37. Comparison driver
`compare_dt.py`; adversarial battery `battery_dt.py`; hostile read
`hostile_dt.py`; full numbers in `sealing-pass-result.json`.

**Comparison (43 cases = 18 shared + 25 mine): 43/43 verdict agreement,
43/43 failure-path agreement.** No disagreement on the shared suite.

**Observer measurement — both halves hold.** Independent half: verdicts
agree 43/43 on the suite and 500/500 on fuzz. Dependent half: whnf vs
full-normalize step counts differ on 3 of 4 strategy probes (redex under
a Pi binder 0 vs 1; redex in an argument 1 vs 2; nested redexes under two
binders 0 vs 2; identical only when the sole redex is at the head, 1 vs
1). First-failure paths differ on doubly-bad terms exactly as predicted
by traversal order (arg-first vs fn-first). Failure-kind distribution
differs in the fuzz (mine mismatch 367 / not-pi 51 / not-a-type 36 vs ref
337 / 73 / 44) while the totals agree (451 rejects, 49 accepts) — the
frame showing up in first-failure kinds.

**Adversarial battery:** eta 5/5, universe 8/8, fuel-Omega 5/5 verdict
agreement, doubly-bad 3/3, generator 200/200 pairs both accept with 0
disagreements, type-of-type + scope 200/200, fuzz 500/500 verdict
agreement. Substitution lemma MEASURED: d1 (App) 37/37, d2 (body)
108/108, generated directly (100→300 trials; the first run's thin
counts 10/10 and 21/21 came from collecting trials off top-level
terms). **Fuel is a safety net, not load-bearing:** all five
Omega-adjacent terms reject (kind not-pi) before whnf can loop; fuzz
fuel exhaustions 0/0, max 0.003 s/case, max 2 beta steps; every
whnf/normalize call site only ever sees validated types.

**Hostile read: 9/10 probes agree. The one disagreement is a real bug,
and the sealing pass caught it.** Capture case a2 —
`(λA:Type 0. λx:A. λA:Type 0. x) B` against `Π x:B. Π A2:Type 0. B`:
the independent kernel accepts; the reference kernel rejects.
Adjudicated against RULES_DT.md: **the independent kernel is right.**
The inner Lam's body `x` has type A_outer (the middle binder), so by
the Lam rule its type is `Π A_inner:Type 0. A_outer`, and beta-reducing
the type by hand gives `Π x:B. Π A:Type 0. B` — exactly the expected
type. The reference kernel instead infers the inner Pi's codomain as
A_inner: its named representation builds `Pi("A", Type 0, Var("A"))`
where `Var("A")` is meant to be the outer A, and its own name-based
reading captures it with the inner binder. The bug fires only when a
Lam's body type mentions an outer variable whose name equals the Lam's
binder name (shadowing) — no test in either suite had that shape, which
is why the suites never caught it. **Proposed fix (NOT applied):** when
the Lam rule builds the Pi type, alpha-rename the Pi binder to a fresh
name not occurring free in the codomain (or represent types with de
Bruijn indices, as the independent kernel does). **Seal status: NOT
sealed** — conditional on that fix, after which `compare_dt.py` and
`battery_dt.py` must be re-run and a2 must flip to agreement with no
other case changing. The independent kernel needs no change.

Other hostile probes were measured zeros: universe smuggling through
the identity function (argument `: Type 1` against a `Type 0` domain)
rejected by both kernels (H1, H2, H5); untyped defeq does not overreach
on eta against a non-function variable (H4); eta and beta under Pi
binders complete in both kernels (H3, H10); shadowing controls accept
in both (H6, H7); a Pi whose codomain is a Lam rejected by both (H8).
Three early battery "failures" (a2's expected type, c5's redex
argument, c5b's expectation) were adjudicated as test-authoring errors
— the kernels were right, the expectations were fixed — and the
independent kernel's `"binder"` position name was aligned to the
reference's `"domain"` for Pi domains.

**Seal (2026-11-25, fix applied and gate re-run): SEALED.** The proposed
blind rename of the Pi binder at construction was shown WRONG before
applying — it would regress the dependent-family case (`λx:A. f x :
Π x:A. F x`, where the body's type mentions the Lam binder itself and
the binder name must stay; the case accepts on the unfixed kernel).
The fix actually applied is rename-on-entry: in the Lam rules of both
`infer` and `check`, when the binder name is already bound in the
context, freshen it with `_fresh(var, free_vars(body) | set(ctx))`
before extending the context, alpha-converting the body via the
capture-avoiding `subst`; and in `defeq`, all three fresh-name
selections now also avoid `set(ctx)`, so a fresh binder never shadows
a context binding. The gate re-run: DT suite 20/20 (two new regression
tests, a2 and the dependent-family case), STLC 11/11, independent
suite 37/37, comparison 43/43 verdict + 43/43 failure-path agreement,
battery capture 5/5 with a2 flipped to agreement, hostile read 10/10,
fuzz 500/500 with the failure-kind distribution unchanged — the only
case that changed anywhere is a2, exactly the gate as specified. The
pre-fix NOT-sealed record is preserved in
`sealing-pass-result-v1-not-sealed.json`; `sealing-pass-result.json`
now records the seal.

## The DT weights kernel — design v0.1 (2026-11-25)

The fragment is the full sealed one: **Π + universes + conversion +
Nat + eliminator** (rung 2 sealed 2026-11-25). Nat-first was chosen
so the encoding is designed once, for the bigger fragment, rather
than built twice. The template is `weights_kernel.py` (STLC CEK
machine, 8 actions, 4,849 weights): the DT machine keeps the shape
— CEK, modes + frames, one control matrix, action gating — and
changes what the state holds. The seven open questions from the
rung-1 design, answered:

### 1. One encoding: terms are types

There is a single fixed-layout binary tree. Node kinds: `LAM`,
`PI`, `APP`, `VAR` (de Bruijn), `TYPE` (with a level field),
`NAT`, `ZERO`, `SUCC`, `REC`, `EMPTY`. A type is a tree in a
register; the context stack and the result register hold trees,
not kind tags. The STLC kind-per-Lam-type table is gone — a
binder's type is an arbitrary type term, so kinds cannot enumerate
types anymore.

A Lam has two children that matter: left = binder type, right =
body. A Pi: left = domain, right = codomain. An App: left = func,
right = arg. A Succ: left = predecessor. `Rec` has four components
(motive, base, step, target) and a binary node has two children,
so `Rec` is a three-node chain: the `REC` node holds the motive
(left) and a `REC3` node (right); `REC3` holds the base (left)
and a `REC2` node (right); `REC2` holds the step (left) and the
target (right). Two extra node kinds, depth +2 — the price of
keeping the binary tree and its fixed permutations. Reading a
binder type is a subtree copy (a fixed permutation, since the
left subtree occupies fixed slots relative to the node).

### 2. whnf as substitution on trees

whnf is a head-beta loop plus iota, over a term register, by
substitution — not closures. The plain kernel's whnf uses
capture-avoiding `subst`; the machine does the same operation as
a fixed mux (for each variable position, select the argument tree
or the original). Substitution keeps defeq as tree comparison
(the STLC equality circuit, extended to the new kinds); closures
would force defeq to compare environments too. The whnf machine
is a CEK-style loop with its own small frame stack for nested
reductions: reduce the App's func, then decide beta; reduce the
Rec's target, then decide iota (Zero → base, Succ → step applied,
neutral → stuck). Fuel is the loop bound (see §6).

### 3–4. defeq as a compare mode, with a type register

A `COMPARE` mode with two term registers (left/right). It whnf's
the left (via the whnf machine), stores it, whnf's the right,
stores it, then compares structurally with the equality circuit.
Eta is the reason for the type register: when exactly one side is
a Lam, the machine must infer the other side's type — so defeq
pushes a frame ("resume comparison after inferring the type"),
runs the infer machine on the neutral, then resumes. The frames
multiply: compare frames (binder types, bodies under a fresh
binder, func/arg) plus the infer frames the eta case needs. The
control matrix grows from 8 actions to however many the modes
need — the count is measured after the build, not predicted. It
stays a lookup table (rows = dispatch conditions, columns =
actions), not a computation. Nat gets no eta: its inhabitants are
observed by the eliminator, not by application — so the compare
mode needs no Nat-eta case, and the Rec comparison is structural
on all four components.

### 5. Universe levels: a level field, width 0–3

Each `TYPE` node carries a level field, one-hot over {0, 1, 2, 3}.
`max` and `+1` are small threshold circuits on the field. Levels
0–2 cover the sealed suite (Type 0, Type 1, the induction
principle's type at Type 1); 3 is headroom. The bound (levels ≤ 3)
is a fragment parameter, stated, not hidden.

### 6. Fuel is load-bearing, and measured

The machine carries an explicit step counter and a max-steps
bound. DT checks take many more steps than STLC (whnf inside
defeq inside check), so the bound matters in a way it didn't
before. The bound is set from measurement: run the DT + Nat
suites, take the max steps, add headroom. Fuel is still a bound,
not a semantic part — well-typed terms terminate — but the number
is load-bearing and is recorded, not assumed.

### 7. Validation is actions, not an assumption

The public entry validates the expected type (infer it, require a
universe) and infer validates binder annotations before extending
the context. In the machine these are explicit actions in the
control matrix (a validate step in the Lam/Pi handling and at the
entry), not assumptions. Their cost is counted with the rest.

### Bounds (measured from the suite, not guessed)

Term depth, context depth, and control depth are set by measuring
the sealed suites (DT 20 + Nat 21 cases, plus the examples and
the REPL terms) and taking the smallest bound that fits them —
the same rule as STLC's bounds, which were parameters, not a
redesign. The Rec chain (+2 depth) and the compare/whnf frames
push all three bounds up from STLC's; the measurements say by
how much. The level bound is 0–3 (§5).

### The bar for v0.1

- A plain-logic DT reference machine (the same design, no
  matrices) runs side by side and must agree **step for step**
  on every term — the STLC lockstep discipline, carried over.
- Verdicts match `kernel_dt.py` on the DT suite (20/20) and the
  Nat suite (21/21), including the two motive-codomain
  regressions the rung-2 seal caught.
- The weight count is measured (nonzero parameters, same
  registry discipline as STLC's 4,849) and the Still measures
  the floor on the new registry.
- The machine checks; it does not find proofs. The proposer is
  still the other half of the loop.
