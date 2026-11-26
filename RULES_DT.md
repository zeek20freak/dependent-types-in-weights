# A Fresh Lean, From First Principles — Dependent Types (Rung 1)

## The idea

STLC was proof of shape: the checker fits in weights, the proposer
closes every arm, the floor is measured. But STLC cannot say what
the Riemann proof says. The step from STLC to dependent types is
the real jump, and it is one idea: **a type is a term.**

In STLC a type is syntax and checking is structural — the checker
never has to compute. Once a type can mention a term, the checker
must evaluate during checking: a type like `(λA:Type 0. A -> A) B`
is a program, and the checker has to run it to `B -> B` before it
can compare. That evaluation is conversion, and conversion is the
whole of this rung.

## First principles

The four from RULES.md carry over unchanged:

1. **The kernel is the whole theory.** Five rules now, nothing else.
   No elaborator, no tactics, no `sorry`.
2. **Every term carries its own annotations.** The checker checks —
   it doesn't infer missing types.
3. **A proof is a term, a proposition is a type.** Curry-Howard,
   taken literally.
4. **Designed for weights.** Terms are small trees, checking is a
   fixed small computation, the only state is the context.

Plus three new ones, forced by dependence:

5. **Types are terms.** There is one syntactic class, not two. A
   "type" is a term whose type is a universe. Everything the
   kernel does to a term, it does to types too — including
   evaluation.
6. **Conversion is computed, never assumed.** Two types are the
   same only if the kernel can reduce one to the other (whnf) and
   compare them (defeq). No type is equal to another by declaration.
7. **Universes are stratified.** `Type i : Type (i+1)`, always, no
   exceptions. A single `Type : Type` would let the theory prove
   anything (Girard's paradox) — so the hierarchy is the price of
   consistency, and it is paid here, at the bottom.

## The theory: minimal dependent types

### Terms

- A variable: `x`
- An abstraction: `λx:A. t` — `A` is a *term* (a type)
- An application: `f a`
- A dependent function type: `Πx:A. B` — `B` is a type under `x : A`.
  When `x` does not occur in `B` this reads as the arrow `A -> B`.
- A universe: `Type i` — `i` a non-negative integer.

### Judgments

`Γ ⊢ t : T` — under context `Γ`, term `t` has type `T`. And `T`
itself must satisfy `Γ ⊢ T : Type i` for some `i`: the kernel
verifies every expected type is a type (at the public entry) and
every binder annotation is a type (in `infer`).

### The five rules — this is the entire kernel

**Var.** If `x : T` is in the context, then `x` has type `T`.

**Type.** `Type i` has type `Type (i+1)`.

**Lam.** If `A` is a type and `t` has type `U` under `Γ, x : A`,
then `λx:A. t` has type `Πx:A. U`.

**Pi.** If `A : Type i` and `B : Type j` under `Γ, x : A`, then
`Πx:A. B : Type (max i j)`.

**App.** If `f : Πx:A. B` and `a : A`, then `f a : B[a/x]` — the
codomain with the argument substituted in. The function's type
is reduced to weak head normal form first, so a function whose
type is a redex still applies.

### Conversion

Definitional equality — alpha, beta, and eta for functions:

- **Alpha:** bound names don't matter. The kernel renames binders
  to a shared fresh name when it compares them.
- **Beta:** `(λx:A. t) a ≡ t[a/x]`. Implemented by `whnf`: reduce
  head redexes until the head is not a redex. There is nothing
  else to reduce — no definitions to unfold (no delta: the
  fragment has no environment, no `let`, no constants) and no
  recursor to compute (no iota: that arrives with Nat).
- **Eta:** `f ≡ λx. f x` when `f : Πx:A. B`. Eta is the only
  part of conversion that needs a type: to expand a neutral you
  have to know it is a function. So `defeq` is untyped except
  for the one place it matters — when exactly one side is a Lam,
  the other side's type is inferred, and if it is a Pi the
  neutral is eta-expanded. There is no other eta in the fragment:
  Π-types are the only types whose inhabitants are determined by
  observation (application), so they are the only types with an
  eta law. When Nat lands, it gets none — its inhabitants are
  observed by the eliminator, not by application.

`whnf` needs no context (pure head-beta, nothing to unfold) and
carries a fuel bound (1000 beta steps) as a safety net, not a
load-bearing part: every well-typed term in this fragment
normalizes, so whnf terminates on well-typed terms. The bound
exists so an ill-typed term reaching whnf on an error path
cannot loop the kernel.

## What counts as a proof

A proof of a proposition `T` is a closed term `t` with `⊢ t : T`,
as before — but now `T` can compute. Examples (all in
`test_kernel_dt.py`, 18/18):

- Dependent identity: `λA:Type 0. λx:A. x` proves
  `Π A:Type 0. Π x:A. A` — the type mentions the bound variable.
- Dependent const: `λA B x y. x` proves
  `Π A B:Type 0. Π x:A. Π y:B. A`.
- A type-level function: `λA:Type 0. A -> A` proves
  `Π A:Type 0. Type 0` — a function from types to types.
- Conversion in the open: `λx:B. x` checks against the
  beta-redex `(λA:Type 0. A -> A) B`, which whnf's to `B -> B`.
- S, the STLC theorem, checks here too — with `A, B, C` as
  universe variables instead of built-in bases.

## The cut, stated explicitly

**Rung 1 is Π + universes + conversion. Nat is rung 2.** The
cut is deliberate, not a gap:

- **Nat, deferred.** The eliminator needs motive-dependent typing
  (`Nat.rec : Π (P : Nat → Type i). P zero → (Π n. P n → P (succ n))
  → Π n. P n`) and iota reduction in whnf (recursion on
  `zero`/`succ`). That is a second new mechanism — inductive
  elimination — not more of the first. It gets its own rung, its
  own tests (addition, the induction principle as a theorem).
  Nothing in rung 1 depends on it: the genuinely dependent
  examples (polymorphic identity, type-level functions) need no
  inductive.
- **Prop, deferred.** Proof irrelevance plus impredicative Π
  formation is a second axis of complexity. The fragment's job
  is to make types depend on terms, and `Type 0 : Type 1` does
  that with a consistent predicative hierarchy. Honest note: the
  real Riemann proof lives in Prop/Type, so Prop will be needed
  eventually — but not before Π + conversion is sealed.
- **No cumulativity.** `Type i` and `Type j` are equal only when
  `i = j`. Strict levels keep defeq a pure equality. (Coq has
  cumulativity; Lean's kernel does not — its defeq on sorts is
  level equality.)
- **No universe polymorphism.** No level variables, no `max`
  over variables — levels are concrete small integers. Real
  Lean's universe polymorphism is a later rung.
- **No inference, no tactics, no elaboration, no `sorry`,
  no axioms.** Carried over from STLC unchanged.

## Rung 2: Nat + eliminator (2026-11-25)

The fork's pick: Nat first, weights kernel after. The ladder
discipline wins — the weights encoding gets designed once, for
the bigger fragment, rather than built twice.

### New term forms

- `Nat` — the type of natural numbers. `Nat : Type 0`.
- `Zero` — `Zero : Nat`.
- `Succ n` — `Succ n : Nat` when `n : Nat`.
- `Rec P z s n` — the eliminator (recursor / induction
  principle). Four components, no binder: the motive `P`, the
  base case `z`, the step case `s`, the target `n`.

### The new rules

**Nat.** `Nat : Type 0`.

**Zero.** `Zero : Nat`.

**Succ.** If `n : Nat`, then `Succ n : Nat`.

**Rec.** If `P : Nat → Type i`, `z : P Zero`,
`s : Π n:Nat. P n → P (Succ n)`, and `n : Nat`, then
`Rec P z s n : P n`. The motive's universe is the result's
universe: `P n : Type i`. The eliminator is motive-dependent
typing — the type of the result depends on the target, through
`P`. That is the second mechanism: inductive elimination.

### Iota: the second reduction

`whnf` gains iota alongside beta:

- `Rec P z s Zero → z`
- `Rec P z s (Succ n) → s n (Rec P z s n)`

The subtlety: iota has to *see* the constructor, so `whnf`
reduces the target `n` to whnf first. If the target's whnf is
neutral (a variable, a stuck application), the `Rec` is stuck
and is itself the whnf — with the target replaced by its whnf,
the same move `whnf` already makes for `App`. Each iota step
consumes fuel, like each beta step; on well-typed terms the
bound stays a safety net (structural recursion terminates).

`defeq` gets iota for free: it calls `whnf` on both sides
first. Its new cases are structural — `Nat` vs `Nat`, `Zero`
vs `Zero`, `Succ` vs `Succ` on the predecessors, `Rec` vs
`Rec` on all four components. No new eta: Nat gets none, as
rung 1's doc already decided — its inhabitants are observed
by the eliminator, not by application.

### What counts as a proof, for Nat

- Addition, computed: `add = λm:Nat. λn:Nat. Rec (λ_:Nat. Nat) n
  (λ_ r. Succ r) m` — `add 2 3` checks against `Nat` and is
  definitionally `5`, by iota. That is the rung's central test:
  the kernel computes.
- The induction principle as a theorem: `λP. λz. λs. λn.
  Rec P z s n` proves `Π (P : Nat → Type 0). P Zero →
  (Π n:Nat. P n → P (Succ n)) → Π n:Nat. P n` — the
  eliminator's own type, inhabited by the eliminator.
- Rejections: `Succ` of a non-Nat, a motive that is not a
  function from `Nat`, a base of the wrong type, a `Rec` on a
  non-`Nat` target.

### The cut, restated for rung 2

- **Prop, still deferred.** Same axis as rung 1: proof
  irrelevance plus impredicative Π formation. The real
  Riemann proof will need it eventually — not before the
  weights kernel.
- **No Nat literals, no binary operators.** `Succ (Succ Zero)`
  is the only spelling of 2; addition is a term built from
  `Rec`, not a primitive. The kernel stays minimal.
- **No cumulativity, no universe polymorphism.** Carried over
  unchanged.

## Observer-view prediction

The observer view is a grounded-theory candidate: it earns
promotion only by predicting something new that is then
measured. The stated example of such a prediction is which
quantities in the dependent-types checker will be
observer-dependent vs independent. This checker makes the
prediction precise enough to check:

- **Independent (predicted): verdicts, and universe levels.**
  Any two sound implementations of this kernel agree on
  accept/reject for every term — that is soundness plus
  confluence (beta-eta reduction is confluent, so whnf's result
  is unique; the verdict depends only on the existence of a
  derivation, not the order it is found in). Universe levels
  are computed by `max`, deterministically — no choice is
  involved.
- **Dependent (predicted): whnf step counts, defeq unfolding
  counts, and first-failure paths.** The number of beta steps
  whnf takes, and the number of unfoldings defeq performs,
  depend on the reduction strategy (which side is reduced
  first, head-reduction vs full normalization) — the observer's
  frame. The first failure a checker reports depends on
  traversal order (in `App`, inferring the function before
  checking the argument, or the reverse, reports a different
  first failure). The sharpest form: **the graded proposer's
  grade is frame-dependent** — failure depth, the proposer's
  gradient, changes under a different traversal order.

**Measurement (not claimed — named, for a later run):** build a
second whnf strategy and an arg-first traversal variant of
`infer_or_fail`, run the same suite, and compare: verdicts must
agree 100%, step counts and failure paths must differ. If
verdicts differ, the prediction is wrong and the view takes the
hit; if only counts and paths differ, the prediction holds.

## Status

- v0.1 (2026-11-24/25): the rules (this file) + a reference kernel
  (`kernel_dt.py`, plain Python, no dependencies) + tests
  (`test_kernel_dt.py`, 18/18 — 10 proofs, 3 direct defeq checks,
  5 rejections with recorded failure paths). STLC regression:
  `test_kernel.py` 11/11, S via `kernel.py` and via `kernel_dt.py`.
  `kernel.py` untouched; the DT kernel is a separate file so the
  sealed STLC artifact stays sealed.
- v0.2 (2026-11-25): rung 2 — Nat + eliminator in the plain kernel.
  `kernel_dt.py` extended (Nat/Zero/Succ/Rec, iota in whnf);
  `test_kernel_dt_nat.py` 21/21 (addition computes, doubling, the
  induction principle as a theorem, 7 rejections with paths).
  SEALED 2026-11-25: independent kernel (de Bruijn, arg-first,
  full normalization) extended for Nat; compare 67/67 verdict +
  67/67 failure-path; fuzz 300/300 verdict (path-only differences
  on multi-error terms are the predicted frame-dependence). The
  seal caught a real bug: the reference accepted `Nat -> Nat` as
  a motive (codomain check too weak); fixed to require the
  codomain to normalize to a universe tag, with regression tests
  on both sides. Receipt: `sealing-pass-result-rung2.json`.
- Next: the weights kernel — Π-types, conversion, and Nat as
  matrices (open design questions in WEIGHTS.md). Publishing
  waits for the weights kernel — the plain checker is
  done-before work.
