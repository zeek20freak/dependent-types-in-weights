"""Tests for the v0.1 dependent kernel (rung 1).

Proofs that must pass, terms that must fail — with the failure path
recorded for each rejection. A checker that accepts everything is
not a checker, and a failure with no location hides where the
search breaks (the Still's lesson, carried over from kernel.py).
"""

from kernel_dt import (Var, Lam, App, Pi, Type, check, check_or_fail,
                       infer_or_fail, defeq, CheckError)

A, B, C = Var("A"), Var("B"), Var("C")
T0, T1 = Type(0), Type(1)

# Base types are context variables of type Type 0 here — the DT
# kernel has no built-in bases; A, B, C are declared, not given.
ctx0 = {"A": T0, "B": T0, "C": T0}


def arr(t, u):
    """Non-dependent arrow, as a Pi with an unused binder."""
    return Pi("_", t, u)


results = []


def expect_ok(name, term, type_, ctx=None):
    fail = check_or_fail(ctx or {}, term, type_)
    if fail is None:
        results.append((True, f"PASS  {name}"))
    else:
        results.append((False, f"FAIL  {name}: rejected with {fail}"))


def expect_reject(name, term, type_, path, kind, ctx=None):
    fail = check_or_fail(ctx or {}, term, type_)
    if fail is None:
        results.append((False, f"FAIL  {name}: accepted but should reject"))
    elif tuple(fail["path"]) == tuple(path) and fail["kind"] == kind:
        results.append((True,
                        f"PASS  {name} (rejected at {list(fail['path'])}, "
                        f"{fail['kind']})"))
    else:
        results.append((False,
                        f"FAIL  {name}: rejected at {list(fail['path'])}, "
                        f"{fail['kind']} — expected {list(path)}, {kind}"))


def expect_infer_reject(name, term, path, kind, ctx=None):
    _, fail = infer_or_fail(ctx or {}, term)
    if fail is None:
        results.append((False, f"FAIL  {name}: inferred but should reject"))
    elif tuple(fail["path"]) == tuple(path) and fail["kind"] == kind:
        results.append((True,
                        f"PASS  {name} (rejected at {list(fail['path'])}, "
                        f"{fail['kind']})"))
    else:
        results.append((False,
                        f"FAIL  {name}: rejected at {list(fail['path'])}, "
                        f"{fail['kind']} — expected {list(path)}, {kind}"))


def expect_defeq(name, ctx, t, u, want):
    try:
        got = defeq(ctx, t, u)
    except CheckError as e:
        results.append((False, f"FAIL  {name}: defeq raised {e}"))
        return
    if got == want:
        results.append((True, f"PASS  {name} (defeq = {got})"))
    else:
        results.append((False, f"FAIL  {name}: defeq = {got}, want {want}"))


# --- Proofs that must pass -----------------------------------------------

# Dependent identity: the type mentions the bound variable A.
id_dep = Lam("A", T0, Lam("x", Var("A"), Var("x")))
expect_ok("dependent identity: λA:Type 0. λx:A. x : Π A:Type 0. Π x:A. A",
          id_dep, Pi("A", T0, Pi("x", Var("A"), Var("A"))), ctx0)

const_dep = Lam("A", T0,
                Lam("B", T0,
                    Lam("x", Var("A"),
                        Lam("y", Var("B"), Var("x")))))
expect_ok("dependent const: Π A:Type 0. Π B:Type 0. Π x:A. Π y:B. A",
          const_dep,
          Pi("A", T0, Pi("B", T0,
                         Pi("x", Var("A"),
                            Pi("y", Var("B"), Var("A"))))), ctx0)

# A type-level function: it computes a type from a term. This is the
# first genuinely dependent object in the suite — a function whose
# codomain is a universe, returning a type.
type_fn = Lam("A", T0, arr(Var("A"), Var("A")))
expect_ok("type-level function: λA:Type 0. A -> A : Π A:Type 0. Type 0",
          type_fn, Pi("A", T0, T0), ctx0)

# Conversion: the expected type is a beta-redex. The checker must
# whnf (λA:Type 0. A -> A) B to B -> B to see the identity fits.
expect_ok("conversion: λx:B. x against (λA:Type 0. A -> A) B",
          Lam("x", B, Var("x")), App(type_fn, B), ctx0)

# The beta-redex is itself a type (its type is Type 0).
expect_ok("(λA:Type 0. A -> A) B is a type",
          App(type_fn, B), T0, ctx0)

# Eta, through check. The inferred type of (K a) is the neutral H F,
# and the expected type is H (λB:Type 0. F B) — the second argument
# is F's eta-expansion. defeq compares the arguments structurally,
# and eta fires there. The Lam must sit under the neutral H: if it
# were the head of a redex, whnf would beta-reduce it away and the
# test would pass by beta, not eta. (Two earlier drafts of this
# test treated a type-level function as if it were a type; the
# kernel's type validation correctly rejected both — the validation
# doing its job.)
ctx_eta2 = dict(ctx0)
ctx_eta2.update({"F": Pi("A", T0, T0),
                 "H": arr(Pi("A", T0, T0), T0),
                 "K": Pi("x", A, App(Var("H"), Var("F"))),
                 "a": A})
expect_ok("eta through check: (K a) against H (λB:Type 0. F B)",
          App(Var("K"), Var("a")),
          App(Var("H"), Lam("B", T0, App(Var("F"), Var("B")))),
          ctx_eta2)

# Universes: Type 0 lives in Type 1, and so does the polymorphic
# identity type (its domain is a universe, level 1).
expect_ok("Type 0 : Type 1", T0, T1)
expect_ok("Π A:Type 0. A -> A : Type 1",
          Pi("A", T0, arr(Var("A"), Var("A"))), T1)

# S, the STLC theorem, lives in the dependent kernel too — A, B, C
# as universe variables instead of built-in bases.
S = Lam("f", arr(A, arr(B, C)),
        Lam("g", arr(A, B),
            Lam("x", A, App(App(Var("f"), Var("x")),
                            App(Var("g"), Var("x"))))))
expect_ok("S: (A->B->C) -> (A->B) -> A -> C",
          S, arr(arr(A, arr(B, C)), arr(arr(A, B), arr(A, C))), ctx0)

# Dependent identity, applied: (id_dep A) x : A.
expect_ok("applied dependent identity: (id_dep A) x : A",
          App(App(id_dep, A), Var("x")), A,
          ctx={**ctx0, "x": A})

# --- Definitional equality, direct ----------------------------------------

ctx_eta = dict(ctx0)
ctx_eta.update({"f": arr(A, B), "g": arr(A, B)})
expect_defeq("eta: f ≡ λx:A. f x",
             ctx_eta, Var("f"),
             Lam("x", A, App(Var("f"), Var("x"))), True)
expect_defeq("distinct variables are not equal: f ≢ g",
             ctx_eta, Var("f"), Var("g"), False)
expect_defeq("alpha: λx:A. x ≡ λy:A. y",
             ctx0, Lam("x", A, Var("x")),
             Lam("y", A, Var("y")), True)

# --- Terms that must fail, with the failure path --------------------------

# Girard: the hierarchy is strict — Type 0's type is Type 1.
expect_reject("Type 0 : Type 0 (Girard)",
              T0, T0, path=(), kind="mismatch")

# Dependent application, wrong argument: (id_dep A) expects an A,
# y is a B. The failure is in the argument position.
expect_reject("dependent application, wrong argument: (id_dep A) y, y : B",
              App(App(id_dep, A), Var("y")), A,
              path=("arg",), kind="mismatch", ctx={**ctx0, "y": B})

# A binder annotation that is not a type: f : A -> A is a function,
# not a universe, so λx:f. x has no well-formed binder.
expect_infer_reject("binder annotation is not a type: λx:f. x, f : A -> A",
                    Lam("x", Var("f"), Var("x")),
                    path=("binder",), kind="not-a-type",
                    ctx={**ctx0, "f": arr(A, A)})

# Type-level self-application: A : Type 0 is a type, not a function.
expect_reject("type-level self-application: λA:Type 0. A A",
              Lam("A", T0, App(Var("A"), Var("A"))),
              Pi("A", T0, T0), path=("body", "func"), kind="not-pi")

# Universe level: the polymorphic identity type lives in Type 1,
# not Type 0.
expect_reject("Π A:Type 0. A -> A : Type 0 (wrong universe)",
              Pi("A", T0, arr(Var("A"), Var("A"))), T0,
              path=(), kind="mismatch")


# Regression: the a2 capture bug (sealing pass, 2026-11-25). The inner
# Lam's body type mentions an outer variable sharing the binder's
# name; the Pi built for the inner Lam must not capture it.
expect_ok("capture: (λA:Type 0. λx:A. λA:Type 0. x) B : Π x:B. Π A2:Type 0. B",
          App(Lam("A", T0, Lam("x", A, Lam("A", T0, Var("x")))), B),
          Pi("x", B, Pi("A2", T0, B)), ctx=ctx0)

# Regression: the fix must not rename blindly. Here the body's type
# mentions the Lam binder itself (a type family applied to it), so
# the Pi binder must stay the same name — renaming it would leave
# the variable free. F : Πy:A. Type 0; f : Πy:A. F y.
ctx_fam = {**ctx0, "F": Pi("y", A, T0),
           "f": Pi("y", A, App(Var("F"), Var("y")))}
expect_ok("capture: (λx:A. f x) : Π x:A. F x (body type mentions the binder)",
          Lam("x", A, App(Var("f"), Var("x"))),
          Pi("x", A, App(Var("F"), Var("x"))),
          ctx=ctx_fam)


# --- Report ---------------------------------------------------------------

for ok, line in results:
    print(line)
passed = sum(1 for ok, _ in results if ok)
print(f"\n{passed}/{len(results)} passed")
if passed != len(results):
    raise SystemExit(1)
