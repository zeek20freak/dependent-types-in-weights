"""Hostile read of RULES_DT.md — attempts to break the rules.

Each probe is a term the rules should reject that a kernel might accept,
or a proof the rules justify that a kernel might reject. Every probe is
run on BOTH kernels; a disagreement is adjudicated against RULES_DT.md.
A probe that finds nothing is a measured zero: the search was capable of
finding a break (the a2 capture bug below is proof of that capability)
and found none on that probe.
"""
import json
import kernel_dt as R
from kernel_dt_independent import (
    check_or_fail as my_check, infer_or_fail as my_infer,
    defeq as my_defeq, Meter)
from test_kernel_dt_independent import (
    translate, translate_ctx, V, L, A, P, T)

CTX_N = {"A": T(0), "B": T(0), "C": T(0)}
CTX_R = {n: R.Type(0) for n in ("A", "B", "C")}
CTX_DB, ENV = translate_ctx(CTX_N)


def to_ref(term):
    """Tuple form (V/L/A/P/T) -> reference-kernel object."""
    tag = term[0]
    if tag == "var":
        return R.Var(term[1])
    if tag == "lam":
        return R.Lam(term[1], to_ref(term[2]), to_ref(term[3]))
    if tag == "app":
        return R.App(to_ref(term[1]), to_ref(term[2]))
    if tag == "pi":
        return R.Pi(term[1], to_ref(term[2]), to_ref(term[3]))
    if tag == "type":
        return R.Type(term[1])
    raise ValueError(tag)


def run_named(name, term, expected):
    """Check `term` against `expected` in both kernels; return a record."""
    ref_fail = R.check_or_fail(CTX_R, to_ref(term), to_ref(expected))
    my_fail = my_check(CTX_DB, translate(term, ENV),
                       translate(expected, ENV), Meter())
    return {"name": name, "ref_accepts": ref_fail is None,
            "mine_accepts": my_fail is None,
            "agree": (ref_fail is None) == (my_fail is None),
            "ref_fail": ref_fail, "my_fail": my_fail}


def run_defeq_named(name, t1, t2):
    ref_ok = R.defeq(CTX_R, to_ref(t1), to_ref(t2))
    my_ok = my_defeq(CTX_DB, translate(t1, ENV), translate(t2, ENV), Meter())
    return {"name": name, "ref": ref_ok, "mine": my_ok,
            "agree": ref_ok == my_ok}


results = []

# H1. Universe smuggling through the identity function.
# (λA:Type 0. λx:A. x) Type 0 — the argument Type 0 : Type 1, but the
# domain is Type 0. Rules must reject (no cumulativity, no polymorphism).
# If either kernel accepts, the universe hierarchy leaks.
results.append(run_named(
    "H1 universe smuggling via identity: (λA:Type 0. λx:A. x) Type 0",
    A(L("A", T(0), L("x", V("A"), V("x"))), T(0)),
    P("x", T(0), T(0))))

# H2. Same smuggling, but the argument is a Pi at Type 1.
# (λA:Type 0. λx:A. x) (Π A:Type 0. A) — the argument : Type 1, domain Type 0.
results.append(run_named(
    "H2 universe smuggling via identity, Pi argument",
    A(L("A", T(0), L("x", V("A"), V("x"))),
      P("A", T(0), V("A"))),
    P("x", T(0), T(0))))

# H3. Eta under a Pi binder (completeness probe): the rules justify
# Π f:(A→A). f ≡ Π f:(A→A). λx:A. f x by eta. A kernel that only
# eta-compares at the top level would reject this defeq.
results.append(run_defeq_named(
    "H3 eta under a Pi binder: Πf:(A→A). f ≡ Πf:(A→A). λx:A. f x",
    P("f", P("x", V("A"), V("A")), V("f")),
    P("f", P("x", V("A"), V("A")),
      L("x", V("A"), A(V("f"), V("x"))))))

# H4. Eta where the type is NOT a function (soundness probe).
# defeq(λx:A. x, A) with A : Type 0 (not a function type): untyped
# defeq might eta-expand and equate them, but they are not equal —
# a function is not a type. Both kernels should say False here.
results.append(run_defeq_named(
    "H4 eta against a non-function variable: λx:A. x vs A",
    L("x", V("A"), V("x")),
    V("A")))

# H5. A Girard-adjacent term: Π A:Type 0. A applied to itself through
# a type-level function. (λA:Type 0. A) (λA:Type 0. A) — the argument
# is a type-level function of type Π A:Type 0. Type 0, which is : Type 1,
# but the domain is Type 0. Must reject.
results.append(run_named(
    "H5 type-level self-application through identity",
    A(L("A", T(0), V("A")),
      L("A", T(0), V("A"))),
    P("A", T(0), T(0))))

# H6. Capture in the OTHER direction: a Lam whose binder shadows a
# variable the ARGUMENT mentions. (λA:Type 0. λx:A. x) A where the
# argument is the free A itself. Expected type Π x:A. A.
# (This is the a2 shape with the argument being the shadowed name.)
results.append(run_named(
    "H6 shadowing with the argument being the shadowed name",
    A(L("A", T(0), L("x", V("A"), V("x"))), V("A")),
    P("x", V("A"), V("A"))))

# H7. Deep shadowing chain: three binders all named A, body type
# mentions the middle one. (λA:Type 0. λA:Type 0. λx:A. x) B B.
# Inner A shadows; x's type is the middle A. After both applications
# the result should be Π x:B. B.
results.append(run_named(
    "H7 triple shadowing, body type mentions middle binder",
    A(A(L("A", T(0),
          L("A", T(0), L("x", V("A"), V("x")))),
        V("B")), V("B")),
    P("x", V("B"), V("B"))))

# H8. A Pi whose codomain is a Lam (a function used as a type).
# λf:(Π x:A. (λy:A. y)). f — the annotation's codomain is a Lam,
# not a type. Both kernels must reject: a Lam is never a type.
results.append(run_named(
    "H8 Pi codomain is a Lam (function as a type)",
    L("f", P("x", V("A"), L("y", V("A"), V("y"))), V("f")),
    P("f", P("x", V("A"), V("A")), V("A"))))

# H9. The a2 capture bug, stated as the hostile read's positive hit:
# (λA:Type 0. λx:A. λA:Type 0. x) B against Π x:B. Π A2:Type 0. B.
# Rules: accept (the inner Lam's type is Π A1:Type 0. A_outer).
# The reference kernel rejects; the independent kernel accepts.
results.append(run_named(
    "H9 (a2) shadowing capture: inner Lam type mentions outer A",
    A(L("A", T(0),
        L("x", V("A"), L("A", T(0), V("x")))),
      V("B")),
    P("x", V("B"), P("A2", T(0), V("B")))))

# H10. defeq of two types that differ only by a redex in the codomain
# under a binder: Π x:A. (λy:Type 0. y) A vs Π x:A. A. Must be True.
results.append(run_defeq_named(
    "H10 beta under a Pi binder: Πx:A. (λy:Type 0. y) A ≡ Πx:A. A",
    P("x", V("A"), A(L("y", T(0), V("y")), V("A"))),
    P("x", V("A"), V("A"))))

for r in results:
    status = "AGREE" if r["agree"] else "DISAGREE"
    if "ref_accepts" in r:
        print(f"{status} [{r['name']}] ref_accepts={r['ref_accepts']} "
              f"mine_accepts={r['mine_accepts']}")
    else:
        print(f"{status} [{r['name']}] ref={r['ref']} mine={r['mine']}")

json.dump(results, open("hostile_dt_result.json", "w"), indent=1,
          default=str)
print("wrote hostile_dt_result.json")
