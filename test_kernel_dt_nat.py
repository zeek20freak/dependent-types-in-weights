"""Tests for the v0.2 dependent kernel (rung 2: Nat + eliminator).

Nat is the second mechanism — inductive elimination, motive-dependent
typing, iota reduction in whnf. The rung-1 suite (test_kernel_dt.py,
20/20) must still pass unchanged; this file is the new rung's own
tests: computation (addition, doubling), the induction principle as
a theorem, and rejections with recorded failure paths.
"""

from kernel_dt import (Var, Lam, App, Pi, Type, Nat, Zero, Succ, Rec,
                       check, check_or_fail, infer_or_fail, defeq,
                       CheckError)

A = Var("A")
T0, T1 = Type(0), Type(1)
N = Nat()
Z = Zero()


def arr(t, u):
    """Non-dependent arrow, as a Pi with an unused binder."""
    return Pi("_", t, u)


def num(n):
    """The literal n, as Succ applied n times to Zero."""
    t = Z
    for _ in range(n):
        t = Succ(t)
    return t


# Constant motive: P = λ_:Nat. Nat.
const_motive = Lam("_", N, N)
# Step for addition: λk:Nat. λr:Nat. Succ r.
add_step = Lam("k", N, Lam("r", N, Succ(Var("r"))))
# add m n = Rec P n add_step m.
add = Lam("m", N, Lam("n", N, Rec(const_motive, Var("n"), add_step,
                                  Var("m"))))
# double n = Rec P Zero (λk r. Succ (Succ r)) n.
dbl_step = Lam("k", N, Lam("r", N, Succ(Succ(Var("r")))))
double = Lam("n", N, Rec(const_motive, Z, dbl_step, Var("n")))

# The induction principle, as a type and as a term.
P = Var("P")
ind_ty = Pi("P", arr(N, T0),
            Pi("z", App(P, Z),
               Pi("s", Pi("n", N, arr(App(P, Var("n")),
                                     App(P, Succ(Var("n"))))),
                  Pi("n", N, App(P, Var("n"))))))
ind = Lam("P", arr(N, T0),
          Lam("z", App(P, Z),
              Lam("s", Pi("n", N, arr(App(P, Var("n")),
                                     App(P, Succ(Var("n"))))),
                  Lam("n", N, Rec(P, Var("z"), Var("s"), Var("n"))))))

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


# --- Nat, basic ------------------------------------------------------------

expect_ok("Nat : Type 0", N, T0)
expect_ok("Zero : Nat", Z, N)
expect_ok("Succ Zero : Nat", num(1), N)
expect_ok("Succ (Succ Zero) : Nat", num(2), N)

# --- Computation -----------------------------------------------------------

expect_ok("add : Nat -> Nat -> Nat",
          add, arr(N, arr(N, N)))
expect_ok("add 2 3 : Nat",
          App(App(add, num(2)), num(3)), N)
expect_ok("double : Nat -> Nat",
          double, arr(N, N))

# --- The induction principle ----------------------------------------------

expect_ok("induction principle: λP. λz. λs. λn. Rec P z s n",
          ind, ind_ty)
expect_ok("induction principle type : Type 1", ind_ty, T1)

# --- Definitional equality, direct (iota) ----------------------------------

expect_defeq("add 2 3 ≡ 5 (iota computes)",
             {}, App(App(add, num(2)), num(3)), num(5), True)
expect_defeq("add 2 3 ≢ 4",
             {}, App(App(add, num(2)), num(3)), num(4), False)
expect_defeq("double 2 ≡ 4",
             {}, App(double, num(2)), num(4), True)

# Iota on Zero, and a stuck recursor, with the motive as a variable.
ctx_rec = {"P": arr(N, T0),
           "z": App(P, Z),
           "s": Pi("n", N, arr(App(P, Var("n")),
                               App(P, Succ(Var("n"))))),
           "n": N}
expect_defeq("iota on Zero: Rec P z s Zero ≡ z",
             ctx_rec, Rec(P, Var("z"), Var("s"), Z), Var("z"), True)
expect_defeq("stuck Rec is itself: Rec P z s n ≡ Rec P z s n",
             ctx_rec, Rec(P, Var("z"), Var("s"), Var("n")),
             Rec(P, Var("z"), Var("s"), Var("n")), True)

# --- Terms that must fail, with the failure path --------------------------

ctx_a = {"A": T0}

# Succ of a non-Nat: the predecessor check fails.
expect_reject("Succ of a non-Nat: Succ (λx:A. x)",
              Succ(Lam("x", A, Var("x"))), N,
              path=("pred",), kind="mismatch", ctx=ctx_a)

# The motive is not a function from Nat: Nat itself is a type.
expect_infer_reject("motive is not a function: Rec Nat ...",
                    Rec(N, Z, add_step, Z),
                    path=("motive",), kind="not-pi")

# Regression: the motive codomain must BE a universe, not merely
# be a type (sealing pass, 2026-11-25 — the independent kernel
# caught the reference accepting Nat -> Nat as a motive, failing
# only later at the base check). P = λx:Nat. Zero has type
# Nat -> Nat: its codomain is a type, but P n would be a natural,
# not a type.
expect_infer_reject("motive codomain is not a universe: P : Nat -> Nat",
                    Rec(Lam("x", N, Z), Z, add_step, Z),
                    path=("motive",), kind="not-a-type")

# Same, with a neutral codomain: F : Nat -> Type 0, P : Nat -> F n.
# F n is a type, but it is not a universe tag, so the motive does
# not have type Nat -> Type i.
ctx_neutral = {"F": Pi("n", N, T0),
               "P": Pi("n", N, App(Var("F"), Var("n")))}
expect_infer_reject("motive codomain is neutral, not a universe",
                    Rec(Var("P"), Z, add_step, Z),
                    path=("motive",), kind="not-a-type", ctx=ctx_neutral)

# The target is not a Nat.
expect_infer_reject("Rec on a non-Nat target",
                    Rec(const_motive, Z, add_step,
                        Lam("x", A, Var("x"))),
                    path=("target",), kind="mismatch", ctx=ctx_a)

# The base has the wrong type: P Zero is Nat, the base is A -> A.
expect_infer_reject("Rec base of the wrong type",
                    Rec(const_motive, Lam("x", A, Var("x")), add_step, Z),
                    path=("base",), kind="mismatch", ctx=ctx_a)

# Zero against a function type.
expect_reject("Zero : Nat -> Nat",
              Z, arr(N, N), path=(), kind="mismatch")


# --- Report ---------------------------------------------------------------

for ok, line in results:
    print(line)
passed = sum(1 for ok, _ in results if ok)
print(f"\n{passed}/{len(results)} passed")
if passed != len(results):
    raise SystemExit(1)
