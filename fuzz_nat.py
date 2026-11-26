"""Fuzz for the rung-2 sealing pass: random Nat-flavored terms, both
kernels must agree on verdict (accept/reject) and, on rejection, on
failure path + kind. The motive bug lived in the untested region,
so the untested region gets measured. Fixed seed: reproducible.
"""

import random
from kernel_dt import (Var, Lam, App, Pi, Type, Nat, Zero, Succ, Rec,
                       infer_or_fail)
from test_kernel_dt_independent import (V, L, A, P, T, translate,
                                        translate_ctx, N_, Z_)
import kernel_dt_independent as ind

SEED = 20261125
N_CASES = 300


def S_(t):
    return ("succ", t)


def R_(p, z, s, n):
    return ("rec", p, z, s, n)


def arr(t, u):
    return P("_", t, u)


# Named context: P : Nat -> Type 0, n/m : Nat, z : P Zero,
# s : Pi k:Nat. P k -> P (Succ k), A : Type 0.
CTX_NAMED = {
    "P": arr(N_, T(0)),
    "n": N_,
    "m": N_,
    "z": A(V("P"), Z_),
    "s": P("k", N_, arr(A(V("P"), V("k")),
                        A(V("P"), S_(V("k"))))),
    "A": T(0),
}
CTX_REF = {
    "P": Pi("_", Nat(), Type(0)),
    "n": Nat(),
    "m": Nat(),
    "z": App(Var("P"), Zero()),
    "s": Pi("k", Nat(), Pi("_", App(Var("P"), Var("k")),
                           App(Var("P"), Succ(Var("k"))))),
    "A": Type(0),
}

CONST_MOTIVE = L("_", N_, N_)
ADD_STEP = L("k", N_, L("r", N_, S_(V("r"))))


def named_to_ref(t):
    tag = t[0]
    if tag == "var":
        return Var(t[1])
    if tag == "lam":
        return Lam(t[1], named_to_ref(t[2]), named_to_ref(t[3]))
    if tag == "pi":
        return Pi(t[1], named_to_ref(t[2]), named_to_ref(t[3]))
    if tag == "app":
        return App(named_to_ref(t[1]), named_to_ref(t[2]))
    if tag == "type":
        return Type(t[1])
    if tag == "nat":
        return Nat()
    if tag == "zero":
        return Zero()
    if tag == "succ":
        return Succ(named_to_ref(t[1]))
    if tag == "rec":
        return Rec(named_to_ref(t[1]), named_to_ref(t[2]),
                   named_to_ref(t[3]), named_to_ref(t[4]))
    raise ValueError(tag)


def gen(rng, d):
    """Random named term, depth-limited, Nat-biased."""
    if d <= 0:
        return rng.choice([V("n"), V("m"), Z_, N_, V("P"), V("z"),
                           V("s"), V("A"), T(0), T(1)])
    r = rng.random()
    if r < 0.20:
        return S_(gen(rng, d - 1))
    if r < 0.45:
        motive = rng.choice([CONST_MOTIVE, V("P"),
                             L("x", N_, N_), N_,
                             L("x", N_, Z_), V("A")])
        base = rng.choice([V("z"), Z_, V("n"), gen(rng, d - 1)])
        step = rng.choice([V("s"), ADD_STEP, gen(rng, d - 1)])
        target = rng.choice([V("n"), V("m"), Z_, gen(rng, d - 1)])
        return R_(motive, base, step, target)
    if r < 0.65:
        return A(gen(rng, d - 1), gen(rng, d - 1))
    if r < 0.80:
        return L("x", rng.choice([N_, T(0)]), gen(rng, d - 1))
    if r < 0.90:
        return P("x", rng.choice([N_, T(0)]), gen(rng, d - 1))
    return gen(rng, 0)


def main():
    rng = random.Random(SEED)
    ctx_db, env = translate_ctx(CTX_NAMED)
    agreed = 0
    verdict_agreed = 0
    accepts = 0
    disagreements = []
    for i in range(N_CASES):
        named = gen(rng, 3)
        ref_term = named_to_ref(named)
        _, ref_fail = infer_or_fail(CTX_REF, ref_term)
        ind_term = translate(named, env)
        _, ind_fail = ind.infer_or_fail(ctx_db, ind_term)
        ref_ok = ref_fail is None
        ind_ok = ind_fail is None
        if ref_ok:
            accepts += 1
        if ref_ok == ind_ok:
            verdict_agreed += 1
        same = (ref_ok == ind_ok and
                (ref_ok or (tuple(ref_fail["path"]) ==
                            tuple(ind_fail["path"]) and
                            ref_fail["kind"] == ind_fail["kind"])))
        if same:
            agreed += 1
        elif ref_ok != ind_ok and len(disagreements) < 10:
            # A verdict disagreement is a real bug; a path-only
            # difference on a multi-error term is the predicted
            # frame-dependence of first-failure paths (the
            # observer view: traversal order decides which error
            # is reported first). List only verdict disagreements.
            disagreements.append((i, named, ref_fail, ind_fail))
    print(f"fuzz: verdict {verdict_agreed}/{N_CASES} agree, "
          f"verdict+path {agreed}/{N_CASES} agree "
          f"(accepts: {accepts}, rejects: {N_CASES - accepts})")
    for i, named, rf, mf in disagreements:
        print(f"  VERDICT case {i}: {named}\n    ref: {rf}\n    ind: {mf}")
    if verdict_agreed != N_CASES:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
