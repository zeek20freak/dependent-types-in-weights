"""Tests for the independent dependent kernel (rung 1 sealing pass).

Two layers, both derived from RULES_DT.md alone:
1. Direct de Bruijn unit tests for the machinery (shift, subst,
   normalize) on raw tuples -- no translation involved.
2. Rule-derived cases in named form, translated to de Bruijn for
   this kernel. The named form is authoring notation only; the
   kernel never sees a name. Cases are my own constructions from
   the rules (dependent identity, dependent const, a type-level
   function, conversion through a beta-redex, S, universe edges,
   eta edges, rejections with recorded paths).

The CASES list is imported by the comparison driver, which runs
both kernels on it. The runner only executes under __main__.
"""

from kernel_dt_independent import (Var, Lam, App, Pi, Type, Nat, Zero,
                                    Succ, Rec, show,
                                    shift, subst, normalize, Meter,
                                    defeq, check_or_fail, infer_or_fail,
                                    CheckError)

# ------------------------------------------------- named authoring form
# Tuples with string tags, only so cases can be written readably.
# The kernel never sees these; translate() converts them.

def V(name):
    return ("var", name)


def L(name, ty, body):
    return ("lam", name, ty, body)


def A(f, a):
    return ("app", f, a)


def P(name, ty, body):
    return ("pi", name, ty, body)


def T(i):
    return ("type", i)


def arr(t, u):
    return P("_", t, u)


def translate(term, env):
    """Named term -> de Bruijn term. env is a list of names,
    innermost binder first; Var(name) is env.index(name)."""
    tag = term[0]
    if tag == "var":
        return ("var", env.index(term[1]))
    if tag == "lam":
        return ("lam", translate(term[2], env),
                translate(term[3], [term[1]] + env))
    if tag == "pi":
        return ("pi", translate(term[2], env),
                translate(term[3], [term[1]] + env))
    if tag == "app":
        return ("app", translate(term[1], env), translate(term[2], env))
    if tag == "nat" or tag == "zero":
        return term
    if tag == "succ":
        return ("succ", translate(term[1], env))
    if tag == "rec":
        return ("rec", translate(term[1], env), translate(term[2], env),
                translate(term[3], env), translate(term[4], env))
    return term  # type


def translate_ctx(ctx_dict):
    """Named context (insertion-ordered name -> named type) ->
    (de Bruijn context list, env). Each type is translated in the
    context of the names declared before it, so the invariant
    'type of Var i is shift(ctx[i], i+1)' holds."""
    env = []
    ctx_db = []
    for name, ty in ctx_dict.items():
        ctx_db.append(translate(ty, env))
        env = [name] + env
    return ctx_db[::-1], env


# ------------------------------------------------------------- contexts

T0, T1, T2 = T(0), T(1), T(2)
A_, B_, C_ = V("A"), V("B"), V("C")
CTX0 = {"A": T0, "B": T0, "C": T0}

ID_DEP = L("A", T0, L("x", V("A"), V("x")))
ID_DEP_TY = P("A", T0, P("x", V("A"), V("A")))
TYPE_FN = L("A", T0, arr(V("A"), V("A")))

# ------------------------------------------------------------- cases
# kind: "check" (must accept), "check_reject" (must reject with
# path/kind), "infer_reject", "defeq" (want a bool).

CASES = [
    {"name": "dependent identity",
     "kind": "check", "term": ID_DEP, "type": ID_DEP_TY, "ctx": CTX0},
    {"name": "dependent const",
     "kind": "check",
     "term": L("A", T0, L("B", T0, L("x", V("A"),
                                    L("y", V("B"), V("x"))))),
     "type": P("A", T0, P("B", T0, P("x", V("A"),
                                      P("y", V("B"), V("A"))))),
     "ctx": CTX0},
    {"name": "type-level function",
     "kind": "check", "term": TYPE_FN, "type": P("A", T0, T0),
     "ctx": CTX0},
    {"name": "conversion through a beta-redex in the expected type",
     "kind": "check", "term": L("x", B_, V("x")),
     "type": A(TYPE_FN, B_), "ctx": CTX0},
    {"name": "the beta-redex is itself a type",
     "kind": "check", "term": A(TYPE_FN, B_), "type": T0, "ctx": CTX0},
    {"name": "eta through check (neutral head, Lam under it)",
     "kind": "check",
     "term": A(V("K"), V("a")),
     "type": A(V("H"), L("B", T0, A(V("F"), V("B")))),
     "ctx": dict(CTX0, F=P("A", T0, T0),
                 H=arr(P("A", T0, T0), T0),
                 K=P("x", A_, A(V("H"), V("F"))), a=A_)},
    {"name": "Type 0 : Type 1",
     "kind": "check", "term": T0, "type": T1, "ctx": {}},
    {"name": "Pi at mixed max: domain level 0, codomain level 1",
     "kind": "check",
     "term": P("x", A_, T0),
     "type": T1, "ctx": CTX0},
    {"name": "nested universes accumulate to Type 2",
     "kind": "check",
     "term": P("A", T0, P("B", T1, arr(V("A"), V("B")))),
     "type": T2, "ctx": {}},
    {"name": "Pi over Type 1 lands at Type 2",
     "kind": "check", "term": P("A", T1, T1), "type": T2, "ctx": {}},
    {"name": "S with universe variables",
     "kind": "check",
     "term": L("f", arr(A_, arr(B_, C_)),
               L("g", arr(A_, B_),
                 L("x", A_, A(A(V("f"), V("x")), A(V("g"), V("x")))))),
     "type": arr(arr(A_, arr(B_, C_)), arr(arr(A_, B_), arr(A_, C_))),
     "ctx": CTX0},
    {"name": "applied dependent identity",
     "kind": "check", "term": A(A(ID_DEP, A_), V("x")), "type": A_,
     "ctx": dict(CTX0, x=A_)},
    {"name": "eta with a redex expected type",
     "kind": "check", "term": L("x", A_, V("x")),
     "type": A(L("P", T0, V("P")), arr(A_, A_)), "ctx": CTX0},
    {"name": "defeq eta: f =?= lam x. f x",
     "kind": "defeq", "term": V("f"),
     "type": L("x", A_, A(V("f"), V("x"))), "want": True,
     "ctx": dict(CTX0, f=arr(A_, B_), g=arr(A_, B_))},
    {"name": "defeq eta, symmetric: lam x. f x =?= f",
     "kind": "defeq", "term": L("x", A_, A(V("f"), V("x"))),
     "type": V("f"), "want": True,
     "ctx": dict(CTX0, f=arr(A_, B_), g=arr(A_, B_))},
    {"name": "defeq distinct variables are not equal",
     "kind": "defeq", "term": V("f"), "type": V("g"), "want": False,
     "ctx": dict(CTX0, f=arr(A_, B_), g=arr(A_, B_))},
    {"name": "defeq alpha",
     "kind": "defeq", "term": L("x", A_, V("x")),
     "type": L("y", A_, V("y")), "want": True, "ctx": CTX0},
    {"name": "defeq eta does not fire for a non-function neutral",
     "kind": "defeq", "term": V("a"),
     "type": L("x", A_, A(V("a"), V("x"))), "want": False,
     "ctx": dict(CTX0, a=A_)},
    {"name": "Girard: Type 0 : Type 0",
     "kind": "check_reject", "term": T0, "type": T0,
     "path": (), "fail_kind": "mismatch", "ctx": {}},
    {"name": "Type 1 : Type 1 (wrong universe)",
     "kind": "check_reject", "term": T1, "type": T1,
     "path": (), "fail_kind": "mismatch", "ctx": {}},
    {"name": "polymorphic identity type : Type 0 (wrong universe)",
     "kind": "check_reject",
     "term": P("A", T0, arr(V("A"), V("A"))), "type": T0,
     "path": (), "fail_kind": "mismatch", "ctx": {}},
    {"name": "dependent application, wrong argument",
     "kind": "check_reject", "term": A(A(ID_DEP, A_), V("y")),
     "type": A_, "path": ("arg",), "fail_kind": "mismatch",
     "ctx": dict(CTX0, y=B_)},
    {"name": "Pi domain is not a type",
     "kind": "infer_reject", "term": P("x", V("f"), V("x")),
     "path": ("domain",), "fail_kind": "not-a-type",
     "ctx": dict(CTX0, f=arr(A_, A_))},
    {"name": "binder annotation is not a type",
     "kind": "infer_reject", "term": L("x", V("f"), V("x")),
     "path": ("binder",), "fail_kind": "not-a-type",
     "ctx": dict(CTX0, f=arr(A_, A_))},
    {"name": "type-level self-application",
     "kind": "check_reject", "term": L("A", T0, A(V("A"), V("A"))),
     "type": P("A", T0, T0), "path": ("body", "func"),
     "fail_kind": "not-pi", "ctx": CTX0},
    {"name": "Pi A:Type 0. A -> A : Type 1",
     "kind": "check", "term": P("A", T0, arr(V("A"), V("A"))),
     "type": T1, "ctx": {}},
    {"name": "capture a2 (sealing-pass regression)",
     "kind": "check",
     "term": A(L("A", T0, L("x", V("A"), L("A", T0, V("x")))), V("B")),
     "type": P("x", V("B"), P("A2", T0, V("B"))), "ctx": CTX0},
    {"name": "dependent-family capture (sealing-pass regression)",
     "kind": "check", "term": L("x", V("A"), A(V("f"), V("x"))),
     "type": P("x", V("A"), A(V("F"), V("x"))),
     "ctx": dict(CTX0, F=P("y", V("A"), T0),
                 f=P("y", V("A"), A(V("F"), V("y"))))},
]

# ------------------------------------------------------------- Nat (rung 2)
# Authoring forms for the new term forms; translate() above handles
# them. Cases mirror test_kernel_dt_nat.py (19), same names.

N_ = ("nat",)
Z_ = ("zero",)


def S_(t):
    return ("succ", t)


def R_(P, z, s, n):
    return ("rec", P, z, s, n)


def num(n):
    t = Z_
    for _ in range(n):
        t = S_(t)
    return t


const_motive = L("_", N_, N_)
add_step = L("k", N_, L("r", N_, S_(V("r"))))
add = L("m", N_, L("n", N_, R_(const_motive, V("n"), add_step, V("m"))))
dbl_step = L("k", N_, L("r", N_, S_(S_(V("r")))))
double = L("n", N_, R_(const_motive, Z_, dbl_step, V("n")))

ind_ty = P("P", arr(N_, T0),
           P("z", A(V("P"), Z_),
             P("s", P("n", N_, arr(A(V("P"), V("n")),
                                    A(V("P"), S_(V("n"))))),
               P("n", N_, A(V("P"), V("n"))))))
ind = L("P", arr(N_, T0),
        L("z", A(V("P"), Z_),
          L("s", P("n", N_, arr(A(V("P"), V("n")),
                                 A(V("P"), S_(V("n"))))),
            L("n", N_, R_(V("P"), V("z"), V("s"), V("n"))))))

ctx_rec = {"P": arr(N_, T0),
           "z": A(V("P"), Z_),
           "s": P("n", N_, arr(A(V("P"), V("n")),
                               A(V("P"), S_(V("n"))))),
           "n": N_}

NAT_CASES = [
    {"name": "Nat : Type 0",
     "kind": "check", "term": N_, "type": T0, "ctx": {}},
    {"name": "Zero : Nat",
     "kind": "check", "term": Z_, "type": N_, "ctx": {}},
    {"name": "Succ Zero : Nat",
     "kind": "check", "term": num(1), "type": N_, "ctx": {}},
    {"name": "Succ (Succ Zero) : Nat",
     "kind": "check", "term": num(2), "type": N_, "ctx": {}},
    {"name": "add : Nat -> Nat -> Nat",
     "kind": "check", "term": add, "type": arr(N_, arr(N_, N_)),
     "ctx": {}},
    {"name": "add 2 3 : Nat",
     "kind": "check", "term": A(A(add, num(2)), num(3)), "type": N_,
     "ctx": {}},
    {"name": "double : Nat -> Nat",
     "kind": "check", "term": double, "type": arr(N_, N_), "ctx": {}},
    {"name": "induction principle: λP. λz. λs. λn. Rec P z s n",
     "kind": "check", "term": ind, "type": ind_ty, "ctx": {}},
    {"name": "induction principle type : Type 1",
     "kind": "check", "term": ind_ty, "type": T1, "ctx": {}},
    {"name": "add 2 3 ≡ 5 (iota computes)",
     "kind": "defeq", "term": A(A(add, num(2)), num(3)),
     "type": num(5), "want": True, "ctx": {}},
    {"name": "add 2 3 ≢ 4",
     "kind": "defeq", "term": A(A(add, num(2)), num(3)),
     "type": num(4), "want": False, "ctx": {}},
    {"name": "double 2 ≡ 4",
     "kind": "defeq", "term": A(double, num(2)), "type": num(4),
     "want": True, "ctx": {}},
    {"name": "iota on Zero: Rec P z s Zero ≡ z",
     "kind": "defeq", "term": R_(V("P"), V("z"), V("s"), Z_),
     "type": V("z"), "want": True, "ctx": ctx_rec},
    {"name": "stuck Rec is itself: Rec P z s n ≡ Rec P z s n",
     "kind": "defeq", "term": R_(V("P"), V("z"), V("s"), V("n")),
     "type": R_(V("P"), V("z"), V("s"), V("n")), "want": True,
     "ctx": ctx_rec},
    {"name": "Succ of a non-Nat: Succ (λx:A. x)",
     "kind": "check_reject", "term": S_(L("x", V("A"), V("x"))),
     "type": N_, "path": ("pred",), "fail_kind": "mismatch",
     "ctx": {"A": T0}},
    {"name": "motive is not a function: Rec Nat ...",
     "kind": "infer_reject", "term": R_(N_, Z_, add_step, Z_),
     "path": ("motive",), "fail_kind": "not-pi", "ctx": {}},
    {"name": "motive codomain is not a universe: P : Nat -> Nat",
     "kind": "infer_reject",
     "term": R_(L("x", N_, Z_), Z_, add_step, Z_),
     "path": ("motive",), "fail_kind": "not-a-type", "ctx": {}},
    {"name": "motive codomain is neutral, not a universe",
     "kind": "infer_reject",
     "term": R_(V("P"), Z_, add_step, Z_),
     "path": ("motive",), "fail_kind": "not-a-type",
     "ctx": {"F": P("n", N_, T0),
             "P": P("n", N_, A(V("F"), V("n")))}},
    {"name": "Rec on a non-Nat target",
     "kind": "infer_reject",
     "term": R_(const_motive, Z_, add_step, L("x", V("A"), V("x"))),
     "path": ("target",), "fail_kind": "mismatch", "ctx": {"A": T0}},
    {"name": "Rec base of the wrong type",
     "kind": "infer_reject",
     "term": R_(const_motive, L("x", V("A"), V("x")), add_step, Z_),
     "path": ("base",), "fail_kind": "mismatch", "ctx": {"A": T0}},
    {"name": "Zero : Nat -> Nat",
     "kind": "check_reject", "term": Z_, "type": arr(N_, N_),
     "path": (), "fail_kind": "mismatch", "ctx": {}},
]


def run_case(case):
    """Run one named case against the independent kernel. Returns
    (ok, line, detail) where detail carries the machine-readable
    outcome: verdict, failure path/kind, beta steps."""
    ctx_db, env = translate_ctx(case["ctx"])
    meter = Meter()
    kind = case["kind"]
    if kind == "check":
        term = translate(case["term"], env)
        ty = translate(case["type"], env)
        fail = check_or_fail(ctx_db, term, ty, meter)
        ok = fail is None
        return ok, ("PASS  %s" % case["name"] if ok else
                    "FAIL  %s: rejected with %s" % (case["name"], fail)), \
            {"verdict": ok, "fail": fail, "beta_steps": meter.beta_steps}
    if kind == "check_reject":
        term = translate(case["term"], env)
        ty = translate(case["type"], env)
        fail = check_or_fail(ctx_db, term, ty, meter)
        if fail is None:
            return False, "FAIL  %s: accepted but should reject" % case["name"], \
                {"verdict": True, "fail": None, "beta_steps": meter.beta_steps}
        ok = (tuple(fail["path"]) == tuple(case["path"])
              and fail["kind"] == case["fail_kind"])
        return ok, ("PASS  %s (rejected at %s, %s)"
                    % (case["name"], list(fail["path"]), fail["kind"]) if ok else
                    "FAIL  %s: rejected at %s, %s -- expected %s, %s"
                    % (case["name"], list(fail["path"]), fail["kind"],
                       list(case["path"]), case["fail_kind"])), \
            {"verdict": False, "fail": fail, "beta_steps": meter.beta_steps}
    if kind == "infer_reject":
        term = translate(case["term"], env)
        _, fail = infer_or_fail(ctx_db, term, meter)
        if fail is None:
            return False, "FAIL  %s: inferred but should reject" % case["name"], \
                {"verdict": True, "fail": None, "beta_steps": meter.beta_steps}
        ok = (tuple(fail["path"]) == tuple(case["path"])
              and fail["kind"] == case["fail_kind"])
        return ok, ("PASS  %s (rejected at %s, %s)"
                    % (case["name"], list(fail["path"]), fail["kind"]) if ok else
                    "FAIL  %s: rejected at %s, %s -- expected %s, %s"
                    % (case["name"], list(fail["path"]), fail["kind"],
                       list(case["path"]), case["fail_kind"])), \
            {"verdict": False, "fail": fail, "beta_steps": meter.beta_steps}
    if kind == "defeq":
        t = translate(case["term"], env)
        u = translate(case["type"], env)
        try:
            got = defeq(ctx_db, t, u, meter)
        except CheckError as e:
            return False, "FAIL  %s: defeq raised %s" % (case["name"], e), \
                {"verdict": None, "fail": e.as_dict(),
                 "beta_steps": meter.beta_steps}
        ok = got == case["want"]
        return ok, ("PASS  %s (defeq = %s)" % (case["name"], got) if ok else
                    "FAIL  %s: defeq = %s, want %s"
                    % (case["name"], got, case["want"])), \
            {"verdict": got, "fail": None, "beta_steps": meter.beta_steps}
    raise ValueError("unknown case kind %r" % (kind,))


# ------------------------------------------- direct de Bruijn unit tests

def run_unit_tests():
    out = []

    def check_unit(name, got, want):
        out.append((got == want,
                    "PASS  unit %s" % name if got == want else
                    "FAIL  unit %s: got %r, want %r" % (name, got, want)))

    # shift: free variables bump, bound variables do not.
    check_unit("shift var", shift(Var(0), 1), Var(1))
    check_unit("shift under binder, bound stays",
               shift(Lam(Type(0), Var(0)), 1), Lam(Type(0), Var(0)))
    check_unit("shift under binder, free bumps",
               shift(Lam(Var(0), Var(1)), 1), Lam(Var(1), Var(2)))
    # subst: replace, drop, shift under binders.
    check_unit("subst app",
               subst(App(Var(0), Var(1)), 0, Type(0)),
               App(Type(0), Var(0)))
    check_unit("subst under binder shifts the substitute",
               subst(Lam(Type(0), Var(1)), 0, Var(0)),
               Lam(Type(0), Var(1)))
    check_unit("subst drops the bound index",
               subst(App(Var(0), Var(2)), 1, Type(0)),
               App(Var(0), Var(1)))
    check_unit("subst replaces and drops",
               subst(App(Var(1), Var(2)), 1, Type(0)),
               App(Type(0), Var(1)))
    # normalize: full reduction, including under binders (whnf would
    # stop at the head Lam and leave the inner redex alone).
    m = Meter()
    check_unit("normalize beta",
               normalize(App(Lam(Type(0), Var(0)), Type(0)), m), Type(0))
    check_unit("normalize reduces under a binder",
               normalize(Lam(Type(0),
                             App(Lam(Type(0), Var(0)), Var(0))), Meter()),
               Lam(Type(0), Var(0)))
    m2 = Meter()
    normalize(Lam(Type(0), App(Lam(Type(0), Var(0)), Var(0))), m2)
    check_unit("normalize under binder costs one beta step",
               m2.beta_steps, 1)
    # defeq on raw de Bruijn: eta against a context variable.
    # f : A -> A, stored as written in the declaration context [A]:
    # Pi(Var 0, Var 1) -- the codomain's Var 1 is A under the Pi
    # binder. (Pi(Var 1, Var 1) would be A -> f, not A -> A.)
    ctx = [Pi(Var(0), Var(1)), Type(0)]  # f : A -> A, A : Type 0
    check_unit("defeq eta, de Bruijn",
               defeq(ctx, Var(0), Lam(Var(1), App(Var(1), Var(0)))),
               True)
    check_unit("defeq eta, wrong neutral",
               defeq(ctx, Var(0), Lam(Var(1), App(Var(0), Var(0)))),
               False)
    # Nat unit tests: shift/subst through the new forms, iota.
    def num_db(n):
        t = Zero()
        for _ in range(n):
            t = Succ(t)
        return t
    check_unit("shift succ",
               shift(Succ(Var(0)), 1), Succ(Var(1)))
    check_unit("shift rec, no binder to cross",
               shift(Rec(Var(0), Var(1), Var(2), Var(3)), 1),
               Rec(Var(1), Var(2), Var(3), Var(4)))
    check_unit("subst rec",
               subst(Rec(Var(0), Var(1), Var(2), Var(3)), 1, Type(0)),
               Rec(Var(0), Type(0), Var(1), Var(2)))
    check_unit("normalize iota, add 2 3 = 5",
               normalize(Rec(Lam(Nat(), Nat()), num_db(3),
                             Lam(Nat(), Lam(Nat(), Succ(Var(0)))),
                             num_db(2)), Meter()),
               num_db(5))
    check_unit("normalize stuck rec is itself",
               normalize(Rec(Var(0), Var(1), Var(2), Var(3)), Meter()),
               Rec(Var(0), Var(1), Var(2), Var(3)))
    m3 = Meter()
    normalize(Rec(Lam(Nat(), Nat()), num_db(3),
                  Lam(Nat(), Lam(Nat(), Succ(Var(0)))), num_db(2)), m3)
    check_unit("normalize iota, add 2 3 costs 3 iota steps",
               m3.iota_steps, 3)
    return out


if __name__ == "__main__":
    results = run_unit_tests()
    for case in CASES:
        ok, line, _ = run_case(case)
        results.append((ok, line))
    for case in NAT_CASES:
        ok, line, _ = run_case(case)
        results.append((ok, line))
    for ok, line in results:
        print(line)
    passed = sum(1 for ok, _ in results if ok)
    print("\n%d/%d passed" % (passed, len(results)))
    if passed != len(results):
        raise SystemExit(1)
