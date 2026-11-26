"""Adversarial battery for the rung 1 sealing pass.

(a) Capture-avoiding substitution under nested Pi-binders with
    shadowing names -- the classic bug habitat.
(b) Eta edge cases (both sides, under App, with redex types,
    neutral type a Pi only after reduction).
(c) Universe-level edges.
(d) The substitution lemma MEASURED: random well-typed terms,
    substitute, re-check typing in both kernels.
(e) Fuel proven a safety net, not load-bearing: Omega-adjacent
    ill-typed terms, plus a random-term fuzz; termination, the
    right failure kind, and a count of fuel exhaustions (expect 0).
(f) Doubly-bad terms: the observer measurement's path half --
    arg-first vs fn-first report different first failures.
(g) Type-of-type invariant and scope check on the random corpus.

Named cases run through compare_dt.run_mine (both kernels).
The generator works in de Bruijn; the reverse translator below
(de Bruijn -> named) feeds generated terms to the reference kernel.
"""

import json
import random
import time

import kernel_dt as R
from kernel_dt_independent import (Var, Lam, App, Pi, Type, shift, subst,
                                    normalize, Meter,
                                    defeq as my_defeq,
                                    check_or_fail as my_check_or_fail,
                                    infer_or_fail as my_infer_or_fail,
                                    infer as my_infer,
                                    CheckError as MyCheckError)
from test_kernel_dt_independent import (CASES as MY_CASES, translate,
                                         translate_ctx, V, L, A, P, T)
from compare_dt import run_mine, named_to_ref

T0, T1, T2, T3 = T(0), T(1), T(2), T(3)
A_, B_, C_ = V("A"), V("B"), V("C")
CTX0 = {"A": T0, "B": T0, "C": T0}


def arr(t, u):
    return P("_", t, u)


# ------------------------------------------------- reverse translator
# De Bruijn -> named, for feeding generated terms to the reference
# kernel. Binder names are fresh u<N> (never colliding with the free
# names A, B, C, ...); free variables come from free_names,
# innermost first.

_counter = [0]


def untranslate(t, stack, free_names):
    tag = t[0]
    if tag == "var":
        n = t[1]
        if n < len(stack):
            return ("var", stack[n])
        return ("var", free_names[n - len(stack)])
    if tag == "lam":
        _counter[0] += 1
        x = "u%d" % _counter[0]
        return ("lam", x, untranslate(t[1], stack, free_names),
                untranslate(t[2], [x] + stack, free_names))
    if tag == "pi":
        _counter[0] += 1
        x = "u%d" % _counter[0]
        return ("pi", x, untranslate(t[1], stack, free_names),
                untranslate(t[2], [x] + stack, free_names))
    if tag == "app":
        return ("app", untranslate(t[1], stack, free_names),
                untranslate(t[2], stack, free_names))
    return ("type", t[1])


# ------------------------------------------------------------ sections
# Named battery cases (kind/check/defeq as in run_mine).

CAPTURE = [
    {"name": "capture/a1 shadowed binders, inner wins",
     "kind": "check",
     "term": L("A", T0, L("x", V("A"), L("A", T0, L("x", V("A"), V("x"))))),
     "type": P("A", T0, P("x", V("A"), P("A", T0, P("x", V("A"), V("A"))))),
     "ctx": CTX0},
    {"name": "capture/a2 App type under a shadowing Pi",
     "kind": "check",
     "term": A(L("A", T0, L("x", V("A"), L("A", T0, V("x")))), V("B")),
     "type": P("x", V("B"), P("A2", T0, V("B"))),
     "ctx": CTX0},
    {"name": "capture/a3 beta-reduction must not capture (defeq True)",
     "kind": "defeq",
     "term": A(L("x", A_, L("y", B_, V("x"))), V("y")),
     "type": L("y2", B_, V("y")), "want": True,
     "ctx": dict(CTX0, y=A_)},
    {"name": "capture/a4 beta-reduction must not capture (defeq False)",
     "kind": "defeq",
     "term": A(L("x", A_, L("y", B_, V("x"))), V("y")),
     "type": L("y2", B_, V("y2")), "want": False,
     "ctx": dict(CTX0, y=A_)},
    {"name": "capture/a5 shadowed type variable in a Pi",
     "kind": "check",
     "term": P("A", T0, P("A", T0, V("A"))), "type": T1, "ctx": {}},
]

ETA = [
    {"name": "eta/b1 eta under an App head",
     "kind": "defeq",
     "term": A(V("f"), V("a")),
     "type": A(L("x", A_, A(V("f"), V("x"))), V("a")), "want": True,
     "ctx": dict(CTX0, f=arr(A_, B_), a=A_)},
    {"name": "eta/b2 neutral type is a Pi only after reduction",
     "kind": "defeq", "term": V("f"),
     "type": L("x", A_, A(V("f"), V("x"))), "want": True,
     "ctx": dict(CTX0, f=A(L("T", T0, V("T")), arr(A_, B_)))},
    {"name": "eta/b3 different neutrals are not eta-equal",
     "kind": "defeq", "term": V("f"),
     "type": L("x", A_, A(V("g"), V("x"))), "want": False,
     "ctx": dict(CTX0, f=arr(A_, B_), g=arr(A_, B_))},
    {"name": "eta/b4 eta-expansion against a redex expected type",
     "kind": "check", "term": L("x", A_, A(V("f"), V("x"))),
     "type": A(L("P", T0, V("P")), arr(A_, B_)),
     "ctx": dict(CTX0, f=arr(A_, B_))},
    {"name": "eta/b5 two eta-expansions, different bound names",
     "kind": "defeq", "term": L("x", A_, A(V("f"), V("x"))),
     "type": L("y", A_, A(V("f"), V("y"))), "want": True,
     "ctx": dict(CTX0, f=arr(A_, B_))},
]

UNIVERSE = [
    {"name": "univ/c1 Type 2 : Type 3",
     "kind": "check", "term": T2, "type": T3, "ctx": {}},
    {"name": "univ/c2 Pi over Type 2 lands at Type 3",
     "kind": "check", "term": P("A", T2, arr(V("A"), V("A"))),
     "type": T3, "ctx": {}},
    {"name": "univ/c2b same Pi rejected at Type 2",
     "kind": "check_reject", "term": P("A", T2, arr(V("A"), V("A"))),
     "type": T2, "path": (), "fail_kind": "mismatch", "ctx": {}},
    {"name": "univ/c3 Pi x:A. A lands at Type 0",
     "kind": "check", "term": P("x", V("A"), V("A")), "type": T0,
     "ctx": CTX0},
    {"name": "univ/c4 Pi x:A. Type 0 lands at Type 1",
     "kind": "check", "term": P("x", V("A"), T0), "type": T1,
     "ctx": CTX0},
    {"name": "univ/c4b same Pi rejected at Type 0",
     "kind": "check_reject", "term": P("x", V("A"), T0), "type": T0,
     "path": (), "fail_kind": "mismatch", "ctx": CTX0},
    {"name": "univ/c5 Pi with a redex domain (reduces to A)",
     "kind": "check",
     "term": P("x", A(L("T", T0, V("T")), V("A")), T0), "type": T1,
     "ctx": CTX0},
    {"name": "univ/c5b Pi with a redex domain, codomain x (rejected: "
              "x is a term of type A, not a type)",
     "kind": "check_reject",
     "term": P("x", A(L("T", T0, V("T")), V("A")), V("x")), "type": T0,
     "path": ("body",), "fail_kind": "not-a-type",
     "ctx": CTX0},
]

OMEGA = [
    {"name": "fuel/e1 Omega (term-level self-application)",
     "kind": "infer_reject",
     "term": A(L("x", A_, A(V("x"), V("x"))),
               L("x", A_, A(V("x"), V("x")))),
     "path": None, "fail_kind": None, "ctx": CTX0},
    {"name": "fuel/e2 Omega (type-level self-application)",
     "kind": "infer_reject",
     "term": A(L("A", T0, A(V("A"), V("A"))),
               L("A", T0, A(V("A"), V("A")))),
     "path": None, "fail_kind": None, "ctx": CTX0},
    {"name": "fuel/e3 Omega as the expected type",
     "kind": "check_reject",
     "term": L("x", A_, V("x")),
     "type": A(L("A", T0, A(V("A"), V("A"))),
               L("A", T0, A(V("A"), V("A")))),
     "path": None, "fail_kind": None, "ctx": CTX0},
    {"name": "fuel/e4 nested Omega",
     "kind": "infer_reject",
     "term": A(L("x", A_, A(V("x"), V("x"))),
               A(L("x", A_, A(V("x"), V("x"))),
                 L("x", A_, A(V("x"), V("x"))))),
     "path": None, "fail_kind": None, "ctx": CTX0},
    {"name": "fuel/e5 Omega as a binder annotation",
     "kind": "infer_reject",
     "term": L("x", A(L("A", T0, A(V("A"), V("A"))),
                      L("A", T0, A(V("A"), V("A")))), V("x")),
     "path": None, "fail_kind": None, "ctx": CTX0},
]

DOUBLY_BAD = [
    # Both sides independently ill-typed, so the first failure a
    # checker reports is decided by its traversal order alone.
    # arg-first (mine) reports the argument side; fn-first (ref)
    # reports the function side. Verdicts must agree; paths differ.
    {"name": "doubly-bad/f1 bad func (binder) + bad arg App(f, f)",
     "kind": "infer_reject",
     "term": A(L("x", V("f"), V("x")), A(V("f"), V("f"))),
     "path": None, "fail_kind": None,
     "ctx": dict(CTX0, f=arr(A_, A_))},
    {"name": "doubly-bad/f2 bad func (binder) + bad arg (binder)",
     "kind": "infer_reject",
     "term": A(L("x", V("f"), V("x")), L("y", V("g"), V("y"))),
     "path": None, "fail_kind": None,
     "ctx": dict(CTX0, f=arr(A_, A_), g=arr(A_, A_))},
    {"name": "doubly-bad/f3 bad func (binder) + bad arg (self-app)",
     "kind": "infer_reject",
     "term": A(L("x", V("f"), V("x")),
               L("y", A_, A(V("y"), V("y")))),
     "path": None, "fail_kind": None,
     "ctx": dict(CTX0, f=arr(A_, A_))},
]


def run_named_battery(cases):
    """Run named cases through run_mine, timing each kernel call.
    For Omega/doubly-bad cases the expected path/kind is None: the
    requirement is only that both kernels reject (verdict agreement),
    with paths compared separately."""
    out = []
    for case in cases:
        t0 = time.perf_counter()
        r = run_mine(case)
        dt = time.perf_counter() - t0
        r["seconds"] = dt
        out.append(r)
    return out


# ------------------------------------------------------------- generator
# Random well-typed terms, by construction, in de Bruijn. The base
# context is [Type 0, Type 0, Type 0] with free names C, B, A
# (index 0 = C).

RNG = random.Random(20261125)
BASE_CTX_DB = [Type(0), Type(0), Type(0)]
FREE_NAMES = ["C", "B", "A"]


def is_type_var(ctx, i):
    """Is Var i a type variable (its type normalizes to a universe)?"""
    try:
        ty = shift(ctx[i], i + 1)
        return normalize(ty, Meter())[0] == "type"
    except MyCheckError:
        return False


def gen_type(ctx, depth):
    opts = ["universe", "var"] + (["pi"] if depth > 0 else [])
    kind = RNG.choice(opts)
    if kind == "universe":
        return Type(RNG.choice([0, 0, 1]))
    if kind == "var":
        cands = [i for i in range(len(ctx)) if is_type_var(ctx, i)]
        if not cands:
            return Type(0)
        return Var(RNG.choice(cands))
    dom = gen_type(ctx, depth - 1)
    body = gen_type([dom] + ctx, depth - 1)
    return Pi(dom, body)


def gen_of_type(ty, ctx, depth):
    """A term of the given type (de Bruijn), or None. Uses my kernel
    only as a generation oracle (defeq / infer); the verification is
    the cross-kernel agreement on the finished terms."""
    # var: a context variable of a defeq type
    cands = []
    for i in range(len(ctx)):
        try:
            if my_defeq(ctx, shift(ctx[i], i + 1), ty, Meter()):
                cands.append(i)
        except MyCheckError:
            pass
    if cands and RNG.random() < 0.7:
        return Var(RNG.choice(cands))
    # lam: if the type normalizes to a Pi
    try:
        nf = normalize(ty, Meter())
    except MyCheckError:
        nf = None
    if nf is not None and nf[0] == "pi" and depth > 0:
        body = gen_of_type(nf[2], [nf[1]] + ctx, depth - 1)
        if body is not None:
            return Lam(nf[1], body)
    # app: a few attempts at a function + argument of matching type
    for _ in range(3):
        if depth <= 0:
            break
        f, fty = gen_term(ctx, depth - 1)
        try:
            fnf = normalize(fty, Meter())
        except MyCheckError:
            continue
        if fnf[0] != "pi":
            continue
        a = gen_of_type(fnf[1], ctx, depth - 1)
        if a is None:
            continue
        t = App(f, a)
        try:
            got = my_infer(t, ctx, (), Meter())
        except MyCheckError:
            continue
        if my_defeq(ctx, got, ty, Meter()):
            return t
    if cands:
        return Var(RNG.choice(cands))
    return None


def gen_term(ctx, depth):
    """A well-typed (term, type) pair by construction."""
    if depth <= 0 or RNG.random() < 0.4:
        i = RNG.randrange(len(ctx))
        return Var(i), shift(ctx[i], i + 1)
    if RNG.random() < 0.5:
        dom = gen_type(ctx, depth - 1)
        body, bty = gen_term([dom] + ctx, depth - 1)
        return Lam(dom, body), Pi(dom, bty)
    f, fty = gen_term(ctx, depth - 1)
    try:
        fnf = normalize(fty, Meter())
    except MyCheckError:
        return gen_term(ctx, 0)
    if fnf[0] != "pi":
        return gen_term(ctx, 0)
    a = gen_of_type(fnf[1], ctx, depth - 1)
    if a is None:
        return gen_term(ctx, 0)
    return App(f, a), subst(fnf[2], 0, a)


def free_vars(t, depth=0):
    """Free de Bruijn indices of a term (indices >= enclosing
    binders)."""
    tag = t[0]
    if tag == "var":
        return {t[1] - depth} if t[1] >= depth else set()
    if tag in ("lam", "pi"):
        return free_vars(t[1], depth) | free_vars(t[2], depth + 1)
    if tag == "app":
        return free_vars(t[1], depth) | free_vars(t[2], depth)
    return set()


def run_generator(n_pairs=200):
    pairs = []
    apps = []   # (f, a, dom, cod, ctx)
    lams = []   # (dom, body, body_ty, ctx)
    gen_rejected = []
    for _ in range(n_pairs):
        term, ty = gen_term(BASE_CTX_DB, 3)
        # record Apps and Lams encountered (top level only, for the
        # substitution checks below)
        if term[0] == "app":
            f, a = term[1], term[2]
            try:
                fty = my_infer(f, BASE_CTX_DB, (), Meter())
                fnf = normalize(fty, Meter())
                if fnf[0] == "pi":
                    apps.append((f, a, fnf[1], fnf[2], list(BASE_CTX_DB)))
            except MyCheckError:
                pass
        if term[0] == "lam":
            try:
                bty = my_infer(term[2], [term[1]] + BASE_CTX_DB, (), Meter())
                lams.append((term[1], term[2], bty, list(BASE_CTX_DB)))
            except MyCheckError:
                pass
        pairs.append((term, ty))
    # verify every generated pair with both kernels
    verified = 0
    disagreements = []
    for term, ty in pairs:
        m = Meter()
        my_fail = my_check_or_fail(BASE_CTX_DB, term, ty, m)
        named_term = untranslate(term, [], FREE_NAMES)
        named_ty = untranslate(ty, [], FREE_NAMES)
        ref_ctx = {n: R.Type(0) for n in ("A", "B", "C")}
        ref_fail = R.check_or_fail(ref_ctx, named_to_ref(named_term),
                                   named_to_ref(named_ty))
        my_ok, ref_ok = my_fail is None, ref_fail is None
        if my_ok and ref_ok:
            verified += 1
        elif not my_ok and not ref_ok:
            # both reject a term the generator claimed well-typed:
            # generator bug, not a kernel disagreement -- but record
            # it and inspect separately.
            gen_rejected.append({"term": repr(term), "type": repr(ty),
                                 "my_fail": my_fail, "ref_fail": ref_fail})
        else:
            disagreements.append({"term": repr(term), "type": repr(ty),
                                  "my_ok": my_ok, "ref_ok": ref_ok,
                                  "my_fail": my_fail, "ref_fail": ref_fail})
    return {"pairs_generated": len(pairs), "verified_both_accept": verified,
            "generator_rejected_both": len(gen_rejected),
            "gen_rejected": gen_rejected[:5],
            "disagreements": disagreements,
            "apps_collected": len(apps), "lams_collected": len(lams),
            "_apps": apps, "_lams": lams, "_pairs": pairs}


def run_substitution_checks(n_trials=100):
    """(d1) App substitution and (d2) body substitution, with trials
    generated directly (not collected from top-level terms, which
    kept the counts thin): build a random Pi type and a matching
    function/argument, substitute, and re-check in both kernels."""
    ref_ctx = {n: R.Type(0) for n in ("A", "B", "C")}
    d1_ok = d1_total = 0
    d1_fail = []
    for _ in range(n_trials):
        dom = gen_type(BASE_CTX_DB, 2)
        cod = gen_type([dom] + BASE_CTX_DB, 2)
        body = gen_of_type(cod, [dom] + BASE_CTX_DB, 2)
        if body is None:
            continue
        f = Lam(dom, body)
        a = gen_of_type(dom, BASE_CTX_DB, 2)
        if a is None:
            continue
        d1_total += 1
        expected = subst(cod, 0, a)
        try:
            got = my_infer(App(f, a), BASE_CTX_DB, (), Meter())
            my_ok = my_defeq(BASE_CTX_DB, got, expected, Meter())
        except MyCheckError:
            my_ok = False
        try:
            ref_ty, ref_fail = R.infer_or_fail(
                ref_ctx, R.App(named_to_ref(untranslate(f, [], FREE_NAMES)),
                               named_to_ref(untranslate(a, [], FREE_NAMES))))
            ref_ok = (ref_fail is None and
                      R.defeq(ref_ctx, ref_ty,
                              named_to_ref(untranslate(expected, [], FREE_NAMES))))
        except R.CheckError:
            ref_ok = False
        if my_ok and ref_ok:
            d1_ok += 1
        else:
            d1_fail.append({"f": repr(f), "a": repr(a), "my_ok": my_ok,
                            "ref_ok": ref_ok})
    d2_ok = d2_total = 0
    d2_fail = []
    for _ in range(n_trials):
        dom = gen_type(BASE_CTX_DB, 2)
        body, body_ty = gen_term([dom] + BASE_CTX_DB, 2)
        a = gen_of_type(dom, BASE_CTX_DB, 2)
        if a is None:
            continue
        d2_total += 1
        sub_body, sub_ty = subst(body, 0, a), subst(body_ty, 0, a)
        my_fail = my_check_or_fail(BASE_CTX_DB, sub_body, sub_ty, Meter())
        ref_fail = R.check_or_fail(
            ref_ctx, named_to_ref(untranslate(sub_body, [], FREE_NAMES)),
            named_to_ref(untranslate(sub_ty, [], FREE_NAMES)))
        if my_fail is None and ref_fail is None:
            d2_ok += 1
        else:
            d2_fail.append({"body": repr(body), "arg": repr(a),
                            "my_fail": my_fail, "ref_fail": ref_fail})
    return {"d1_app_substitution": {"ok": d1_ok, "total": d1_total,
                                    "failures": d1_fail[:5]},
            "d2_body_substitution": {"ok": d2_ok, "total": d2_total,
                                     "failures": d2_fail[:5]}}


def run_type_of_type(pairs):
    """(g) For every generated pair: the inferred type is itself a
    type, and its free variables stay inside the context."""
    ok = total = 0
    fails = []
    for term, ty in pairs:
        total += 1
        try:
            got = my_infer(term, BASE_CTX_DB, (), Meter())
            ty_of_ty = my_infer(got, BASE_CTX_DB, (), Meter())
            is_universe = normalize(ty_of_ty, Meter())[0] == "type"
            scope_ok = all(i < len(BASE_CTX_DB) for i in free_vars(got))
        except MyCheckError:
            is_universe, scope_ok = False, False
        if is_universe and scope_ok:
            ok += 1
        else:
            fails.append({"term": repr(term), "is_universe": is_universe,
                          "scope_ok": scope_ok})
    return {"ok": ok, "total": total, "failures": fails[:5]}


# ------------------------------------------------------------- fuzz
# Random terms (half pure noise, half mutations of generated terms),
# random expected types. Both kernels must terminate, agree on the
# verdict, and never exhaust fuel.

def random_term(depth):
    r = RNG.random()
    if depth <= 0 or r < 0.35:
        if RNG.random() < 0.5:
            return Var(RNG.randrange(3))
        return Type(RNG.choice([0, 1]))
    if r < 0.6:
        return Lam(random_term(depth - 1), random_term(depth - 1))
    if r < 0.8:
        return Pi(random_term(depth - 1), random_term(depth - 1))
    return App(random_term(depth - 1), random_term(depth - 1))


def mutate(term):
    """One random local mutation of a generated term."""
    if term[0] == "var":
        return Var(RNG.randrange(3))
    if term[0] == "type":
        return Type(RNG.choice([0, 1]))
    if term[0] in ("lam", "pi"):
        if RNG.random() < 0.5:
            return (term[0], mutate(term[1]), term[2])
        return (term[0], term[1], mutate(term[2]))
    if RNG.random() < 0.5:
        return ("app", mutate(term[1]), term[2])
    return ("app", term[1], mutate(term[2]))


def run_fuzz(pairs, n=500):
    agree = 0
    fuel_my = fuel_ref = 0
    kinds_my, kinds_ref = {}, {}
    max_seconds = 0.0
    max_beta = 0
    timeouts = []
    for i in range(n):
        if i % 2 == 0:
            term = random_term(3)
        else:
            term, _ = pairs[RNG.randrange(len(pairs))]
            term = mutate(term)
        ty = gen_type(BASE_CTX_DB, 1)
        t0 = time.perf_counter()
        m = Meter()
        my_fail = my_check_or_fail(BASE_CTX_DB, term, ty, m)
        max_beta = max(max_beta, m.beta_steps)
        named_term = untranslate(term, [], FREE_NAMES)
        named_ty = untranslate(ty, [], FREE_NAMES)
        ref_ctx = {n: R.Type(0) for n in ("A", "B", "C")}
        ref_fail = R.check_or_fail(ref_ctx, named_to_ref(named_term),
                                   named_to_ref(named_ty))
        dt = time.perf_counter() - t0
        max_seconds = max(max_seconds, dt)
        if dt > 2.0:
            timeouts.append({"i": i, "seconds": dt, "term": repr(term)})
        my_ok, ref_ok = my_fail is None, ref_fail is None
        if my_ok == ref_ok:
            agree += 1
        if my_fail is not None:
            kinds_my[my_fail["kind"]] = kinds_my.get(my_fail["kind"], 0) + 1
            if my_fail["kind"] == "fuel":
                fuel_my += 1
        if ref_fail is not None:
            kinds_ref[ref_fail["kind"]] = kinds_ref.get(ref_fail["kind"], 0) + 1
            if ref_fail["kind"] == "fuel":
                fuel_ref += 1
    return {"trials": n, "verdict_agreement": agree,
            "fuel_exhaustions_mine": fuel_my,
            "fuel_exhaustions_ref": fuel_ref,
            "failure_kinds_mine": kinds_my,
            "failure_kinds_ref": kinds_ref,
            "max_seconds_per_case": max_seconds,
            "max_beta_steps_mine": max_beta,
            "slow_cases": timeouts}


def run_unbound():
    """Unbound variables, tested directly on each kernel (a free
    variable has no derivation under the Var rule, so both must
    reject -- but the rules do not pin the failure shape here, so
    record exactly what each kernel does)."""
    out = {}
    # Mine: Var(5) in a 3-entry context.
    _, fail = my_infer_or_fail(BASE_CTX_DB, Var(5), Meter())
    out["mine"] = {"term": "Var(5) in 3-entry ctx", "fail": fail}
    # Reference: a name with no declaration.
    try:
        _, fail = R.infer_or_fail({"A": R.Type(0)}, R.Var("zz"))
        out["ref"] = {"term": "Var('zz') not in ctx", "fail": fail}
    except Exception as e:  # noqa: BLE001 -- record whatever it does
        out["ref"] = {"term": "Var('zz') not in ctx",
                      "fail": {"kind": "raised-%s" % type(e).__name__,
                               "message": str(e)}}
    return out


if __name__ == "__main__":
    battery = {}
    unbound = run_unbound()
    battery["unbound_variable"] = unbound
    print("unbound: mine=%s" % (unbound["mine"]["fail"],))
    print("unbound: ref =%s" % (unbound["ref"]["fail"],))
    for section, cases in [("capture", CAPTURE), ("eta", ETA),
                           ("universe", UNIVERSE)]:
        res = run_named_battery(cases)
        ok = sum(1 for r in res if r["agree"] and r["path_agree"])
        battery[section] = {
            "total": len(res),
            "verdict_and_path_agreement": "%d/%d" % (ok, len(res)),
            "cases": res}
        for r in res:
            print("%s[%s] %s: ref=%r mine=%r path_agree=%s (%.3fs)" %
                  ("PASS " if r["agree"] and r["path_agree"] else "FAIL ",
                   section, r["name"], r["ref_verdict"], r["my_verdict"],
                   r["path_agree"], r["seconds"]))
    # Omega + doubly-bad: verdict agreement required; paths are
    # EXPECTED to differ (arg-first vs fn-first) -- report both.
    for section, cases in [("fuel_omega", OMEGA), ("doubly_bad", DOUBLY_BAD)]:
        res = run_named_battery(cases)
        agree = sum(1 for r in res if r["agree"])
        battery[section] = {
            "total": len(res), "verdict_agreement": "%d/%d" % (agree, len(res)),
            "cases": [{k: v for k, v in r.items()} for r in res]}
        for r in res:
            print("%s[%s] %s: ref_verdict=%r mine_verdict=%r "
                  "ref_path=%s my_path=%s ref_kind=%s my_kind=%s" %
                  ("PASS " if r["agree"] else "FAIL ", section, r["name"],
                   r["ref_verdict"], r["my_verdict"],
                   (r["ref_fail"] or {}).get("path"),
                   (r["my_fail"] or {}).get("path"),
                   (r["ref_fail"] or {}).get("kind"),
                   (r["my_fail"] or {}).get("kind")))
    gen = run_generator(200)
    print("\ngenerator: %d pairs, both accept %d, both reject (generator "
          "bug) %d, disagreements %d" %
          (gen["pairs_generated"], gen["verified_both_accept"],
           gen["generator_rejected_both"], len(gen["disagreements"])))
    if gen["disagreements"]:
        for d in gen["disagreements"][:5]:
            print("  DISAGREEMENT: %s" % (d,))
    if gen["generator_rejected_both"]:
        for d in gen["gen_rejected"]:
            print("  GENERATOR-BUG (both reject): %s" % (d,))
    sub = run_substitution_checks(300)
    print("substitution d1 (App): %d/%d" %
          (sub["d1_app_substitution"]["ok"], sub["d1_app_substitution"]["total"]))
    print("substitution d2 (body): %d/%d" %
          (sub["d2_body_substitution"]["ok"], sub["d2_body_substitution"]["total"]))
    for f in sub["d1_app_substitution"]["failures"] + sub["d2_body_substitution"]["failures"]:
        print("  SUBST-FAIL: %s" % (f,))
    tot = run_type_of_type(gen["_pairs"])
    print("type-of-type + scope: %d/%d" % (tot["ok"], tot["total"]))
    for f in tot["failures"]:
        print("  TOT-FAIL: %s" % (f,))
    fz = run_fuzz(gen["_pairs"], 500)
    print("fuzz: verdict agreement %d/%d, fuel mine=%d ref=%d, "
          "max %.3fs/case, max beta %d" %
          (fz["verdict_agreement"], fz["trials"], fz["fuel_exhaustions_mine"],
           fz["fuel_exhaustions_ref"], fz["max_seconds_per_case"],
           fz["max_beta_steps_mine"]))
    print("  failure kinds mine: %s" % (fz["failure_kinds_mine"],))
    print("  failure kinds ref: %s" % (fz["failure_kinds_ref"],))
    for t in fz["slow_cases"]:
        print("  SLOW: %s" % (t,))
    gen_out = {k: v for k, v in gen.items() if not k.startswith("_")}
    battery["generator"] = gen_out
    battery["substitution_lemma"] = sub
    battery["type_of_type_and_scope"] = tot
    battery["fuzz"] = fz
    with open("battery_dt_result.json", "w") as fh:
        json.dump(battery, fh, indent=2, default=str)
    print("\nwrote battery_dt_result.json")
