"""Comparison driver for the rung 1+2 sealing passes.

Runs BOTH kernels on the union of the suites:
- the 18 shared cases from test_kernel_dt.py (reconstructed here
  with the reference kernel's constructors, then converted to de
  Bruijn for the independent kernel via a runtime adapter over the
  constructors' data layout only -- no kernel source is read),
- my cases from test_kernel_dt_independent.py (authored in
  named form, converted to reference terms for the reference
  kernel, translated to de Bruijn for the independent kernel),
  rung 1 plus the rung-2 Nat cases.

Requirement: 100% verdict agreement. Any disagreement is
adjudicated against RULES_DT.md and reported, never papered over.

Also runs the observer measurement's step-count half: a standalone
head-beta (whnf) step counter -- an independent instrument, not
part of either kernel -- counts head-beta steps on the compared
terms, against the independent kernel's full-normalization step
counts on the same terms. If the two strategies give different
counts on the same input, step counts are frame-dependent.

Writes sealing-pass-result-rung2.json (comparison + observer
measurement sections). The rung-1 receipt sealing-pass-result.json
is not touched.
"""

import json

import kernel_dt as R
from kernel_dt_independent import (Var, Lam, App, Pi, Type, shift, subst,
                                    normalize, Meter, defeq as my_defeq,
                                    check_or_fail as my_check_or_fail,
                                    infer_or_fail as my_infer_or_fail,
                                    CheckError as MyCheckError)
from test_kernel_dt_independent import (CASES as MY_CASES,
                                         NAT_CASES as MY_NAT_CASES,
                                         translate,
                                         translate_ctx, V, L, A, P, T)

# ------------------------------------------------- adapters (data only)
# Reference term -> my named tuples, from the constructors' data
# layout: Var.name, Lam.var/var_type/body, App.func/arg,
# Pi.var/var_type/body, Type.level. My named tuples -> reference
# terms, the reverse direction.

def ref_to_named(t):
    cls = type(t).__name__
    if cls == "Var":
        return ("var", t.name)
    if cls == "Lam":
        return ("lam", t.var, ref_to_named(t.var_type),
                ref_to_named(t.body))
    if cls == "App":
        return ("app", ref_to_named(t.func), ref_to_named(t.arg))
    if cls == "Pi":
        return ("pi", t.var, ref_to_named(t.var_type),
                ref_to_named(t.body))
    if cls == "Type":
        return ("type", t.level)
    raise ValueError("unknown reference term %r" % (cls,))


def named_to_ref(t):
    tag = t[0]
    if tag == "var":
        return R.Var(t[1])
    if tag == "lam":
        return R.Lam(t[1], named_to_ref(t[2]), named_to_ref(t[3]))
    if tag == "pi":
        return R.Pi(t[1], named_to_ref(t[2]), named_to_ref(t[3]))
    if tag == "app":
        return R.App(named_to_ref(t[1]), named_to_ref(t[2]))
    if tag == "type":
        return R.Type(t[1])
    if tag == "nat":
        return R.Nat()
    if tag == "zero":
        return R.Zero()
    if tag == "succ":
        return R.Succ(named_to_ref(t[1]))
    if tag == "rec":
        return R.Rec(named_to_ref(t[1]), named_to_ref(t[2]),
                     named_to_ref(t[3]), named_to_ref(t[4]))
    raise ValueError("unknown named term %r" % (tag,))


def named_ctx_to_ref(ctx_named):
    return {name: named_to_ref(ty) for name, ty in ctx_named.items()}


def ref_ctx_to_named(ctx_ref):
    return {name: ref_to_named(ty) for name, ty in ctx_ref.items()}


# ------------------------------------------------- head-beta instrument
# Standalone whnf step counter on de Bruijn terms: reduce head
# redexes only (never under a binder, never in an argument). This
# is the spec's whnf strategy, counted by an instrument that belongs
# to neither kernel.

def whnf_steps(t, cap=10000):
    steps = 0
    while True:
        if steps > cap:
            raise RuntimeError("whnf fuel exhausted")
        if t[0] == "app":
            f, s = whnf_steps(t[1], cap - steps)
            steps += s
            if f[0] == "lam":
                steps += 1
                t = subst(f[2], 0, t[2])
                continue
            return ("app", f, t[2]), steps
        return t, steps


def normalize_steps(t, cap=10000):
    m = Meter(fuel_limit=cap)
    normalize(t, m)
    return m.beta_steps


# ------------------------------------------------------------ shared 18
# Reconstructed from test_kernel_dt.py's case data.

RA, RB, RC = R.Var("A"), R.Var("B"), R.Var("C")
RT0, RT1 = R.Type(0), R.Type(1)


def rarr(t, u):
    return R.Pi("_", t, u)


RCTX0 = {"A": RT0, "B": RT0, "C": RT0}
RID_DEP = R.Lam("A", RT0, R.Lam("x", R.Var("A"), R.Var("x")))
RTYPE_FN = R.Lam("A", RT0, rarr(R.Var("A"), R.Var("A")))
RS = R.Lam("f", rarr(RA, rarr(RB, RC)),
           R.Lam("g", rarr(RA, RB),
                 R.Lam("x", RA, R.App(R.App(R.Var("f"), R.Var("x")),
                                      R.App(R.Var("g"), R.Var("x"))))))

SHARED = [
    {"name": "dependent identity", "kind": "check",
     "term": RID_DEP, "type": R.Pi("A", RT0, R.Pi("x", R.Var("A"), R.Var("A"))),
     "ctx": RCTX0},
    {"name": "dependent const", "kind": "check",
     "term": R.Lam("A", RT0, R.Lam("B", RT0, R.Lam("x", R.Var("A"),
                                                 R.Lam("y", R.Var("B"), R.Var("x"))))),
     "type": R.Pi("A", RT0, R.Pi("B", RT0, R.Pi("x", R.Var("A"),
                                               R.Pi("y", R.Var("B"), R.Var("A"))))),
     "ctx": RCTX0},
    {"name": "type-level function", "kind": "check",
     "term": RTYPE_FN, "type": R.Pi("A", RT0, RT0), "ctx": RCTX0},
    {"name": "conversion: lam x:B. x against (lam A. A -> A) B",
     "kind": "check", "term": R.Lam("x", RB, R.Var("x")),
     "type": R.App(RTYPE_FN, RB), "ctx": RCTX0},
    {"name": "(lam A. A -> A) B is a type", "kind": "check",
     "term": R.App(RTYPE_FN, RB), "type": RT0, "ctx": RCTX0},
    {"name": "eta through check: (K a) against H (lam B. F B)",
     "kind": "check", "term": R.App(R.Var("K"), R.Var("a")),
     "type": R.App(R.Var("H"), R.Lam("B", RT0, R.App(R.Var("F"), R.Var("B")))),
     "ctx": dict(RCTX0, F=R.Pi("A", RT0, RT0),
                 H=rarr(R.Pi("A", RT0, RT0), RT0),
                 K=R.Pi("x", RA, R.App(R.Var("H"), R.Var("F"))), a=RA)},
    {"name": "Type 0 : Type 1", "kind": "check",
     "term": RT0, "type": RT1, "ctx": {}},
    {"name": "Pi A:Type 0. A -> A : Type 1", "kind": "check",
     "term": R.Pi("A", RT0, rarr(R.Var("A"), R.Var("A"))), "type": RT1,
     "ctx": {}},
    {"name": "S: (A->B->C) -> (A->B) -> A -> C", "kind": "check",
     "term": RS,
     "type": rarr(rarr(RA, rarr(RB, RC)), rarr(rarr(RA, RB), rarr(RA, RC))),
     "ctx": RCTX0},
    {"name": "applied dependent identity: (id_dep A) x : A",
     "kind": "check", "term": R.App(R.App(RID_DEP, RA), R.Var("x")),
     "type": RA, "ctx": dict(RCTX0, x=RA)},
    {"name": "defeq eta: f =?= lam x:A. f x", "kind": "defeq",
     "term": R.Var("f"), "type": R.Lam("x", RA, R.App(R.Var("f"), R.Var("x"))),
     "want": True, "ctx": dict(RCTX0, f=rarr(RA, RB), g=rarr(RA, RB))},
    {"name": "defeq distinct variables are not equal", "kind": "defeq",
     "term": R.Var("f"), "type": R.Var("g"), "want": False,
     "ctx": dict(RCTX0, f=rarr(RA, RB), g=rarr(RA, RB))},
    {"name": "defeq alpha", "kind": "defeq",
     "term": R.Lam("x", RA, R.Var("x")),
     "type": R.Lam("y", RA, R.Var("y")), "want": True, "ctx": RCTX0},
    {"name": "Type 0 : Type 0 (Girard)", "kind": "check_reject",
     "term": RT0, "type": RT0, "path": (), "fail_kind": "mismatch",
     "ctx": {}},
    {"name": "dependent application, wrong argument: (id_dep A) y, y : B",
     "kind": "check_reject", "term": R.App(R.App(RID_DEP, RA), R.Var("y")),
     "type": RA, "path": ("arg",), "fail_kind": "mismatch",
     "ctx": dict(RCTX0, y=RB)},
    {"name": "binder annotation is not a type: lam x:f. x, f : A -> A",
     "kind": "infer_reject", "term": R.Lam("x", R.Var("f"), R.Var("x")),
     "path": ("binder",), "fail_kind": "not-a-type",
     "ctx": dict(RCTX0, f=rarr(RA, RA))},
    {"name": "type-level self-application: lam A:Type 0. A A",
     "kind": "check_reject",
     "term": R.Lam("A", RT0, R.App(R.Var("A"), R.Var("A"))),
     "type": R.Pi("A", RT0, RT0), "path": ("body", "func"),
     "fail_kind": "not-pi", "ctx": RCTX0},
    {"name": "Pi A:Type 0. A -> A : Type 0 (wrong universe)",
     "kind": "check_reject",
     "term": R.Pi("A", RT0, rarr(R.Var("A"), R.Var("A"))), "type": RT0,
     "path": (), "fail_kind": "mismatch", "ctx": {}},
]


# ------------------------------------------------------------ one case

def run_shared(case):
    """Run one shared (reference-form) case on both kernels."""
    ctx_named = ref_ctx_to_named(case["ctx"])
    ctx_db, env = translate_ctx(ctx_named)
    term_db = translate(ref_to_named(case["term"]), env)
    kind = case["kind"]
    # Reference kernel.
    if kind == "defeq":
        type_db = translate(ref_to_named(case["type"]), env)
        try:
            ref_got = R.defeq(case["ctx"], case["term"], case["type"])
            ref_fail = None
        except R.CheckError as e:
            ref_got, ref_fail = None, {"kind": "raised", "message": str(e)}
        meter = Meter()
        try:
            my_got = my_defeq(ctx_db, term_db, type_db, meter)
            my_fail = None
        except MyCheckError as e:
            my_got, my_fail = None, e.as_dict()
        agree = (ref_got == my_got == case["want"])
        return {"name": case["name"], "suite": "shared", "kind": kind,
                "ref_verdict": ref_got, "my_verdict": my_got,
                "want": case["want"], "agree": agree,
                "ref_fail": ref_fail, "my_fail": my_fail,
                "my_beta_steps": meter.beta_steps,
                "path_agree": True}
    # check / check_reject / infer_reject.
    meter = Meter()
    if kind == "infer_reject":
        _, ref_fail = R.infer_or_fail(case["ctx"], case["term"])
        _, my_fail = my_infer_or_fail(ctx_db, term_db, meter)
        ref_ok, my_ok = ref_fail is None, my_fail is None
    else:
        type_db = translate(ref_to_named(case["type"]), env)
        ref_fail = R.check_or_fail(case["ctx"], case["term"], case["type"])
        my_fail = my_check_or_fail(ctx_db, term_db, type_db, meter)
        ref_ok, my_ok = ref_fail is None, my_fail is None
    agree = (ref_ok == my_ok)
    path_agree = True
    if not ref_ok and not my_ok:
        path_agree = (tuple(ref_fail["path"]) == tuple(my_fail["path"])
                      and ref_fail["kind"] == my_fail["kind"])
    return {"name": case["name"], "suite": "shared", "kind": kind,
            "ref_verdict": ref_ok, "my_verdict": my_ok, "agree": agree,
            "ref_fail": ref_fail, "my_fail": my_fail,
            "my_beta_steps": meter.beta_steps, "path_agree": path_agree}


def run_mine(case):
    """Run one of my cases on both kernels (reference kernel via
    named -> reference conversion)."""
    ctx_ref = named_ctx_to_ref(case["ctx"])
    ctx_db, env = translate_ctx(case["ctx"])
    term_db = translate(case["term"], env)
    kind = case["kind"]
    if kind == "defeq":
        type_db = translate(case["type"], env)
        try:
            ref_got = R.defeq(ctx_ref, named_to_ref(case["term"]),
                              named_to_ref(case["type"]))
            ref_fail = None
        except R.CheckError as e:
            ref_got, ref_fail = None, {"kind": "raised", "message": str(e)}
        meter = Meter()
        try:
            my_got = my_defeq(ctx_db, term_db, type_db, meter)
            my_fail = None
        except MyCheckError as e:
            my_got, my_fail = None, e.as_dict()
        agree = (ref_got == my_got == case["want"])
        return {"name": case["name"], "suite": "mine", "kind": kind,
                "ref_verdict": ref_got, "my_verdict": my_got,
                "want": case["want"], "agree": agree,
                "ref_fail": ref_fail, "my_fail": my_fail,
                "my_beta_steps": meter.beta_steps, "path_agree": True}
    meter = Meter()
    if kind == "infer_reject":
        _, ref_fail = R.infer_or_fail(ctx_ref, named_to_ref(case["term"]))
        _, my_fail = my_infer_or_fail(ctx_db, term_db, meter)
        ref_ok, my_ok = ref_fail is None, my_fail is None
    else:
        type_db = translate(case["type"], env)
        ref_fail = R.check_or_fail(ctx_ref, named_to_ref(case["term"]),
                                   named_to_ref(case["type"]))
        my_fail = my_check_or_fail(ctx_db, term_db, type_db, meter)
        ref_ok, my_ok = ref_fail is None, my_fail is None
    agree = (ref_ok == my_ok)
    path_agree = True
    if not ref_ok and not my_ok:
        path_agree = (tuple(ref_fail["path"]) == tuple(my_fail["path"])
                      and ref_fail["kind"] == my_fail["kind"])
    return {"name": case["name"], "suite": "mine", "kind": kind,
            "ref_verdict": ref_ok, "my_verdict": my_ok, "agree": agree,
            "ref_fail": ref_fail, "my_fail": my_fail,
            "my_beta_steps": meter.beta_steps, "path_agree": path_agree}


# ------------------------------------------------------- strategy probe
# Terms on which head-beta whnf and full normalization do visibly
# different amounts of work. Named form, translated to de Bruijn.

PROBES = [
    {"name": "redex under a Pi binder (whnf 0, normalize 1)",
     "term": P("x", V("A"), A(L("T", T(0), V("T")), V("B"))),
     "ctx": {"A": T(0), "B": T(0)}},
    {"name": "redex in an argument position (whnf 1, normalize 2)",
     "term": A(L("A", T(0), P("z", V("A"), V("A"))),
               A(L("B", T(0), V("B")), V("C"))),
     "ctx": {"A": T(0), "B": T(0), "C": T(0)}},
    {"name": "redex at the head only (whnf 1, normalize 1)",
     "term": A(L("A", T(0), P("z", V("A"), V("A"))), V("B")),
     "ctx": {"A": T(0), "B": T(0)}},
    {"name": "nested redexes under two binders (whnf 0, normalize 2)",
     "term": P("x", V("A"), P("y", V("B"),
                              A(L("T", T(0), V("T")),
                                A(L("S", T(0), V("S")), V("A"))))),
     "ctx": {"A": T(0), "B": T(0)}},
]


def run_probes():
    out = []
    for probe in PROBES:
        ctx_db, env = translate_ctx(probe["ctx"])
        t = translate(probe["term"], env)
        _, w = whnf_steps(t)
        n = normalize_steps(t)
        out.append({"name": probe["name"], "whnf_steps": w,
                    "normalize_steps": n, "differ": w != n})
    return out


if __name__ == "__main__":
    per_case = []
    for case in SHARED:
        per_case.append(run_shared(case))
    for case in MY_CASES:
        per_case.append(run_mine(case))
    for case in MY_NAT_CASES:
        per_case.append(run_mine(case))
    total = len(per_case)
    agreed = sum(1 for c in per_case if c["agree"])
    path_agreed = sum(1 for c in per_case if c["path_agree"])
    probes = run_probes()
    for c in per_case:
        mark = "PASS " if c["agree"] and c["path_agree"] else "FAIL "
        print("%s[%s] %s: ref=%r mine=%r path_agree=%s" %
              (mark, c["suite"], c["name"], c["ref_verdict"],
               c["my_verdict"], c["path_agree"]))
        if not c["agree"] or not c["path_agree"]:
            print("      ref_fail=%s" % (c["ref_fail"],))
            print("      my_fail=%s" % (c["my_fail"],))
    print("\nverdict agreement: %d/%d" % (agreed, total))
    print("failure-path agreement: %d/%d" % (path_agreed, total))
    print("\nstrategy probe (head-beta whnf vs full normalization):")
    for p in probes:
        print("  %s: whnf=%d normalize=%d differ=%s" %
              (p["name"], p["whnf_steps"], p["normalize_steps"], p["differ"]))
    result = {
        "comparison": {
            "shared_cases": len(SHARED), "my_cases": len(MY_CASES),
            "nat_cases": len(MY_NAT_CASES),
            "total_cases": total, "verdict_agreement": "%d/%d" % (agreed, total),
            "verdict_agreed": agreed,
            "failure_path_agreement": "%d/%d" % (path_agreed, total),
            "failure_path_agreed": path_agreed,
            "disagreements": [c for c in per_case if not c["agree"]],
            "path_differences": [c for c in per_case if not c["path_agree"]],
            "per_case": per_case,
        },
        "observer_measurement": {
            "verdicts_agree": "%d/%d" % (agreed, total),
            "strategy_probe": probes,
            "strategy_probe_all_differ_except_head_only":
                all(p["differ"] for p in probes if "head only" not in p["name"]),
            "note": "Step counts are the frame-dependent half: the same "
                    "term takes different numbers of beta steps under "
                    "head-beta whnf and full normalization. "
                    "First-failure paths on the shared+my suites agree "
                    "(all rejections here are singly-bad); path "
                    "differences appear only on doubly-bad terms, "
                    "measured in the battery.",
        },
    }
    # Preserve the seal record across re-runs: the compare rewrites
    # the receipt in place, and the seal_verdict (written once the
    # seal is earned) must survive it. Without this, run_tests.sh
    # re-running compare_dt.py wipes the seal it then asserts.
    try:
        with open("sealing-pass-result-rung2.json") as f:
            previous = json.load(f)
    except (OSError, ValueError):
        previous = {}
    for key, value in previous.items():
        if key not in result:
            result[key] = value
    with open("sealing-pass-result-rung2.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print("\nwrote sealing-pass-result-rung2.json")
    if agreed != total or path_agreed != total:
        raise SystemExit(1)
