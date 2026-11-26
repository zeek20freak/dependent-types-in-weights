"""Examples for the v0.1 dependent kernel (rung 1) — a playground file.

Run it:  python3 examples_dt.py

For a live version, run python3 repl_dt.py — type a term, get its
type back, or a term and a type, and get the verdict on the spot.

What this is: kernel_dt.py is a minimal dependent type checker — five
rules (Var, Type, Lam, Pi, App), Pi types + universes + conversion,
about four hundred lines, no dependencies. A term is a proof, a type
is a proposition, and the checker is the judge: you hand it a term and
a type, it says yes or no, and when it says no it tells you exactly
where — the first failure as a path, like ("arg",) or ("body", "func").

How to play with it:
  * Read the three sections. Each example prints its verdict.
  * Change one thing in any example and re-run — watch the verdict
    flip. Section 3 is built for exactly that: four passing proofs,
    each broken in one place.
  * Add your own: copy one example() line, change the term or the
    type, and say what you expect — ("accept",) or
    ("reject", path, kind). The file checks every expectation and
    exits non-zero if any verdict is not what it claims, so a wrong
    guess gets caught instead of silently standing.

Terms are Python objects, not text: Lam("x", A, Var("x")) is the
function that returns its argument. A, B, C are declared base types
(context variables of type Type 0); the kernel has no built-in bases.
See RULES_DT.md for the five rules, kernel_dt.py for the checker,
and sealing-pass-result.json for the evidence that a second,
independent kernel agrees with this one on every case (43/43).
"""

from kernel_dt import (Var, Lam, App, Pi, Type, check_or_fail,
                       infer_or_fail, defeq, CheckError)

A, B, C = Var("A"), Var("B"), Var("C")
T0, T1 = Type(0), Type(1)

# Base types are context variables of type Type 0 here — declared,
# not given.
ctx0 = {"A": T0, "B": T0, "C": T0}


def arr(t, u):
    """Non-dependent arrow, as a Pi with an unused binder."""
    return Pi("_", t, u)


# --- The runner -----------------------------------------------------------
# expect is ("accept",), ("reject",), or ("reject", path, kind).
# Every verdict is checked against its expectation; a mismatch is
# printed and counted, and the run exits non-zero at the end.

_results = []


def example(name, term, type_, ctx, expect, note=""):
    fail = check_or_fail(ctx, term, type_)
    if expect[0] == "accept":
        ok = fail is None
        got = "ACCEPT" if ok else f"REJECT at {fail['path']}, {fail['kind']}"
        want = "ACCEPT"
    elif len(expect) == 1:  # ("reject",) — any failure counts
        ok = fail is not None
        got = (f"REJECT at {fail['path']}, {fail['kind']}"
               if fail else "ACCEPT")
        want = "REJECT"
    else:  # ("reject", path, kind)
        _, path, kind = expect
        ok = (fail is not None and tuple(fail["path"]) == tuple(path)
              and fail["kind"] == kind)
        got = (f"REJECT at {fail['path']}, {fail['kind']}"
               if fail else "ACCEPT")
        want = f"REJECT at {list(path)}, {kind}"
    mark = "ok  " if ok else "MISMATCH"
    print(f"[{mark}] {name}\n      {term} : {type_}\n      -> {got}"
          + ("" if ok else f"   (expected {want})"))
    if note:
        print(f"      {note}")
    _results.append(ok)


def infer_example(name, term, ctx, expect, note=""):
    _, fail = infer_or_fail(ctx, term)
    _, path, kind = expect
    ok = (fail is not None and tuple(fail["path"]) == tuple(path)
          and fail["kind"] == kind)
    got = (f"REJECT at {fail['path']}, {fail['kind']}"
           if fail else "ACCEPT")
    mark = "ok  " if ok else "MISMATCH"
    print(f"[{mark}] {name}\n      {term}\n      -> {got}"
          + ("" if ok else f"   (expected REJECT at {list(path)}, {kind})"))
    if note:
        print(f"      {note}")
    _results.append(ok)


def defeq_example(name, ctx, t, u, want, note=""):
    try:
        got = defeq(ctx, t, u)
    except CheckError as e:
        print(f"[MISMATCH] {name}\n      defeq raised {e}")
        _results.append(False)
        return
    ok = got == want
    mark = "ok  " if ok else "MISMATCH"
    print(f"[{mark}] {name}\n      {t}  ==  {u}\n      -> defeq = {got}")
    if note:
        print(f"      {note}")
    _results.append(ok)


def section(title):
    print(f"\n--- {title} ---")


# --- Shared pieces ---------------------------------------------------------

id_dep = Lam("A", T0, Lam("x", Var("A"), Var("x")))
const_dep = Lam("A", T0,
                Lam("B", T0,
                    Lam("x", Var("A"),
                        Lam("y", Var("B"), Var("x")))))
# A type-level function: it computes a type from a term.
type_fn = Lam("A", T0, arr(Var("A"), Var("A")))
S = Lam("f", arr(A, arr(B, C)),
        Lam("g", arr(A, B),
            Lam("x", A, App(App(Var("f"), Var("x")),
                            App(Var("g"), Var("x"))))))
S_type = arr(arr(A, arr(B, C)), arr(arr(A, B), arr(A, C)))


def main():
    section("1. Proofs that pass — the checker says yes")

    example("dependent identity: the type mentions the bound variable",
            id_dep, Pi("A", T0, Pi("x", Var("A"), Var("A"))), ctx0,
            ("accept",),
            "λA:Type 0. λx:A. x — the identity, but the type depends on A.")
    example("dependent const",
            const_dep,
            Pi("A", T0, Pi("B", T0,
                           Pi("x", Var("A"),
                              Pi("y", Var("B"), Var("A"))))), ctx0,
            ("accept",),
            "Returns its first argument and ignores the second.")
    example("a type-level function: it computes a type from a term",
            type_fn, Pi("A", T0, T0), ctx0,
            ("accept",),
            "λA:Type 0. A -> A — the first genuinely dependent object: "
            "a function whose codomain is a universe.")
    example("conversion: the expected type is a beta-redex",
            Lam("x", B, Var("x")), App(type_fn, B), ctx0,
            ("accept",),
            "The checker reduces (λA:Type 0. A -> A) B to B -> B and "
            "sees the identity fits.")
    example("S, the combinator, with A, B, C as universe variables",
            S, S_type, ctx0,
            ("accept",),
            "λf. λg. λx. f x (g x) — the STLC theorem, living in the "
            "dependent kernel too.")
    example("applied dependent identity: (id_dep A) x : A",
            App(App(id_dep, A), Var("x")), A, {**ctx0, "x": A},
            ("accept",),
            "Instantiate the polymorphic identity at A, then apply it.")
    example("capture, handled: the inner Lam shadows an outer variable",
            App(Lam("A", T0, Lam("x", A, Lam("A", T0, Var("x")))), B),
            Pi("x", B, Pi("A2", T0, B)), ctx0,
            ("accept",),
            "The sealing pass caught this one: the inner Lam's body "
            "type mentions an outer A sharing the binder's name. The "
            "fixed kernel reads it as the outer A. (Before the fix, "
            "this rejected.)")
    ctx_fam = {**ctx0, "F": Pi("y", A, T0),
               "f": Pi("y", A, App(Var("F"), Var("y")))}
    example("a type family applied to the binder itself",
            Lam("x", A, App(Var("f"), Var("x"))),
            Pi("x", A, App(Var("F"), Var("x"))), ctx_fam,
            ("accept",),
            "F is a family over A and f : Πy:A. F y. The body's type "
            "mentions the binder itself — so the binder name must "
            "stay. This is the case that proved the first proposed "
            "fix wrong.")

    section("2. Terms that fail — the checker says no, and says where")

    example("Girard: Type 0 is not its own type",
            T0, T0, ctx0,
            ("reject", (), "mismatch"),
            "The hierarchy is strict — Type 0 lives in Type 1, never "
            "in itself. This is what keeps the paradox out.")
    example("dependent application, wrong argument",
            App(App(id_dep, A), Var("y")), A, {**ctx0, "y": B},
            ("reject", ("arg",), "mismatch"),
            "(id_dep A) expects an A; y is a B. The failure is in the "
            "argument position — the path says so.")
    infer_example("a binder annotation that is not a type",
                  Lam("x", Var("f"), Var("x")),
                  {**ctx0, "f": arr(A, A)},
                  ("reject", ("binder",), "not-a-type"),
                  "f : A -> A is a function, not a universe, so λx:f. x "
                  "has no well-formed binder.")
    example("type-level self-application",
            Lam("A", T0, App(Var("A"), Var("A"))),
            Pi("A", T0, T0), ctx0,
            ("reject", ("body", "func"), "not-pi"),
            "A : Type 0 is a type, not a function — it can't be "
            "applied to itself.")
    example("wrong universe: the polymorphic identity type is Type 1",
            Pi("A", T0, arr(Var("A"), Var("A"))), T0, ctx0,
            ("reject", (), "mismatch"),
            "Its domain is a universe, so the whole Pi lives one level "
            "up.")

    section("3. Break it yourself — one change each, watch the verdict flip")

    example("identity, wrong codomain: ... Π x:A. B instead of A",
            id_dep, Pi("A", T0, Pi("x", Var("A"), Var("B"))), ctx0,
            ("reject",),
            "TRY: put Var('A') back as the last codomain and re-run — "
            "it accepts again.")
    example("S with f and g swapped in the body",
            Lam("f", arr(A, arr(B, C)),
                Lam("g", arr(A, B),
                    Lam("x", A, App(App(Var("g"), Var("x")),
                                    App(Var("f"), Var("x")))))),
            S_type, ctx0,
            ("reject",),
            "TRY: swap them back. The order is the proof — g's result "
            "is B, and B is not a function.")
    example("conversion, wrong argument type: λx:C. x against ... B",
            Lam("x", C, Var("x")), App(type_fn, B), ctx0,
            ("reject",),
            "The expected type reduces to B -> B; the term gives "
            "C -> C. TRY: change the binder to B.")
    example("applied identity, x : B instead of A",
            App(App(id_dep, A), Var("x")), A, {**ctx0, "x": B},
            ("reject", ("arg",), "mismatch"),
            "Same shape as the section-2 failure — the path points at "
            "the argument again. TRY: give x the type A.")

    section("4. Definitional equality, direct")

    ctx_eta = {**ctx0, "f": arr(A, B), "g": arr(A, B)}
    defeq_example("eta: f is the same as λx:A. f x",
                  ctx_eta, Var("f"),
                  Lam("x", A, App(Var("f"), Var("x"))), True,
                  "Eta-for-functions: a function equals its expansion.")
    defeq_example("distinct variables are not equal: f is not g",
                  ctx_eta, Var("f"), Var("g"), False,
                  "Two different names in the context stay different.")
    defeq_example("alpha: λx:A. x is the same as λy:A. y",
                  ctx0, Lam("x", A, Var("x")),
                  Lam("y", A, Var("y")), True,
                  "Bound names don't matter — only the shape does.")

    print(f"\n{sum(_results)}/{len(_results)} examples as expected")
    if not all(_results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
