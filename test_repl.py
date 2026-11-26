"""Tests for the REPL parser (repl_dt.py).

The parser is a front end, not part of the sealed kernel — so it is
verified against the kernel here: every example from examples_dt.py
is written in surface syntax, parsed, and its verdict is checked
against the same expectation the example carries. A parser bug that
changed a term would flip a verdict and fail loudly here.

Run it:  python3 test_repl.py
"""

from kernel_dt import Var, App, Pi, Type, check_or_fail, infer_or_fail, defeq
from repl_dt import parse_term_source, ParseError

A, B, C = Var("A"), Var("B"), Var("C")
T0, T1 = Type(0), Type(1)
ctx0 = {"A": T0, "B": T0, "C": T0}
ctx_eta = {**ctx0, "f": Pi("_", A, B), "g": Pi("_", A, B)}
ctx_fam = {**ctx0, "F": Pi("y", A, T0),
           "f": Pi("y", A, App(Var("F"), Var("y")))}

results = []


def expect_check(name, term_src, type_src, ctx, expect):
    try:
        term = parse_term_source(term_src)
        typ = parse_term_source(type_src)
    except ParseError as e:
        results.append((False, f"FAIL  {name}: parse error {e}"))
        return
    fail = check_or_fail(ctx, term, typ)
    if expect[0] == "accept":
        ok = fail is None
        got = "ACCEPT" if ok else f"REJECT at {fail['path']}, {fail['kind']}"
    elif len(expect) == 1:
        ok = fail is not None
        got = (f"REJECT at {fail['path']}, {fail['kind']}"
               if fail else "ACCEPT")
    else:
        _, path, kind = expect
        ok = (fail is not None and tuple(fail["path"]) == tuple(path)
              and fail["kind"] == kind)
        got = (f"REJECT at {fail['path']}, {fail['kind']}"
               if fail else "ACCEPT")
    results.append((ok, f"{'PASS' if ok else 'FAIL'}  {name} ({got})"))


def expect_infer(name, term_src, ctx, expect):
    try:
        term = parse_term_source(term_src)
    except ParseError as e:
        results.append((False, f"FAIL  {name}: parse error {e}"))
        return
    _, fail = infer_or_fail(ctx, term)
    _, path, kind = expect
    ok = (fail is not None and tuple(fail["path"]) == tuple(path)
          and fail["kind"] == kind)
    got = (f"REJECT at {fail['path']}, {fail['kind']}"
           if fail else "ACCEPT")
    results.append((ok, f"{'PASS' if ok else 'FAIL'}  {name} ({got})"))


def expect_defeq(name, t_src, u_src, ctx, want):
    try:
        t = parse_term_source(t_src)
        u = parse_term_source(u_src)
    except ParseError as e:
        results.append((False, f"FAIL  {name}: parse error {e}"))
        return
    got = defeq(ctx, t, u)
    ok = got == want
    results.append((ok,
                    f"{'PASS' if ok else 'FAIL'}  {name} (defeq = {got})"))


def expect_parse_error(name, src):
    try:
        parse_term_source(src)
    except ParseError as e:
        results.append((True, f"PASS  {name} (parse error: {e})"))
        return
    results.append((False, f"FAIL  {name}: parsed without error"))


# --- Section 1 of examples_dt.py, in surface syntax ------------------------

expect_check("dependent identity",
             r"\A:Type 0. \x:A. x", "Pi A:Type 0. Pi x:A. A", ctx0,
             ("accept",))
expect_check("dependent const",
             r"\A:Type 0. \B:Type 0. \x:A. \y:B. x",
             "Pi A:Type 0. Pi B:Type 0. Pi x:A. Pi y:B. A", ctx0,
             ("accept",))
expect_check("type-level function",
             r"\A:Type 0. A -> A", "Pi A:Type 0. Type 0", ctx0,
             ("accept",))
expect_check("conversion through a beta-redex",
             r"\x:B. x", r"(\A:Type 0. A -> A) B", ctx0,
             ("accept",))
expect_check("S",
             r"\f:(A -> B -> C). \g:(A -> B). \x:A. (f x) (g x)",
             "(A -> B -> C) -> (A -> B) -> A -> C", ctx0,
             ("accept",))
expect_check("applied dependent identity",
             r"((\A:Type 0. \x:A. x) A) x", "A", {**ctx0, "x": A},
             ("accept",))
expect_check("capture, handled (a2)",
             r"((\A:Type 0. \x:A. \A:Type 0. x) B)",
             "Pi x:B. Pi A2:Type 0. B", ctx0,
             ("accept",))
expect_check("type family applied to the binder",
             r"\x:A. f x", "Pi x:A. F x", ctx_fam,
             ("accept",))

# --- Section 2 of examples_dt.py, in surface syntax ------------------------

expect_check("Girard: Type 0 : Type 0",
             "Type 0", "Type 0", ctx0,
             ("reject", (), "mismatch"))
expect_check("dependent application, wrong argument",
             r"((\A:Type 0. \x:A. x) A) y", "A", {**ctx0, "y": B},
             ("reject", ("arg",), "mismatch"))
expect_infer("binder annotation is not a type",
             r"\x:f. x", {**ctx0, "f": Pi("_", A, A)},
             ("reject", ("binder",), "not-a-type"))
expect_check("type-level self-application",
             r"\A:Type 0. A A", "Pi A:Type 0. Type 0", ctx0,
             ("reject", ("body", "func"), "not-pi"))
expect_check("wrong universe",
             "Pi A:Type 0. A -> A", "Type 0", ctx0,
             ("reject", (), "mismatch"))

# --- Section 3 of examples_dt.py (break-it), in surface syntax -------------

expect_check("identity, wrong codomain",
             r"\A:Type 0. \x:A. x", "Pi A:Type 0. Pi x:A. B", ctx0,
             ("reject",))
expect_check("S with f and g swapped",
             r"\f:(A -> B -> C). \g:(A -> B). \x:A. (g x) (f x)",
             "(A -> B -> C) -> (A -> B) -> A -> C", ctx0,
             ("reject",))
expect_check("conversion, wrong argument type",
             r"\x:C. x", r"(\A:Type 0. A -> A) B", ctx0,
             ("reject",))
expect_check("applied identity, x : B",
             r"((\A:Type 0. \x:A. x) A) x", "A", {**ctx0, "x": B},
             ("reject", ("arg",), "mismatch"))

# --- Section 4 of examples_dt.py (defeq), in surface syntax ----------------

expect_defeq("eta: f == \\x:A. f x",
             "f", r"\x:A. f x", ctx_eta, True)
expect_defeq("distinct variables: f == g",
             "f", "g", ctx_eta, False)
expect_defeq("alpha: \\x:A. x == \\y:A. y",
             r"\x:A. x", r"\y:A. y", ctx0, True)

# --- Unicode spellings ------------------------------------------------------

expect_check("unicode lambda and Pi",
             "\u03bbA:Type 0. \u03bbx:A. x",
             "\u03a0A:Type 0. \u03a0x:A. A", ctx0,
             ("accept",))
expect_check("unicode arrow",
             r"\A:Type 0. A → A", "Pi A:Type 0. Type 0", ctx0,
             ("accept",))

# --- Parse errors ------------------------------------------------------------

expect_parse_error("unbalanced paren", "(A -> B")
expect_parse_error("lambda with no body", r"\x:A. ")
expect_parse_error("Pi with no body", "Pi x:A")
expect_parse_error("Type with no level", "Type")
expect_parse_error("stray closing paren", "A B C )")
expect_parse_error("arrow with no right side", "A ->")
expect_parse_error("unexpected character", "A @ B")


# --- Report ------------------------------------------------------------------

for ok, line in results:
    print(line)
passed = sum(1 for ok, _ in results if ok)
print(f"\n{passed}/{len(results)} passed")
if passed != len(results):
    raise SystemExit(1)
