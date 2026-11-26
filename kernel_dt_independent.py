"""Independent dependent-types kernel (rungs 1-2 sealing pass).

Derived from RULES_DT.md alone. Deliberately structurally different
from the reference kernel in three ways, so that agreement between
the two is evidence about the rules, not about shared code:

1. De Bruijn indices, not named variables. Capture is structurally
   impossible: a binder is a number, substitution is shifting, and
   there is no renaming step that could capture. The comparison
   therefore tests whether the named-variable implementation's
   capture handling agrees with a representation where capture
   cannot happen.
2. Arg-first traversal in App. The argument is inferred before the
   function, the opposite of the usual fn-first order. The first
   failure a checker reports on a doubly-bad application therefore
   comes from the argument side here, from the function side there
   -- the observer-view frame difference, made concrete.
3. Full beta-normalization as the conversion strategy. Both sides
   of a definitional-equality check are normalized completely
   (under binders, in arguments) before comparison, instead of
   reducing only head redexes (whnf). Confluence plus strong
   normalization makes the verdicts the same; the reduction work
   is not -- step counts are the frame-dependent quantity.

Failure contract (the proposer's gradient reads off this shape):
failures are dicts {path, kind, expected, actual, message} with
path a tuple of term positions: () top level, "func"/"arg" for an
application, "body" for a Lam/Pi body, "binder" for a Lam/Pi
annotation. Kinds: "mismatch" (definitional equality failed),
"not-a-type" (a binder annotation or expected type is not a
universe), "not-pi" (a function position's type is not a Pi),
"unbound" (a variable index outside the context), "fuel"
(reduction budget exhausted -- a safety net, see below).

Terms are tuples: ("var", n), ("lam", ty, body), ("app", f, a),
("pi", ty, body), ("type", i). A context is a list of types with
the most recently bound binder at index 0; the type of Var i is
shift(ctx[i], i+1), with context entries stored as written.

Rung 2 (Nat + eliminator) carries the same three differences
over: the eliminator is a four-component term ("rec", P, z, s, n)
with no binder, so de Bruijn needs no new shifting machinery for
it; iota is a reduction in the full normalizer (the target is
normalized first, then Zero/Succ decide, neutral means stuck);
Rec typing is motive-dependent, checked motive first, then
target, base, step. No eta for Nat: its inhabitants are observed
by the eliminator, not by application.
"""

# ---------------------------------------------------------------- terms

def Var(n):
    return ("var", n)


def Lam(ty, body):
    return ("lam", ty, body)


def App(f, a):
    return ("app", f, a)


def Pi(ty, body):
    return ("pi", ty, body)


def Type(i):
    return ("type", i)


def Nat():
    return ("nat",)


def Zero():
    return ("zero",)


def Succ(n):
    return ("succ", n)


def Rec(P, z, s, n):
    return ("rec", P, z, s, n)


def show(t, names=()):
    """Pretty-print a de Bruijn term with generated binder names."""
    tag = t[0]
    if tag == "var":
        n = t[1]
        if n < len(names):
            return names[n]
        return "free%d" % (n - len(names))
    if tag == "type":
        return "Type %d" % t[1]
    if tag == "nat":
        return "Nat"
    if tag == "zero":
        return "Zero"
    if tag == "succ":
        return "(Succ %s)" % show(t[1], names)
    if tag == "rec":
        return "(Rec %s %s %s %s)" % (show(t[1], names), show(t[2], names),
                                      show(t[3], names), show(t[4], names))
    if tag == "lam":
        x = "x%d" % len(names)
        return "(lam %s:%s. %s)" % (x, show(t[1], names),
                                    show(t[2], (x,) + tuple(names)))
    if tag == "pi":
        x = "x%d" % len(names)
        return "(Pi %s:%s. %s)" % (x, show(t[1], names),
                                   show(t[2], (x,) + tuple(names)))
    if tag == "app":
        return "(%s %s)" % (show(t[1], names), show(t[2], names))
    return repr(t)


# ------------------------------------------------------- shift / subst

def shift(t, d, cutoff=0):
    """Bump free variables (index >= cutoff) by d."""
    tag = t[0]
    if tag == "var":
        n = t[1]
        return ("var", n + d) if n >= cutoff else t
    if tag in ("lam", "pi"):
        return (tag, shift(t[1], d, cutoff), shift(t[2], d, cutoff + 1))
    if tag == "app":
        return ("app", shift(t[1], d, cutoff), shift(t[2], d, cutoff))
    if tag == "succ":
        return ("succ", shift(t[1], d, cutoff))
    if tag == "rec":
        return ("rec", shift(t[1], d, cutoff), shift(t[2], d, cutoff),
                shift(t[3], d, cutoff), shift(t[4], d, cutoff))
    return t  # type, nat, zero


def subst(t, j, s):
    """Capture-avoiding substitution: replace Var j with s."""
    tag = t[0]
    if tag == "var":
        n = t[1]
        if n == j:
            return s
        if n > j:
            return ("var", n - 1)
        return t
    if tag in ("lam", "pi"):
        return (tag, subst(t[1], j, s), subst(t[2], j + 1, shift(s, 1)))
    if tag == "app":
        return ("app", subst(t[1], j, s), subst(t[2], j, s))
    if tag == "succ":
        return ("succ", subst(t[1], j, s))
    if tag == "rec":
        return ("rec", subst(t[1], j, s), subst(t[2], j, s),
                subst(t[3], j, s), subst(t[4], j, s))
    return t  # type, nat, zero


# ------------------------------------------------------------- errors

class CheckError(Exception):
    def __init__(self, path, kind, expected=None, actual=None, message=""):
        super().__init__(message or kind)
        self.path = tuple(path)
        self.kind = kind
        self.expected = expected
        self.actual = actual
        self.message = message or kind

    def as_dict(self):
        return {"path": list(self.path), "kind": self.kind,
                "expected": self.expected, "actual": self.actual,
                "message": self.message}


# ------------------------------------------------------------ metering

class Meter:
    """Counts beta steps across one top-level call. The fuel limit is
    a safety net, not a load-bearing part: well-typed terms in this
    fragment normalize, so it should never fire on a well-typed
    input. It exists so an ill-typed term reaching normalize on an
    error path cannot loop the kernel."""

    def __init__(self, fuel_limit=10000):
        self.beta_steps = 0
        self.iota_steps = 0
        self.fuel_limit = fuel_limit


# ------------------------------------------------------- normalization
# Full beta-normalization: reduce everywhere (under binders, in
# arguments), not just head redexes. This is the strategy that
# differs from the spec's whnf; confluence makes the verdicts the
# same and the step counts different.

def normalize(t, meter):
    tag = t[0]
    if tag in ("var", "type", "nat", "zero"):
        return t
    if tag in ("lam", "pi"):
        return (tag, normalize(t[1], meter), normalize(t[2], meter))
    if tag == "succ":
        return ("succ", normalize(t[1], meter))
    if tag == "rec":
        # Iota: the target is normalized first, so the constructor
        # is visible. Zero -> base; Succ m -> step applied to m
        # and to the recursive call; neutral target -> stuck Rec
        # with all four components normalized.
        P = normalize(t[1], meter)
        z = normalize(t[2], meter)
        s = normalize(t[3], meter)
        n = normalize(t[4], meter)
        if n[0] == "zero":
            meter.iota_steps += 1
            if meter.beta_steps + meter.iota_steps > meter.fuel_limit:
                raise CheckError((), "fuel", None, None,
                                 "reduction fuel exhausted")
            return normalize(z, meter)
        if n[0] == "succ":
            meter.iota_steps += 1
            if meter.beta_steps + meter.iota_steps > meter.fuel_limit:
                raise CheckError((), "fuel", None, None,
                                 "reduction fuel exhausted")
            m = n[1]
            return normalize(("app", ("app", s, m),
                              ("rec", P, z, s, m)), meter)
        return ("rec", P, z, s, n)
    # app: normalize both sides, then beta-reduce if the head is a Lam
    f = normalize(t[1], meter)
    a = normalize(t[2], meter)
    if f[0] == "lam":
        meter.beta_steps += 1
        if meter.beta_steps + meter.iota_steps > meter.fuel_limit:
            raise CheckError((), "fuel", None, None,
                             "reduction fuel exhausted")
        return normalize(subst(f[2], 0, a), meter)
    return ("app", f, a)


# ------------------------------------------------- definitional equality

def defeq_raw(a, b, ctx, meter):
    """Boolean definitional equality: normalize both sides fully,
    then compare structurally, with eta for functions."""
    return _compare(normalize(a, meter), normalize(b, meter), ctx, meter)


def _compare(a, b, ctx, meter):
    """Compare two beta-normal forms. `ctx` is the de Bruijn context
    the (sub)terms live in; it is needed only for the eta case, where
    the neutral's type must be inferred."""
    if a[0] == "lam" and b[0] == "lam":
        return (_compare(a[1], b[1], ctx, meter)
                and _compare(a[2], b[2], [a[1]] + ctx, meter))
    if a[0] == "lam" or b[0] == "lam":
        # Eta: exactly one side is a Lam. Infer the neutral's type;
        # if it is a Pi, eta-expand the neutral and compare. The
        # Lam's annotation must be defeq to the Pi domain, or the
        # two sides live in different contexts.
        lam, neu = (a, b) if a[0] == "lam" else (b, a)
        try:
            neu_ty = infer(neu, ctx, (), meter)
        except CheckError:
            return False
        nf_ty = normalize(neu_ty, meter)
        if nf_ty[0] != "pi":
            return False
        if not _compare(lam[1], nf_ty[1], ctx, meter):
            return False
        return _compare(lam[2],
                         ("app", shift(neu, 1), ("var", 0)),
                         [nf_ty[1]] + ctx, meter)
    if a[0] == "app" and b[0] == "app":
        return (_compare(a[1], b[1], ctx, meter)
                and _compare(a[2], b[2], ctx, meter))
    if a[0] == "pi" and b[0] == "pi":
        return (_compare(a[1], b[1], ctx, meter)
                and _compare(a[2], b[2], [a[1]] + ctx, meter))
    if a[0] == "nat" and b[0] == "nat":
        return True
    if a[0] == "zero" and b[0] == "zero":
        return True
    if a[0] == "succ" and b[0] == "succ":
        return _compare(a[1], b[1], ctx, meter)
    if a[0] == "rec" and b[0] == "rec":
        return (_compare(a[1], b[1], ctx, meter)
                and _compare(a[2], b[2], ctx, meter)
                and _compare(a[3], b[3], ctx, meter)
                and _compare(a[4], b[4], ctx, meter))
    if a[0] == "type" and b[0] == "type":
        return a[1] == b[1]
    if a[0] == "var" and b[0] == "var":
        return a[1] == b[1]
    return False


def defeq(ctx, t, u, meter=None):
    """Public definitional equality: ctx is a de Bruijn context
    (list of types). Returns a bool; raises CheckError on fuel."""
    meter = meter or Meter()
    return defeq_raw(t, u, ctx, meter)


# ------------------------------------------------------------- checking

def as_universe(t, ctx, path, meter):
    """Return the level i such that t : Type i, or raise not-a-type
    at `path`. This is the 'is a type' gate, used for binder
    annotations, Pi domains/codomains, and the public entry's
    expected-type validation."""
    ty = infer(t, ctx, path, meter)
    nf = normalize(ty, meter)
    if nf[0] == "type":
        return nf[1]
    raise CheckError(path, "not-a-type", "a universe (Type i)", show(ty),
                     "%s is not a type (its type is %s)" % (show(t), show(ty)))


def infer(t, ctx, path, meter):
    tag = t[0]
    if tag == "var":
        n = t[1]
        if n < len(ctx):
            return shift(ctx[n], n + 1)
        raise CheckError(path, "unbound", None, None,
                         "variable %d is unbound" % n)
    if tag == "type":
        i = t[1]
        if i < 0:
            raise ValueError("negative universe level")
        return ("type", i + 1)
    if tag == "nat":
        return ("type", 0)
    if tag == "zero":
        return ("nat",)
    if tag == "succ":
        # The predecessor is checked at ("pred",): Succ of a
        # non-Nat fails there, not at the top level.
        pred_ty = infer(t[1], ctx, path + ("pred",), meter)
        if not defeq_raw(pred_ty, ("nat",), ctx, meter):
            raise CheckError(path + ("pred",), "mismatch", "Nat",
                             show(pred_ty),
                             "successor predecessor is not a Nat")
        return ("nat",)
    if tag == "rec":
        # Motive-dependent typing. Checked motive first: the base
        # and step checks both need P, and a bad motive must be
        # reported at ("motive",) rather than failing inside them.
        P, z, s, n = t[1], t[2], t[3], t[4]
        P_ty = infer(P, ctx, path + ("motive",), meter)
        nf_P = normalize(P_ty, meter)
        if nf_P[0] != "pi":
            raise CheckError(path + ("motive",), "not-pi", "a Pi type",
                             show(P_ty), "motive type is not a Pi type")
        dom, cod = nf_P[1], nf_P[2]
        if not defeq_raw(dom, ("nat",), ctx, meter):
            raise CheckError(path + ("motive",), "mismatch", "Nat",
                             show(dom), "motive domain is not Nat")
        # The codomain must be a universe -- P : Nat -> Type i --
        # which is that it normalizes to a ("type", i) tag, not
        # merely that it is a type. as_universe would wrongly
        # accept a motive of type Nat -> Nat (its codomain is a
        # type, but P n would be a natural, not a type).
        nf_cod = normalize(cod, meter)
        if nf_cod[0] != "type":
            raise CheckError(path + ("motive",), "not-a-type",
                             "a universe (Type i)", show(cod),
                             "motive codomain is not a universe")
        # Target: n : Nat.
        n_ty = infer(n, ctx, path + ("target",), meter)
        if not defeq_raw(n_ty, ("nat",), ctx, meter):
            raise CheckError(path + ("target",), "mismatch", "Nat",
                             show(n_ty), "recursor target is not a Nat")
        # Base: z : P Zero.
        z_ty = infer(z, ctx, path + ("base",), meter)
        if not defeq_raw(z_ty, ("app", P, ("zero",)), ctx, meter):
            raise CheckError(path + ("base",), "mismatch",
                             show(("app", P, ("zero",))), show(z_ty),
                             "base case type does not match P Zero")
        # Step: s : Pi n:Nat. P n -> P (Succ n). The expected
        # type is built from P with the correct shifts: under one
        # binder P is shift(P, 1) and n is Var 0; under two, P is
        # shift(P, 2) and n is Var 1.
        s_ty = infer(s, ctx, path + ("step",), meter)
        expected_s = ("pi", ("nat",),
                      ("pi", ("app", shift(P, 1), ("var", 0)),
                       ("app", shift(P, 2), ("succ", ("var", 1)))))
        if not defeq_raw(s_ty, expected_s, ctx, meter):
            raise CheckError(path + ("step",), "mismatch",
                             show(expected_s), show(s_ty),
                             "step case type does not match "
                             "Pi n:Nat. P n -> P (Succ n)")
        return ("app", P, n)
    if tag == "lam":
        as_universe(t[1], ctx, path + ("binder",), meter)
        body_ty = infer(t[2], [t[1]] + ctx, path + ("body",), meter)
        return ("pi", t[1], body_ty)
    if tag == "pi":
        # The Pi domain is "domain" (the reference kernel's word);
        # "binder" is reserved for Lam annotations, matching the
        # established failure-path contract. Depths are identical
        # either way; the name is what differs.
        i = as_universe(t[1], ctx, path + ("domain",), meter)
        j = as_universe(t[2], [t[1]] + ctx, path + ("body",), meter)
        return ("type", max(i, j))
    if tag == "app":
        # ARG-FIRST: the argument is inferred before the function.
        # A doubly-bad application therefore reports the argument's
        # failure first; a fn-first checker reports the function's.
        a_ty = infer(t[2], ctx, path + ("arg",), meter)
        f_ty = infer(t[1], ctx, path + ("func",), meter)
        nf = normalize(f_ty, meter)
        if nf[0] == "pi":
            dom, cod = nf[1], nf[2]
            if not defeq_raw(a_ty, dom, ctx, meter):
                raise CheckError(path + ("arg",), "mismatch",
                                 show(dom), show(a_ty),
                                 "argument type does not match the domain")
            return subst(cod, 0, t[2])
        raise CheckError(path + ("func",), "not-pi", "a Pi type",
                         show(f_ty), "function type is not a Pi type")
    raise ValueError("unknown term tag %r" % (tag,))


def check(t, ty, ctx, path, meter):
    if t[0] == "lam":
        nf = normalize(ty, meter)
        if nf[0] == "pi":
            dom, cod = nf[1], nf[2]
            if not defeq_raw(t[1], dom, ctx, meter):
                raise CheckError(path + ("binder",), "mismatch",
                                 show(dom), show(t[1]),
                                 "lambda annotation does not match "
                                 "the Pi domain")
            check(t[2], cod, [t[1]] + ctx, path + ("body",), meter)
            return
        raise CheckError(path, "mismatch", "a Pi type", show(ty),
                         "lambda checked against a non-Pi type")
    inferred = infer(t, ctx, path, meter)
    if not defeq_raw(inferred, ty, ctx, meter):
        raise CheckError(path, "mismatch", show(ty), show(inferred),
                         "inferred type is not definitionally equal to "
                         "the expected type")


def check_or_fail(ctx, term, ty, meter=None):
    """Public entry. Validates the expected type first (it must
    itself be a type), then checks. Returns None on success, or the
    first failure as {path, kind, expected, actual, message}."""
    meter = meter or Meter()
    try:
        as_universe(ty, ctx, (), meter)
        check(term, ty, ctx, (), meter)
        return None
    except CheckError as e:
        return e.as_dict()


def infer_or_fail(ctx, term, meter=None):
    """Public inference. Returns (type, None) or (None, failure)."""
    meter = meter or Meter()
    try:
        return infer(term, ctx, (), meter), None
    except CheckError as e:
        return None, e.as_dict()
