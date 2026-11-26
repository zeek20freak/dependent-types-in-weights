"""Reference kernel for the minimal dependent fragment, v0.2 (rung 2).

Pi + universes + conversion (alpha/beta/eta) + Nat with its
eliminator (rung 2). No Prop, no cumulativity, no universe
polymorphism. Plain Python, no dependencies —
the spec in executable form, like kernel.py for STLC.

The jump from STLC, stated once: in STLC a type is syntax and checking
is structural; here a type is a TERM, so checking must EVALUATE.
That is the whole difference, and it is all of the work:

  * whnf — weak head normal form: reduce head beta-redexes, and
    head iota-redexes (the Nat recursor on Zero/Succ). Iota has
    to see the constructor, so whnf reduces the recursor's target
    first; a neutral target leaves the recursor stuck. There is
    nothing else to reduce. No definitions to unfold (no delta —
    the fragment has no environment, no let, no constants).
  * defeq — definitional equality, type-directed only where it has to
    be: eta for functions needs a type to expand against, so defeq
    is untyped except that it infers the type of a neutral when
    exactly one side is a Lam. Everything else is structural on
    whnf'd forms.
  * universes — Type i : Type (i+1), Pi formation takes max. No
    Type : Type (Girard), no cumulativity (strict levels).
  * Nat — Nat : Type 0; Zero : Nat; Succ n : Nat; Rec P z s n : P n
    for a motive P : Nat -> Type i. The eliminator is the second
    mechanism: inductive elimination, motive-dependent typing.

Failure paths: infer/check raise CheckError carrying (path, kind,
expected, actual); infer_or_fail/check_or_fail are thin wrappers
that turn the exception into the same failure dict shape as
kernel.py's check_or_fail (path/kind/expected/actual/message) —
the graded proposer reads failure depth off that shape, so it is
kept exactly. The implementation differs from kernel.py (exception-
carried paths instead of duplicated _or_fail logic); the shape is
identical.

Soundness note: check_or_fail (the public entry) verifies the
expected type is itself a type. Internal check trusts it: every
type that reaches an internal check is either validated at the top
or computed by infer (which validates binder annotations and
returns well-formed types). The context is trusted input, as in
kernel.py.
"""

# --- Terms ---------------------------------------------------------------
# Types ARE terms. A "type" is a term whose type is a universe.

class Var:
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, Var) and self.name == other.name

    def __hash__(self):
        return hash(("Var", self.name))

    def __repr__(self):
        return self.name


class Lam:
    """λvar:var_type. body — var_type is a TERM (a type)."""
    def __init__(self, var, var_type, body):
        self.var = var
        self.var_type = var_type
        self.body = body

    def __eq__(self, other):
        return (isinstance(other, Lam)
                and self.var == other.var
                and self.var_type == other.var_type
                and self.body == other.body)

    def __hash__(self):
        return hash(("Lam", self.var, self.var_type, self.body))

    def __repr__(self):
        return f"(λ{self.var}:{self.var_type}. {self.body})"


class App:
    """func arg"""
    def __init__(self, func, arg):
        self.func = func
        self.arg = arg

    def __eq__(self, other):
        return (isinstance(other, App)
                and self.func == other.func
                and self.arg == other.arg)

    def __hash__(self):
        return hash(("App", self.func, self.arg))

    def __repr__(self):
        return f"({self.func} {self.arg})"


class Pi:
    """Πvar:var_type. body — dependent function type. var_type is a TERM."""
    def __init__(self, var, var_type, body):
        self.var = var
        self.var_type = var_type
        self.body = body

    def __eq__(self, other):
        return (isinstance(other, Pi)
                and self.var == other.var
                and self.var_type == other.var_type
                and self.body == other.body)

    def __hash__(self):
        return hash(("Pi", self.var, self.var_type, self.body))

    def __repr__(self):
        # Non-dependent Pi reads as an arrow.
        if self.var not in free_vars(self.body):
            return f"({self.var_type} -> {self.body})"
        return f"(Π{self.var}:{self.var_type}. {self.body})"


class Type:
    """Type i — the i-th universe. Type i : Type (i+1)."""
    def __init__(self, level):
        assert isinstance(level, int) and level >= 0
        self.level = level

    def __eq__(self, other):
        return isinstance(other, Type) and self.level == other.level

    def __hash__(self):
        return hash(("Type", self.level))

    def __repr__(self):
        return f"Type {self.level}"


class Nat:
    """Nat — the type of natural numbers. Nat : Type 0."""
    def __eq__(self, other):
        return isinstance(other, Nat)

    def __hash__(self):
        return hash("Nat")

    def __repr__(self):
        return "Nat"


class Zero:
    """Zero : Nat."""
    def __eq__(self, other):
        return isinstance(other, Zero)

    def __hash__(self):
        return hash("Zero")

    def __repr__(self):
        return "Zero"


class Succ:
    """Succ n : Nat when n : Nat."""
    def __init__(self, pred):
        self.pred = pred

    def __eq__(self, other):
        return isinstance(other, Succ) and self.pred == other.pred

    def __hash__(self):
        return hash(("Succ", self.pred))

    def __repr__(self):
        return f"(Succ {self.pred})"


class Rec:
    """Rec motive base step target — the Nat eliminator (recursor /
    induction principle). No binder: motive, base, step and target
    are all plain terms. Rec P z s n : P n when P : Nat -> Type i,
    z : P Zero, s : Π n:Nat. P n -> P (Succ n), n : Nat."""
    def __init__(self, motive, base, step, target):
        self.motive = motive
        self.base = base
        self.step = step
        self.target = target

    def __eq__(self, other):
        return (isinstance(other, Rec)
                and self.motive == other.motive
                and self.base == other.base
                and self.step == other.step
                and self.target == other.target)

    def __hash__(self):
        return hash(("Rec", self.motive, self.base, self.step,
                     self.target))

    def __repr__(self):
        return (f"(Rec {self.motive} {self.base} {self.step} "
                f"{self.target})")


# --- Free variables, substitution ----------------------------------------

def free_vars(t):
    if isinstance(t, Var):
        return {t.name}
    if isinstance(t, (Lam, Pi)):
        return free_vars(t.var_type) | (free_vars(t.body) - {t.var})
    if isinstance(t, App):
        return free_vars(t.func) | free_vars(t.arg)
    if isinstance(t, Succ):
        return free_vars(t.pred)
    if isinstance(t, Rec):
        return (free_vars(t.motive) | free_vars(t.base)
                | free_vars(t.step) | free_vars(t.target))
    return set()  # Type, Nat, Zero


def _fresh(base, avoid):
    """A name derived from base that occurs in none of avoid.
    Deterministic, no global state."""
    name = base
    while name in avoid:
        name += "'"
    return name


def subst(t, x, u):
    """Capture-avoiding substitution: t with free x replaced by u."""
    if isinstance(t, Var):
        return u if t.name == x else t
    if isinstance(t, (Lam, Pi)):
        ty = subst(t.var_type, x, u)
        if t.var == x:
            return type(t)(t.var, ty, t.body)
        if t.var in free_vars(u):
            z = _fresh(t.var, free_vars(u) | free_vars(t.body) | {x})
            body = subst(subst(t.body, t.var, Var(z)), x, u)
            return type(t)(z, ty, body)
        return type(t)(t.var, ty, subst(t.body, x, u))
    if isinstance(t, App):
        return App(subst(t.func, x, u), subst(t.arg, x, u))
    if isinstance(t, Succ):
        return Succ(subst(t.pred, x, u))
    if isinstance(t, Rec):
        # No binder in Rec: substitute in all four components.
        return Rec(subst(t.motive, x, u), subst(t.base, x, u),
                   subst(t.step, x, u), subst(t.target, x, u))
    return t  # Type, Nat, Zero


# --- Errors, with paths ----------------------------------------------------

class CheckError(Exception):
    def __init__(self, message, kind="error", path=(),
                 expected=None, actual=None):
        super().__init__(message)
        self.message = message
        self.kind = kind
        self.path = path
        self.expected = expected
        self.actual = actual

    def failure(self):
        return {"path": self.path, "kind": self.kind,
                "expected": self.expected, "actual": self.actual,
                "message": self.message}


# --- whnf: weak head normal form ------------------------------------------
# Head beta-reduction plus head iota-reduction (the Nat recursor).
# No context: there is nothing to unfold (no definitions, no let,
# no constants — delta does not exist here). Iota has to see the
# constructor, so the recursor's target is reduced to whnf first;
# a neutral target leaves the recursor stuck (itself the whnf,
# with the target replaced by its whnf — the same move made for App).
#
# Fuel is defensive, not load-bearing: every well-typed term in this
# fragment normalizes (no fixpoint, no Type : Type, structural
# recursion on Nat), so whnf terminates on well-typed terms. The
# bound exists so an ill-typed term reaching whnf on an error path
# cannot loop the kernel. Each beta and each iota step costs 1 fuel.

WHNF_FUEL = 1000


def whnf(t, fuel=None):
    """Reduce head beta- and iota-redexes until the head is not one."""
    budget = [WHNF_FUEL if fuel is None else fuel]

    def go(t):
        while True:
            if isinstance(t, App):
                f = go(t.func)
                if isinstance(f, Lam):
                    if budget[0] <= 0:
                        raise CheckError("whnf fuel exhausted",
                                         kind="fuel")
                    budget[0] -= 1
                    t = subst(f.body, f.var, t.arg)
                    continue
                return App(f, t.arg)
            if isinstance(t, Rec):
                n_w = go(t.target)
                if isinstance(n_w, Zero):
                    if budget[0] <= 0:
                        raise CheckError("whnf fuel exhausted",
                                         kind="fuel")
                    budget[0] -= 1
                    t = t.base
                    continue
                if isinstance(n_w, Succ):
                    if budget[0] <= 0:
                        raise CheckError("whnf fuel exhausted",
                                         kind="fuel")
                    budget[0] -= 1
                    t = App(App(t.step, n_w.pred),
                            Rec(t.motive, t.base, t.step, n_w.pred))
                    continue
                return Rec(t.motive, t.base, t.step, n_w)
            return t

    return go(t)


# --- The kernel -------------------------------------------------------------

def _level_of_type(ctx, t, path=()):
    """infer t; it must be a universe; return its level.
    This is the Pi-formation check and the binder-annotation check."""
    ty = infer(ctx, t, path)
    w = whnf(ty)
    if isinstance(w, Type):
        return w.level
    raise CheckError(f"expected a type, got {ty}", kind="not-a-type",
                     path=path, expected="a universe", actual=ty)


def infer(ctx, term, path=()):
    """Return the type of term under ctx, or raise CheckError."""
    if isinstance(term, Var):
        # Rule: Var
        if term.name not in ctx:
            raise CheckError(f"unbound variable {term.name}",
                             kind="unbound", path=path)
        return ctx[term.name]
    if isinstance(term, Type):
        # Rule: Type — Type i : Type (i+1). The hierarchy is strict:
        # there is no level i with Type i : Type i (Girard).
        return Type(term.level + 1)
    if isinstance(term, Lam):
        # Rule: Lam — the annotation must be a type; extend the
        # context, infer the body.
        _level_of_type(ctx, term.var_type, path + ("binder",))
        # Never shadow: if the binder name is already bound in ctx,
        # a stored type may mention the outer binding of this name,
        # and the body's inferred type would then be ambiguous under
        # the new binder (the a2 capture bug: the Pi built here
        # captured an outer variable of the same name). Freshen the
        # binder first — alpha-convert the body — so inner references
        # use the fresh name and outer references keep the old one.
        # (Renaming the Pi binder at construction instead would be
        # wrong: when the body's type mentions the binder itself,
        # e.g. a type family applied to it, the binder name must
        # stay, or the variable is left free.)
        var = term.var
        body = term.body
        if var in ctx:
            var = _fresh(var, free_vars(body) | set(ctx))
            body = subst(body, term.var, Var(var))
        new_ctx = dict(ctx)
        new_ctx[var] = term.var_type
        body_type = infer(new_ctx, body, path + ("body",))
        return Pi(var, term.var_type, body_type)
    if isinstance(term, Pi):
        # Rule: Pi formation — domain a type at level i, codomain a
        # type at level j under the binder; the Pi lives at max(i, j).
        i = _level_of_type(ctx, term.var_type, path + ("domain",))
        new_ctx = dict(ctx)
        new_ctx[term.var] = term.var_type
        j = _level_of_type(new_ctx, term.body, path + ("body",))
        return Type(max(i, j))
    if isinstance(term, App):
        # Rule: App — the function's type must whnf to a Pi; the
        # argument must have the domain type; the result is the
        # codomain with the argument substituted in.
        func_type = infer(ctx, term.func, path + ("func",))
        func_type_w = whnf(func_type)
        if not isinstance(func_type_w, Pi):
            raise CheckError(
                f"applied a non-function of type {func_type}",
                kind="not-pi", path=path + ("func",),
                expected="a Pi type", actual=func_type)
        check(ctx, term.arg, func_type_w.var_type, path + ("arg",))
        return subst(func_type_w.body, func_type_w.var, term.arg)
    if isinstance(term, Nat):
        # Rule: Nat — Nat : Type 0.
        return Type(0)
    if isinstance(term, Zero):
        # Rule: Zero — Zero : Nat.
        return Nat()
    if isinstance(term, Succ):
        # Rule: Succ — the predecessor must be a Nat.
        check(ctx, term.pred, Nat(), path + ("pred",))
        return Nat()
    if isinstance(term, Rec):
        # Rule: Rec (the eliminator) — motive-dependent typing.
        # P must have type Nat -> Type i: infer it, whnf to a Pi
        # whose domain is Nat and whose codomain is a universe.
        motive_ty = infer(ctx, term.motive, path + ("motive",))
        motive_ty_w = whnf(motive_ty)
        if not isinstance(motive_ty_w, Pi):
            raise CheckError(
                f"recursor motive has non-function type {motive_ty}",
                kind="not-pi", path=path + ("motive",),
                expected="Nat -> Type i", actual=motive_ty)
        if not defeq(ctx, motive_ty_w.var_type, Nat(),
                     path + ("motive",)):
            raise CheckError(
                f"recursor motive domain {motive_ty_w.var_type} "
                f"is not Nat",
                kind="mismatch", path=path + ("motive",),
                expected=Nat(), actual=motive_ty_w.var_type)
        # The codomain must BE a universe — P : Nat -> Type i — not
        # merely be a type. _level_of_type would wrongly accept a
        # motive of type Nat -> Nat (its codomain is a type, but
        # P n would be a natural, not a type). whnf needs no
        # context, and the body is read under the binder; a
        # neutral codomain (e.g. F n for a variable F) is rejected
        # too — the codomain must normalize to a universe tag.
        # (Caught by the independent kernel in the rung-2 sealing
        # pass, 2026-11-25: the reference accepted Nat -> Nat as
        # a motive and only failed later, at the base check.)
        cod_w = whnf(motive_ty_w.body)
        if not isinstance(cod_w, Type):
            raise CheckError(
                f"recursor motive codomain {motive_ty_w.body} "
                f"is not a universe",
                kind="not-a-type", path=path + ("motive",),
                expected="a universe (Type i)",
                actual=motive_ty_w.body)
        # z : P Zero
        check(ctx, term.base, App(term.motive, Zero()),
              path + ("base",))
        # s : Π n:Nat. P n -> P (Succ n)
        nv = _fresh("n", free_vars(term.motive) | set(ctx))
        rv = _fresh("r", free_vars(term.motive) | {nv} | set(ctx))
        step_ty = Pi(nv, Nat(),
                     Pi(rv, App(term.motive, Var(nv)),
                        App(term.motive, Succ(Var(nv)))))
        check(ctx, term.step, step_ty, path + ("step",))
        # n : Nat; the result is P n.
        check(ctx, term.target, Nat(), path + ("target",))
        return App(term.motive, term.target)
    raise CheckError(f"unknown term {term!r}", kind="unknown-term",
                     path=path)


def check(ctx, term, type_, path=()):
    """True if term has type_ under ctx; raises CheckError otherwise.
    Internal: trusts type_ to be a type (validated at the public
    entry, check_or_fail, or computed by infer)."""
    type_w = whnf(type_)
    if isinstance(term, Lam) and isinstance(type_w, Pi):
        # Lam against a Pi: the binder annotation must match the
        # domain, then the body is checked against the codomain.
        if not defeq(ctx, term.var_type, type_w.var_type,
                     path + ("binder",)):
            raise CheckError(
                f"binder type {term.var_type} does not match "
                f"domain {type_w.var_type}",
                kind="mismatch", path=path + ("binder",),
                expected=type_w.var_type, actual=term.var_type)
        new_ctx = dict(ctx)
        # Never shadow (same capture class as infer's Lam rule):
        # freshen the binder before extending the context, so a
        # stored type mentioning the outer binding of this name is
        # not misread under the new binder.
        var = term.var
        body = term.body
        if var in ctx:
            var = _fresh(var, free_vars(body) | set(ctx))
            body = subst(body, term.var, Var(var))
        new_ctx[var] = type_w.var_type
        body_type = subst(type_w.body, type_w.var, Var(var))
        check(new_ctx, body, body_type, path + ("body",))
        return True
    inferred = infer(ctx, term, path)
    if not defeq(ctx, inferred, type_, path):
        raise CheckError(f"expected {type_}, got {inferred}",
                         kind="mismatch", path=path,
                         expected=type_, actual=inferred)
    return True


# --- defeq: definitional equality ------------------------------------------
# alpha + beta + eta for functions. Untyped, except for the one place
# eta needs a type: when exactly one side is a Lam, the other side's
# type is inferred, and if it whnf's to a Pi the neutral is
# eta-expanded. Every other case is structural on whnf'd forms.
#
# Termination: every recursive call is on strict subterms, except
# the eta branch, where the Lam is consumed and the neutral gains
# one App node — the pair's total size drops by the binder type's
# size (>= 1). whnf terminates on well-typed terms (strong
# normalization) with the fuel bound as backstop.

def _apply_var(term, z):
    """term applied to variable z: if term is a Lam, its body with
    the binder replaced by z (eta); else App(term, z)."""
    if isinstance(term, Lam):
        return subst(term.body, term.var, Var(z))
    return App(term, Var(z))


def defeq(ctx, t, u, path=()):
    """True if t and u are definitionally equal. Returns a bool;
    raises CheckError only on whnf fuel exhaustion."""
    t_w = whnf(t)
    u_w = whnf(u)
    if isinstance(t_w, Lam) and isinstance(u_w, Lam):
        # Both functions: binder types must agree, then bodies under
        # a shared fresh name (this is where alpha lives). The fresh
        # name must not shadow a context binding either (same
        # capture class as the Lam rules).
        if not defeq(ctx, t_w.var_type, u_w.var_type,
                     path + ("binder",)):
            return False
        z = _fresh(t_w.var, free_vars(t_w.body) | free_vars(u_w.body)
                   | set(ctx))
        ctx2 = dict(ctx)
        ctx2[z] = t_w.var_type
        return defeq(ctx2,
                     subst(t_w.body, t_w.var, Var(z)),
                     subst(u_w.body, u_w.var, Var(z)),
                     path + ("body",))
    if isinstance(t_w, Lam) or isinstance(u_w, Lam):
        # Eta: exactly one side is a Lam. The other side must have a
        # Pi type (inferred); expand it and compare.
        lam = t_w if isinstance(t_w, Lam) else u_w
        other = u_w if isinstance(t_w, Lam) else t_w
        try:
            other_ty = whnf(infer(ctx, other, path))
        except CheckError as e:
            if e.kind == "fuel":
                raise
            return False
        if not isinstance(other_ty, Pi):
            return False
        if not defeq(ctx, lam.var_type, other_ty.var_type,
                     path + ("binder",)):
            return False
        z = _fresh(lam.var, free_vars(lam.body) | free_vars(other)
                   | set(ctx))
        ctx2 = dict(ctx)
        ctx2[z] = other_ty.var_type
        return defeq(ctx2, _apply_var(t_w, z), _apply_var(u_w, z),
                     path + ("body",))
    # Both non-Lam: structural on whnf'd forms.
    if isinstance(t_w, Var) and isinstance(u_w, Var):
        return t_w.name == u_w.name
    if isinstance(t_w, Type) and isinstance(u_w, Type):
        return t_w.level == u_w.level
    if isinstance(t_w, Pi) and isinstance(u_w, Pi):
        if not defeq(ctx, t_w.var_type, u_w.var_type,
                     path + ("domain",)):
            return False
        z = _fresh(t_w.var, free_vars(t_w.body) | free_vars(u_w.body)
                   | set(ctx))
        ctx2 = dict(ctx)
        ctx2[z] = t_w.var_type
        return defeq(ctx2,
                     subst(t_w.body, t_w.var, Var(z)),
                     subst(u_w.body, u_w.var, Var(z)),
                     path + ("body",))
    if isinstance(t_w, App) and isinstance(u_w, App):
        return (defeq(ctx, t_w.func, u_w.func, path + ("func",))
                and defeq(ctx, t_w.arg, u_w.arg, path + ("arg",)))
    # Nat: structural. No eta for Nat — its inhabitants are
    # observed by the eliminator, not by application. Iota comes
    # free: both sides were whnf'd above, so a recursor on a
    # constructor has already reduced.
    if isinstance(t_w, Nat) and isinstance(u_w, Nat):
        return True
    if isinstance(t_w, Zero) and isinstance(u_w, Zero):
        return True
    if isinstance(t_w, Succ) and isinstance(u_w, Succ):
        return defeq(ctx, t_w.pred, u_w.pred, path + ("pred",))
    if isinstance(t_w, Rec) and isinstance(u_w, Rec):
        return (defeq(ctx, t_w.motive, u_w.motive,
                       path + ("motive",))
                and defeq(ctx, t_w.base, u_w.base, path + ("base",))
                and defeq(ctx, t_w.step, u_w.step, path + ("step",))
                and defeq(ctx, t_w.target, u_w.target,
                          path + ("target",)))
    return False


# --- Failure location -------------------------------------------------------
# Thin wrappers over the raising versions: same failure-dict shape as
# kernel.py's check_or_fail (path/kind/expected/actual/message).

def infer_or_fail(ctx, term, path=()):
    """Return (type, None) on success, (None, failure) on failure."""
    try:
        return infer(ctx, term, path), None
    except CheckError as e:
        return None, e.failure()


def check_or_fail(ctx, term, type_, path=()):
    """Return None if term has type_ under ctx, else the failure dict.
    Public entry: verifies the expected type is itself a type first."""
    try:
        _level_of_type(ctx, type_, path)
    except CheckError as e:
        return e.failure()
    try:
        check(ctx, term, type_, path)
        return None
    except CheckError as e:
        return e.failure()
