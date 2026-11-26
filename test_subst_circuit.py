#!/usr/bin/env python3
"""Test the substitution circuit (subst_circuit.py) against the plain
kernel's subst (kernel_dt.py) and the independent de Bruijn subst
(subst_db in subst_circuit.py).

Three layers, so a failure localises:
  (A) subst_db vs kernel_dt.subst on named terms (validates subst_db).
  (B) circuit vs subst_db on random de Bruijn terms, general (k, s)
      (validates the circuit against the direct de Bruijn spec).
  (C) circuit vs kernel_dt.subst on named suite terms + named fuzz
      (the headline: the circuit matches the sealed plain kernel).

Named -> de Bruijn conversion uses TWO contexts, as the machine
does: the source t is converted with the target variable x at free
position 0 (so the circuit's k = 0 targets it); the replacement u
and the expected result are converted with x REMOVED (the outer
context, where x is bound, not free). Converting u with x included
would misnumber its free variables by one — that is a harness bug,
not a circuit bug, and it is the reason for the two lists.

Also asserted on every circuit run: the output one-hot is valid
(each field row sums to exactly 1), and at most one graft flag lies
on any root -> slot path (the structural fact the construction
relies on — a VAR is a leaf, so no priority encoder is needed).
"""
import random
import sys

import numpy as np

import kernel_dt as kdt
import weights_kernel_dt as W
import subst_circuit as SC

N = W.N_SLOTS


# ---------------------------------------------------------------------------
# Named -> de Bruijn with an explicit free-variable list
# ---------------------------------------------------------------------------

def named_to_db_free(term, free_list, env=()):
    """kernel_dt class term -> de Bruijn tuple. env: binder names
    (innermost last). A free variable gets index len(env) +
    free_list.index(name). free_list must contain every free var."""
    if isinstance(term, kdt.Var):
        name = term.name
        if name in env:
            return ("var", list(reversed(env)).index(name))
        return ("var", len(env) + free_list.index(name))
    if isinstance(term, kdt.Lam):
        return ("lam", named_to_db_free(term.var_type, free_list, env),
                named_to_db_free(term.body, free_list, env + (term.var,)))
    if isinstance(term, kdt.Pi):
        return ("pi", named_to_db_free(term.var_type, free_list, env),
                named_to_db_free(term.body, free_list, env + (term.var,)))
    if isinstance(term, kdt.App):
        return ("app", named_to_db_free(term.func, free_list, env),
                named_to_db_free(term.arg, free_list, env))
    if isinstance(term, kdt.Type):
        return ("type", term.level)
    if isinstance(term, kdt.Nat):
        return ("nat",)
    if isinstance(term, kdt.Zero):
        return ("zero",)
    if isinstance(term, kdt.Succ):
        return ("succ", named_to_db_free(term.pred, free_list, env))
    if isinstance(term, kdt.Rec):
        return ("rec", named_to_db_free(term.motive, free_list, env),
                named_to_db_free(term.base, free_list, env),
                named_to_db_free(term.step, free_list, env),
                named_to_db_free(term.target, free_list, env))
    raise ValueError("unknown term %r" % (term,))


def db_max_index(t):
    tag = t[0]
    if tag == "var":
        return t[1]
    if tag in ("lam", "pi", "app"):
        return max(db_max_index(t[1]), db_max_index(t[2]))
    if tag == "succ":
        return db_max_index(t[1])
    if tag == "rec":
        return max(db_max_index(t[1]), db_max_index(t[2]),
                   db_max_index(t[3]), db_max_index(t[4]))
    return -1


def db_depth(t):
    return W.tree_depth_db(t)


# ---------------------------------------------------------------------------
# Circuit driver
# ---------------------------------------------------------------------------

def circuit_subst(t_db, k, s, u_db, counter):
    """Run the circuit on de Bruijn tuples; return the output tuple.
    Also returns the raw output vector and the graft flags, for the
    structural assertions."""
    S = W.tree_to_vec(W.encode_db(t_db))
    T = W.tree_to_vec(W.encode_db(u_db))
    O = SC.subst_circuit(S, T, SC.onehot_int(k, 11), SC.onehot_int(s, 11),
                         counter)
    return W.decode_tree(W.vec_to_tree(O.reshape(-1))), O


def check_onehot_valid(O):
    """Every slot's kind/level/idx field must be exactly one-hot
    (rows that are all-EMPTY are the exception: all three fields
    zero). Returns (bad_slots, bad_fields)."""
    bad = []
    for q in range(N):
        k = O[q, 0:12]
        lv = O[q, 12:16]
        ix = O[q, 16:24]
        ks, ls, xs = k.sum(), lv.sum(), ix.sum()
        if ks == 0 and ls == 0 and xs == 0:
            continue  # EMPTY slot
        if not (ks == 1 and ls == 1 and xs == 1):
            bad.append((q, ks, ls, xs))
    return bad


def max_grafts_on_path(S, k, counter):
    """Recompute the graft flags the way the circuit does, and return
    the maximum number of graft flags on any root -> slot path.
    Must be <= 1 (VAR is a leaf)."""
    anc, rel, m_b, m_anc, sh_down = SC.layout(counter)
    S = np.asarray(S, dtype=float).reshape(N, SC.SLOT_DIM)
    S_K = S[:, 0:12]
    S_I = S[:, 16:24]
    ar8 = np.arange(8, dtype=float)
    ar11 = np.arange(11, dtype=float)
    v_s = S_K[:, SC.LAM] + S_K[:, SC.PI]
    b = m_b @ v_s
    idx_int_s = S_I @ ar8
    k_int = float(SC.onehot_int(k, 11) @ ar11)
    diff = idx_int_s - k_int - b
    eq = (diff >= 0) & (diff <= 0)
    graft = (S_K[:, SC.VAR] > 0.5) & eq
    worst = 0
    for q in range(N):
        chain = [a for a in anc[q] if a >= 0]
        worst = max(worst, sum(1 for a in chain if graft[a]))
    return worst


# ---------------------------------------------------------------------------
# (A) subst_db vs kernel_dt.subst (named)
# ---------------------------------------------------------------------------

def named_case(t, x, u):
    """One named case. Returns (ok, detail)."""
    free_outer = sorted((kdt.free_vars(t) | kdt.free_vars(u)) - {x})
    free_t = [x] + free_outer
    t_db = named_to_db_free(t, free_t)
    u_db = named_to_db_free(u, free_outer)
    expected = kdt.subst(t, x, u)
    expected_db = named_to_db_free(expected, free_outer)
    got_db = SC.subst_db(t_db, 0, 0, u_db)
    ok = got_db == expected_db
    return ok, (t, x, u, expected_db, got_db)


# ---------------------------------------------------------------------------
# Random term generators
# ---------------------------------------------------------------------------

def rand_db_term(rng, depth, max_idx, free_names=0):
    """Random de Bruijn tuple. free_names: how many free variables
    (indices >= the binder depth) may appear — kept small so the
    subst results stay in the layout."""
    if depth <= 0:
        r = rng.random()
        if r < 0.45:
            # a variable: bound (0..depth-ish) or free (a bit above)
            return ("var", rng.randint(0, max_idx))
        if r < 0.6:
            return ("type", rng.randint(0, 1))
        if r < 0.75:
            return ("nat",)
        return ("zero",)
    r = rng.random()
    if r < 0.2:
        return ("lam", rand_db_term(rng, depth - 1, max_idx),
                rand_db_term(rng, depth - 1, max_idx))
    if r < 0.35:
        return ("pi", rand_db_term(rng, depth - 1, max_idx),
                rand_db_term(rng, depth - 1, max_idx))
    if r < 0.55:
        return ("app", rand_db_term(rng, depth - 1, max_idx),
                rand_db_term(rng, depth - 1, max_idx))
    if r < 0.65:
        return ("succ", rand_db_term(rng, depth - 1, max_idx))
    if r < 0.8:
        return ("rec", rand_db_term(rng, depth - 1, max_idx),
                rand_db_term(rng, depth - 1, max_idx),
                rand_db_term(rng, depth - 1, max_idx),
                rand_db_term(rng, depth - 1, max_idx))
    return rand_db_term(rng, 0, max_idx)


def rand_named_term(rng, depth, names):
    """Random named kernel_dt term over the given free names."""
    if depth <= 0:
        r = rng.random()
        if r < 0.5:
            return kdt.Var(rng.choice(names))
        if r < 0.65:
            return kdt.Type(rng.randint(0, 1))
        if r < 0.8:
            return kdt.Nat()
        return kdt.Zero()
    r = rng.random()
    nm = "v%d" % rng.randint(0, 3)
    if r < 0.2:
        return kdt.Lam(nm, rand_named_term(rng, depth - 1, names),
                       rand_named_term(rng, depth - 1, names + [nm]))
    if r < 0.35:
        return kdt.Pi(nm, rand_named_term(rng, depth - 1, names),
                      rand_named_term(rng, depth - 1, names + [nm]))
    if r < 0.55:
        return kdt.App(rand_named_term(rng, depth - 1, names),
                       rand_named_term(rng, depth - 1, names))
    if r < 0.65:
        return kdt.Succ(rand_named_term(rng, depth - 1, names))
    if r < 0.8:
        return kdt.Rec(rand_named_term(rng, depth - 1, names),
                       rand_named_term(rng, depth - 1, names),
                       rand_named_term(rng, depth - 1, names),
                       rand_named_term(rng, depth - 1, names))
    return rand_named_term(rng, 0, names)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = random.Random(20261125)
    counter = SC.Counter()
    # Build the layout once (registers the layout matrices' nnz).
    SC.layout(counter)
    failures = []
    skipped = 0

    # (A) subst_db vs kernel_dt.subst, on the suite terms.
    n_a = 0
    check_cases, infer_cases, defeq_cases = W.build_cases()
    suite_terms = []
    for name, term, ty, ctx, want in check_cases:
        suite_terms.append((name, term))
        if ty is not None:
            suite_terms.append((name + " :type", ty))
    for name, term, ctx, want in infer_cases:
        suite_terms.append((name, term))
    for name, ctx, t, u, want in defeq_cases:
        suite_terms.append((name + " :l", t))
        suite_terms.append((name + " :r", u))
    small_us = [kdt.Zero(), kdt.Nat(), kdt.Type(0),
                kdt.Var("w"), kdt.Succ(kdt.Zero()),
                kdt.Lam("q", kdt.Nat(), kdt.Var("q"))]
    for name, term in suite_terms:
        fvs = sorted(kdt.free_vars(term))
        for x in fvs:
            for u in small_us:
                if x in kdt.free_vars(u):
                    continue
                ok, detail = named_case(term, x, u)
                n_a += 1
                if not ok:
                    failures.append(("A subst_db vs kernel_dt.subst",
                                     name, x, detail))
    print("A: subst_db vs kernel_dt.subst on suite terms: %d cases, "
          "%d failures" % (n_a, len(failures)))

    # (B) circuit vs subst_db, de Bruijn fuzz, general (k, s).
    n_b = 0
    worst_grafts = 0
    for i in range(400):
        t = rand_db_term(rng, 4, 3)
        u = rand_db_term(rng, 3, 2)
        k = rng.randint(0, 3)
        s = rng.randint(0, 3)
        try:
            expected = SC.subst_db(t, k, s, u)
        except Exception as e:  # pragma: no cover
            failures.append(("B subst_db raised", i, e))
            continue
        if db_depth(expected) > W.MAX_DEPTH or db_max_index(expected) > 7:
            skipped += 1
            continue
        if db_depth(t) > W.MAX_DEPTH or db_max_index(t) > 7:
            skipped += 1
            continue
        if db_depth(u) > W.MAX_DEPTH or db_max_index(u) > 7:
            skipped += 1
            continue
        got, O = circuit_subst(t, k, s, u, counter)
        n_b += 1
        if got != expected:
            failures.append(("B circuit vs subst_db", i, t, k, s, u,
                             expected, got))
        bad = check_onehot_valid(O)
        if bad:
            failures.append(("B one-hot invalid", i, bad[:5]))
        wg = max_grafts_on_path(W.tree_to_vec(W.encode_db(t)), k, counter)
        worst_grafts = max(worst_grafts, wg)
        if wg > 1:
            failures.append(("B >1 graft on a path", i, wg))
    print("B: circuit vs subst_db fuzz: %d cases, %d skipped "
          "(out of layout), worst grafts/path=%d"
          % (n_b, skipped, worst_grafts))

    # (C) circuit vs kernel_dt.subst, named suite terms + named fuzz.
    n_c = 0
    for name, term in suite_terms:
        fvs = sorted(kdt.free_vars(term))
        for x in fvs:
            for u in small_us:
                if x in kdt.free_vars(u):
                    continue
                free_outer = sorted(
                    (kdt.free_vars(term) | kdt.free_vars(u)) - {x})
                free_t = [x] + free_outer
                try:
                    t_db = named_to_db_free(term, free_t)
                    u_db = named_to_db_free(u, free_outer)
                    expected_db = named_to_db_free(
                        kdt.subst(term, x, u), free_outer)
                except ValueError:
                    continue
                if (db_depth(t_db) > W.MAX_DEPTH
                        or db_max_index(t_db) > 7
                        or db_depth(u_db) > W.MAX_DEPTH
                        or db_max_index(u_db) > 7
                        or db_depth(expected_db) > W.MAX_DEPTH
                        or db_max_index(expected_db) > 7):
                    skipped += 1
                    continue
                got, O = circuit_subst(t_db, 0, 0, u_db, counter)
                n_c += 1
                if got != expected_db:
                    failures.append(("C circuit vs kernel_dt.subst",
                                     name, x, u, expected_db, got))
                bad = check_onehot_valid(O)
                if bad:
                    failures.append(("C one-hot invalid", name, bad[:5]))
    # named fuzz
    for i in range(150):
        t = rand_named_term(rng, 3, ["x", "y"])
        fvs = sorted(kdt.free_vars(t))
        if not fvs:
            continue
        x = rng.choice(fvs)
        u = rand_named_term(rng, 2, ["y", "w"])
        if x in kdt.free_vars(u):
            continue
        free_outer = sorted((kdt.free_vars(t) | kdt.free_vars(u)) - {x})
        free_t = [x] + free_outer
        try:
            t_db = named_to_db_free(t, free_t)
            u_db = named_to_db_free(u, free_outer)
            expected_db = named_to_db_free(kdt.subst(t, x, u), free_outer)
        except ValueError:
            continue
        if (db_depth(t_db) > W.MAX_DEPTH or db_max_index(t_db) > 7
                or db_depth(u_db) > W.MAX_DEPTH
                or db_max_index(u_db) > 7
                or db_depth(expected_db) > W.MAX_DEPTH
                or db_max_index(expected_db) > 7):
            skipped += 1
            continue
        got, O = circuit_subst(t_db, 0, 0, u_db, counter)
        n_c += 1
        if got != expected_db:
            failures.append(("C circuit vs kernel_dt.subst (fuzz)",
                             i, t, x, u, expected_db, got))
        bad = check_onehot_valid(O)
        if bad:
            failures.append(("C one-hot invalid (fuzz)", i, bad[:5]))
    print("C: circuit vs kernel_dt.subst: %d cases, %d skipped total"
          % (n_c, skipped))

    print()
    print(counter.report())
    print()
    if failures:
        print("FAILURES: %d" % len(failures))
        for f in failures[:10]:
            print("  ", f)
        return 1
    print("ALL SUBST-CIRCUIT TESTS PASSED (A=%d, B=%d, C=%d)"
          % (n_a, n_b, n_c))
    return 0


if __name__ == "__main__":
    sys.exit(main())
