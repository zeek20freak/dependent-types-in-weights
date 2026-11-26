#!/usr/bin/env python3
"""
subst_circuit.py — de Bruijn substitution as an explicit fixed circuit.

Crux 2 of the DT weights kernel (2026-11-25). The previous builder
claimed substitution cannot reduce to fixed matrices (only a lookup
table over all (source, T) pairs would work — an explosion). That
claim is overstated. This file builds substitution as a structured
fixed construction — matrices + threshold gates, the same discipline
as the STLC equality circuit in weights_kernel.py (XNOR per entry,
AND over entries) — and measures its count instead of estimating it.

Operation: SUBST(S, k, s, T) — the de Bruijn substitution the
RefMachine's WALK engine performs (_walk_subst in
weights_kernel_dt.py), which mirrors kernel_dt.subst:
  - Walk S. Under a LAM/PI binder, the body recurses with (k+1, s+1);
    the binder type keeps (k, s).
  - At VAR j: j == k' -> graft T shifted by s'; j > k' -> VAR j-1;
    j < k' -> VAR j. (k', s') = (k + b, s + b) at that node, where b
    is the binder count on the path from the root.
  - The graft copies T with shift s' and cutoff 0: a VAR in T at
    binder-depth d within T shifts iff its index >= d (bound vars
    inside T do not shift).

The construction, per output slot q (the layout is fixed, so every
slot's path from the root is a compile-time constant):
  1. b[q]: the binder count on the path root -> q — a fixed sparse
     matrix M_B (right-turn LAM/PI ancestors) applied to S's kind
     one-hots.
  2. graft[a]: S[a] is VAR with index == k + b[a] — an integer
     equality (two thresholds + AND) per slot.
  3. A: S with the VAR adjustment (index - 1 where index > k + b[q])
     — one threshold per slot + a fixed shift-down permutation.
  4. B: the grafted content. For output slot q and graft ancestor a,
     the content is T at the relative slot rel(a, q), with its VAR
     indices shifted by s + b[a] where free in T (index >= the binder
     depth within T at that relative slot). The routing is a fixed
     index array REL (one entry per (q, ancestor) pair); the selection
     by graft(a) is the bilinear gating primitive (the STLC machine's
     action gating, free — the matrices being gated are counted).
  5. O = grafted ? B : A (gating).

Two structural facts keep it polynomial, not an explosion:
  - A VAR node is a leaf, so at most ONE graft ancestor lies on any
    root -> q path (no priority encoder is needed; the selection is a
    unique-select). Verified empirically in the test (asserted: at
    most one graft flag on any path, on every fuzz case).
  - Grafted regions below distinct VAR leaves are disjoint, so the
    overlay never conflicts.

Counting discipline (the STLC registry's: nonzero parameters):
  - Every fixed matrix contributes its nnz (sparse matrices are
    built and their .nnz used; index arrays count as the equivalent
    permutation's nnz: one per routed scalar).
  - Every threshold unit contributes its nonzero weights (fan-in
    entries + bias if nonzero). An equality [x == v] is two
    thresholds + one AND.
  - The bilinear gating (multiplying a vector by a 0/1 selector) is
    the STLC machine's gating primitive and is NOT counted as a
    weight — the matrices being gated are. (STLC counts the action
    matrices and the control matrix, not the gating multipliers.)
  - Pure reshaping / slicing / integer addition feeding a threshold
    layer is folded into that layer's fan-in, not counted separately.
The count is computed programmatically from the built objects, and
reported with a breakdown — it is a measurement, not an estimate.
"""

import numpy as np

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp = None

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import weights_kernel_dt as W

N = W.N_SLOTS                    # 2047
MAX_DEPTH = W.MAX_DEPTH          # 10
MAX_ANC = MAX_DEPTH + 1          # 11 (a slot's path has at most 11 nodes)
KINDS = W.KINDS
KIND = W.KIND
N_KINDS = W.N_KINDS              # 12
LEVELS = W.LEVELS                # 4
VAR_IDX = W.VAR_IDX              # 8
SLOT_DIM = W.SLOT_DIM            # 24
KIND_OFF = W.KIND_OFF            # 0
LEVEL_OFF = W.LEVEL_OFF          # 12
IDX_OFF = W.IDX_OFF              # 16
LAM = KIND["LAM"]
PI = KIND["PI"]
VAR = KIND["VAR"]


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------

class Counter:
    """Registry of the construction's cost, STLC discipline:
    nonzero parameters (matrix nnz + threshold-unit weights)."""

    def __init__(self):
        self.parts = {}   # name -> weights
        self.gates = {}   # name -> threshold units

    def add_matrix(self, name, nnz):
        self.parts[name] = self.parts.get(name, 0) + int(nnz)

    def add_thresholds(self, name, n_units, weights_per_unit):
        self.gates[name] = self.gates.get(name, 0) + int(n_units)
        self.parts[name] = (self.parts.get(name, 0)
                            + int(n_units) * int(weights_per_unit))

    def add_exact(self, name, weights, gates):
        """Record an exact (already computed) cost."""
        self.parts[name] = self.parts.get(name, 0) + int(weights)
        self.gates[name] = self.gates.get(name, 0) + int(gates)

    def total_weights(self):
        return sum(self.parts.values())

    def total_gates(self):
        return sum(self.gates.values())

    def report(self):
        lines = ["count breakdown (nonzero parameters / threshold units):"]
        for name in self.parts:
            lines.append("  %-28s %8d weights  %8d gates"
                         % (name, self.parts[name],
                            self.gates.get(name, 0)))
        lines.append("  %-28s %8d weights  %8d gates"
                     % ("TOTAL", self.total_weights(),
                        self.total_gates()))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Layout precomputation (fixed by the layout, not by the data)
# ---------------------------------------------------------------------------

def _depth_of_slot(q):
    d = 0
    while q > 0:
        q = (q - 1) // 2
        d += 1
    return d


def _ancestors(q):
    """Ancestors of slot q, root first, q last."""
    chain = []
    while q >= 0:
        chain.append(q)
        if q == 0:
            break
        q = (q - 1) // 2
    return list(reversed(chain))


def _is_right_turn(anc, q):
    """Is q in the right subtree of ancestor anc (anc != q)?"""
    # The child of anc on the path to q is the node after anc.
    chain = _ancestors(q)
    i = chain.index(anc)
    nxt = chain[i + 1]
    return nxt == 2 * anc + 2


def _relative_slot(anc, q):
    """The slot in a tree rooted at 0 reached by the path anc -> q
    (the turn sequence below anc). I.e. where q's content sits
    relative to a graft point at anc."""
    chain = _ancestors(q)
    i = chain.index(anc)
    rel = 0
    for node in chain[i + 1:]:
        # node is the left (2*parent+1) or right (2*parent+2) child
        # of the previous node; from rel's perspective the same turn.
        parent = (node - 1) // 2
        rel = 2 * rel + 1 if node == 2 * parent + 1 else 2 * rel + 2
    return rel


def build_layout(counter):
    """Build ANC, REL, M_B, M_ANC, SH_DOWN. Registers their nnz."""
    anc = np.full((N, MAX_ANC), -1, dtype=int)
    rel = np.full((N, MAX_ANC), -1, dtype=int)
    for q in range(N):
        chain = _ancestors(q)
        for j, a in enumerate(chain):
            anc[q, j] = a
            rel[q, j] = _relative_slot(a, q)
    # M_B[q, x] = 1 if x is a right-turn ancestor of q (x != q).
    rows, cols = [], []
    for q in range(N):
        for x in _ancestors(q)[:-1]:
            if _is_right_turn(x, q):
                rows.append(q)
                cols.append(x)
    m_b = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(N, N))
    counter.add_matrix("M_B binder-count", m_b.nnz)
    # M_ANC[q, a] = 1 if a is an ancestor of q (including q).
    rows, cols = [], []
    for q in range(N):
        for a in _ancestors(q):
            rows.append(q)
            cols.append(a)
    m_anc = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(N, N))
    counter.add_matrix("M_ANC ancestor", m_anc.nnz)
    # SH_DOWN: idx one-hot shift down by 1, block-diagonal over slots.
    # (N*8, N*8): one-hot w -> w-1 for w = 1..7 (w = 0 maps to zero).
    rows, cols = [], []
    for q in range(N):
        for w in range(1, VAR_IDX):
            rows.append(q * VAR_IDX + (w - 1))
            cols.append(q * VAR_IDX + w)
    sh_down = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(N * VAR_IDX, N * VAR_IDX))
    counter.add_matrix("SH_DOWN idx shift-down", sh_down.nnz)
    # REL routing: one entry per (q, ancestor) pair routes a whole slot
    # (kind 12 + level 4 + idx-int 1 + binder-depth-int 1 = 18 scalars).
    n_pairs = int((rel >= 0).sum())
    counter.add_matrix("REL routing kind+level (16/pair)",
                       n_pairs * (N_KINDS + LEVELS))
    counter.add_matrix("REL routing idx-int + depth-int (2/pair)",
                       n_pairs * 2)
    # Index-to-integer maps (one-hot -> integer): 8N + 11 + 11.
    counter.add_matrix("idx one-hot -> int (S and T)", 2 * N * VAR_IDX)
    counter.add_matrix("k/s one-hot -> int", 2 * 11)
    # Threshold-layer costs (fixed by the layout; recorded once here,
    # not per circuit call — the circuit is one fixed construction).
    # An equality [x == v] is three threshold units (two thresholds
    # + one AND); the weights below are the exact totals per slot.
    # graft equality [idx==k+b]: 12 + 12 + 3 = 27 weights, 3 gates.
    counter.add_exact("graft equality [idx==k+b]", 27 * N, 3 * N)
    # graft AND [is_var & eq]: 3 weights.
    counter.add_thresholds("graft AND [is_var & eq]", N, 3)
    # grafted OR over ancestors: fan-in (#ancestors of q) + bias (exact).
    counter.add_exact("grafted OR over ancestors",
                      int((anc >= 0).sum()) + N, N)
    # VAR-adjust threshold [idx>k+b]: fan-in 13.
    counter.add_thresholds("VAR-adjust threshold [idx>k+b]", N, 13)
    # graft-shift condition [j>=d]: fan-in 2 (bias 0).
    counter.add_thresholds("graft-shift condition [j>=d]", N, 2)
    # grafted idx one-hot encode: [out==w] is two thresholds + AND;
    # w = 0 costs 7 weights (both biases 0), w >= 1 costs 9.
    counter.add_exact("grafted idx one-hot encode (w=0)", 7 * N, 3 * N)
    counter.add_exact("grafted idx one-hot encode (w>=1)",
                      9 * N * (VAR_IDX - 1), 3 * N * (VAR_IDX - 1))
    # T present-bit [kind row nonempty]: fan-in 12 kind bits + bias.
    counter.add_thresholds("T present-bit [kind OR]", N, 13)
    return anc, rel, m_b, m_anc, sh_down


_LAYOUT = None


def layout(counter):
    global _LAYOUT
    if _LAYOUT is None:
        _LAYOUT = build_layout(counter)
    return _LAYOUT


# ---------------------------------------------------------------------------
# The circuit
# ---------------------------------------------------------------------------

def onehot_int(v, width):
    """Integer v (0..width-1) -> one-hot vector of length width."""
    oh = np.zeros(width)
    oh[int(v)] = 1.0
    return oh


def subst_circuit(S, T, k_onehot, s_onehot, counter=None):
    """SUBST(S, k, s, T) as a fixed circuit.

    S, T: (N, SLOT_DIM) one-hot tree vectors (kind 12, level 4,
    idx 8 per slot). k_onehot, s_onehot: one-hot over 0..10 (the
    target index and the shift at the top level). Returns O, the
    (N, SLOT_DIM) one-hot output tree. counter: a Counter to record
    the threshold-layer costs (layout matrices are recorded once,
    at build time, in the module-level layout counter).

    Every operation below is a fixed matrix multiply, a fixed
    threshold layer, or the bilinear gating primitive (multiplying
    a vector by a 0/1 selector). No branch depends on the contents
    of S or T: the only data-dependent values are the computed
    bits/vectors themselves.
    """
    if counter is None:
        counter = Counter()
    anc, rel, m_b, m_anc, sh_down = layout(counter)

    # Accept a flat TREE_DIM vector or an (N, SLOT_DIM) array.
    S = np.asarray(S, dtype=float).reshape(N, SLOT_DIM)
    T = np.asarray(T, dtype=float).reshape(N, SLOT_DIM)
    S_K = S[:, KIND_OFF:KIND_OFF + N_KINDS]
    S_L = S[:, LEVEL_OFF:LEVEL_OFF + LEVELS]
    S_I = S[:, IDX_OFF:IDX_OFF + VAR_IDX]
    T_K = T[:, KIND_OFF:KIND_OFF + N_KINDS]
    T_L = T[:, LEVEL_OFF:LEVEL_OFF + LEVELS]
    T_I = T[:, IDX_OFF:IDX_OFF + VAR_IDX]

    ar8 = np.arange(VAR_IDX, dtype=float)
    ar11 = np.arange(11, dtype=float)

    # --- 1. Binder counts b[q] on S, and d_T[r] on T. ---
    v_s = S_K[:, LAM] + S_K[:, PI]
    b = m_b @ v_s                       # (N,) ints 0..10
    v_t = T_K[:, LAM] + T_K[:, PI]
    d_t = m_b @ v_t                     # (N,) ints 0..10

    # --- 2. graft[a]: S[a] is VAR with index == k + b[a]. ---
    idx_int_s = S_I @ ar8               # (N,)
    k_int = float(k_onehot @ ar11)
    s_int = float(s_onehot @ ar11)
    diff = idx_int_s - k_int - b        # (N,)
    # [diff == 0] as two thresholds + AND, per slot (counted once
    # in build_layout).
    t1 = diff >= 0
    t2 = diff <= 0
    eq = t1 & t2
    is_var = S_K[:, VAR] > 0.5
    graft = is_var & eq                 # (N,) bool

    # Ancestor gathers (fixed index arrays; the selection by graft
    # is the gating primitive).
    anc_safe = np.maximum(anc, 0)
    anc_ok = anc >= 0
    graft_anc = graft[anc_safe] & anc_ok        # (N, MAX_ANC) bool
    grafted = (m_anc @ graft.astype(float)) >= 1  # (N,) bool
    # Threshold fan-in per slot: (#ancestors of q) + bias (exact;
    # counted once in build_layout).

    # --- 3. A: S with the VAR adjustment. ---
    # gt[q] = [idx_int_s[q] > k + b[q]]: one threshold per slot
    # (counted once in build_layout).
    gt = diff >= 1
    a_i_shifted = (sh_down @ S_I.reshape(-1)).reshape(N, VAR_IDX)
    a_i = np.where(gt[:, None], a_i_shifted, S_I)
    a_k, a_l = S_K, S_L

    # --- 4. B: the grafted content. ---
    rel_safe = np.maximum(rel, 0)
    # kind / level: gather T's fields at REL, gate by graft_anc, sum.
    b_k = (T_K[rel_safe] * graft_anc[:, :, None]).sum(axis=1)
    b_l = (T_L[rel_safe] * graft_anc[:, :, None]).sum(axis=1)
    # idx as an integer, and the binder depth within T at rel.
    idx_int_t = T_I @ ar8               # (N,)
    j_int = (idx_int_t[rel_safe] * graft_anc).sum(axis=1)   # (N,)
    d_rel = (d_t[rel_safe] * graft_anc).sum(axis=1)         # (N,)
    # amt[q] = s + b[a*] (the shift at the graft ancestor).
    b_anc = b[anc_safe]                 # (N, MAX_ANC)
    amt = s_int * grafted + (graft_anc * b_anc).sum(axis=1)
    # A VAR in T shifts iff its index >= its binder depth within T.
    # cond[q] = [j_int[q] >= d_rel[q]]: one threshold per slot
    # (counted once in build_layout).
    cond = j_int >= d_rel
    out_idx = j_int + amt * cond        # (N,) ints
    # One-hot encode out_idx into 8 bits, gated by whether the
    # grafted node is actually present in T: below a graft point but
    # beyond T's own depth the slot is EMPTY (all fields zero), not
    # a node with index 0. present_t[r] = [T's kind row at r nonempty].
    present_t = (T_K.sum(axis=1) > 0.5).astype(float)
    present_rel = (present_t[rel_safe] * graft_anc).sum(axis=1)
    b_i = ((out_idx[:, None] == ar8[None, :]).astype(float)
           * present_rel[:, None])

    # --- 5. O = grafted ? B : A (gating). ---
    o_k = np.where(grafted[:, None], b_k, a_k)
    o_l = np.where(grafted[:, None], b_l, a_l)
    o_i = np.where(grafted[:, None], b_i, a_i)
    O = np.concatenate([o_k, o_l, o_i], axis=1)
    return O


# ---------------------------------------------------------------------------
# Independent de Bruijn substitution (the direct spec of the circuit,
# on tuples — used as ground truth for the fuzz; validated against
# kernel_dt.subst on named terms in the test file).
# ---------------------------------------------------------------------------

def shift_db(t, d, cutoff=0):
    """Shift free vars >= cutoff by d (de Bruijn, tuple form)."""
    tag = t[0]
    if tag == "var":
        return ("var", t[1] + d if t[1] >= cutoff else t[1])
    if tag in ("lam", "pi"):
        return (tag, shift_db(t[1], d, cutoff),
                shift_db(t[2], d, cutoff + 1))
    if tag == "app":
        return ("app", shift_db(t[1], d, cutoff),
                shift_db(t[2], d, cutoff))
    if tag == "succ":
        return ("succ", shift_db(t[1], d, cutoff))
    if tag == "rec":
        return ("rec", shift_db(t[1], d, cutoff),
                shift_db(t[2], d, cutoff), shift_db(t[3], d, cutoff),
                shift_db(t[4], d, cutoff))
    return t  # type, nat, zero


def subst_db(t, k, s, u):
    """De Bruijn substitution: t with Var(k) replaced by u (shifted
    by s). Mirrors _walk_subst/_walk_copy in weights_kernel_dt.py."""
    tag = t[0]
    if tag == "var":
        j = t[1]
        if j == k:
            return shift_db(u, s, 0)
        if j > k:
            return ("var", j - 1)
        return ("var", j)
    if tag in ("lam", "pi"):
        return (tag, subst_db(t[1], k, s, u),
                subst_db(t[2], k + 1, s + 1, u))
    if tag == "app":
        return ("app", subst_db(t[1], k, s, u),
                subst_db(t[2], k, s, u))
    if tag == "succ":
        return ("succ", subst_db(t[1], k, s, u))
    if tag == "rec":
        return ("rec", subst_db(t[1], k, s, u),
                subst_db(t[2], k, s, u), subst_db(t[3], k, s, u),
                subst_db(t[4], k, s, u))
    return t  # type, nat, zero
