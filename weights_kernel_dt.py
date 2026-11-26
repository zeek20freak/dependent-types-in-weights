#!/usr/bin/env python3
"""
weights_kernel_dt.py — the sealed dependent-types checker as explicit matrices.

Rung 1 (Pi + universes + conversion) + rung 2 (Nat + eliminator), from
kernel_dt.py, compiled into a matrix machine. The bet: the gate as matrices,
compiled in, not learned in.

Design: WEIGHTS.md, "The DT weights kernel — design v0.1 (2026-11-25)".
Layout: ONE fixed-layout binary tree, depth 10 -> 2047 slots. Node kinds:
LAM, PI, APP, VAR (de Bruijn), TYPE (level field one-hot 0..3), NAT, ZERO,
SUCC, REC, REC3, REC2, EMPTY. Rec is a 3-node chain (REC holds motive +
REC3; REC3 holds base + REC2; REC2 holds step + target; depth +2).
Per slot: kind one-hot (12) + level one-hot (4) + var-index one-hot (8) =
24 dims. Tree vector dim = 2047 * 24 = 49,128.

Two machines share the design:
  - RefMachine: plain Python on slot-array lists (the "plain logic").
  - MatrixMachine: the same steps as scipy.sparse CSR matrices.
Lockstep: decode(MatrixMachine state) == RefMachine state after every step.

Bounds (measured from the sealed suites, DT 20 + Nat 21 + examples + REPL):
  max tree depth 10, max ctx entries 8 (dynamic), max universe level 1,
  max de Bruijn index ~4 (verify during build; index field 0..7).
Control depth and max steps are measured from the RefMachine run and fixed
afterwards (see measure section at the bottom).

The count: nonzero entries of every matrix in the registry, same discipline
as STLC's 4,849.

No kernel_dt.py / RULES_DT.md / WEIGHTS.md files are modified.
"""

import json
import math
import os
import sys

import numpy as np

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp = None

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import kernel_dt as kdt  # noqa: E402  (the sealed plain kernel; read-only here)

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

MAX_DEPTH = 10                      # measured: deepest suite term is depth 10
N_SLOTS = 2 ** (MAX_DEPTH + 1) - 1  # 2047

KINDS = ["LAM", "PI", "APP", "VAR", "TYPE", "NAT",
         "ZERO", "SUCC", "REC", "REC3", "REC2", "EMPTY"]
KIND = {k: i for i, k in enumerate(KINDS)}
N_KINDS = len(KINDS)

LEVELS = 4          # one-hot over 0..3 (measured max level is 1; 3 is headroom)
VAR_IDX = 8         # one-hot over 0..7 (measured max de Bruijn index ~4)

SLOT_DIM = N_KINDS + LEVELS + VAR_IDX   # 24
TREE_DIM = N_SLOTS * SLOT_DIM           # 49,128

# per-slot field offsets
KIND_OFF = 0
LEVEL_OFF = N_KINDS
IDX_OFF = N_KINDS + LEVELS

# ---------------------------------------------------------------------------
# Sources (where a term reference lives) — a (sel, ptr) pair
# ---------------------------------------------------------------------------
# sel: INPUT (the constant input term) or a register. Registers:
#   RES  — result tree (inferred type / whnf output / compare result)
#   WORK — whnf / subst working tree
#   AUX  — auxiliary tree (expected type in CHECK, right side in COMPARE)
#   R0..R5 — general tree registers (WALK destination, frame-owned trees)
SOURCES = ["INPUT", "RES", "WORK", "AUX", "R0", "R1", "R2", "R3", "R4", "R5"]
SRC = {s: i for i, s in enumerate(SOURCES)}
N_SRC = len(SOURCES)
INPUT_SRC = SRC["INPUT"]
RES_SRC = SRC["RES"]
WORK_SRC = SRC["WORK"]
AUX_SRC = SRC["AUX"]
REG_SRC = SRC["R0"]  # R0..R5 are SRC indices REG_SRC..REG_SRC+5

# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
MODES = ["INFER", "CHECK", "WHNF", "COMPARE", "WALK", "RETURN", "HALT", "ERROR"]
MODE = {m: i for i, m in enumerate(MODES)}
N_MODES = len(MODES)

# ---------------------------------------------------------------------------
# Frame kinds (continuations on the control stack)
# Each frame: (kind, ref1, ref2, ref3, small1, small2)
#   refs are (sel, ptr) pairs, or None. small fields are ints.
# ---------------------------------------------------------------------------
# INFER frames
#   LAM_TY    (ty_ref, body_ref)          after infer(binder ty); whnf it
#   LAM_TY_W  (ty_ref, body_ref)          after whnf(binder ty ty); must be Type
#   LAM_BODY  (ty_tree_ref)               after infer(body); build Pi
#   PI_DOM    (ty_ref, body_ref)          after infer(domain); whnf it
#   PI_DOM_W  (ty_ref, body_ref, level)   after whnf; infer codomain
#   PI_BODY   (level)                     after infer(codomain); whnf it
#   PI_BODY_W (level)                     after whnf; result Type(max)
#   APP_F     (arg_ref)                   after infer(func); whnf func type
#   APP_F_W   (arg_ref)                   after whnf; must be Pi; check arg
#   APP_A     (cod_ref, arg_ref)          after check(arg, dom); subst cod
#   APP_SUBST ()                          after subst(cod, arg); result is it
#   REC_M     (base_ref, step_ref, tgt_ref) after infer(motive); validate it
#   REC_M_W   (base_ref, step_ref, tgt_ref, motive_ref) after whnf(motive ty)
#   REC_BASE  (step_ref, tgt_ref, motive_ref, P_ty) after check(base, P 0)
#   REC_STEP  (tgt_ref, motive_ref)       after check(step, step-type)
#   REC_TGT   (motive_ref)                after check(target, Nat)
#   SUCC_T    ()                          after check(pred, Nat); result Nat
# CHECK frames
#   CHK_LAM_D  (body_ref, cod_ref)        after defeq(binder ty, dom)
#   CHK_LAM_B  ()                         after check(body, cod); done
#   CHK_INF    (expected_ref)             after infer(term); defeq vs expected
#   CHK_DEFEQ  ()                         after defeq; accept iff true
# WHNF frames
#   WHNF_APP   (arg_ref)                  after whnf(func); beta or rebuild
#   WHNF_APP_S ()                         after subst(body, arg); loop whnf
#   WHNF_REC   (motive_ref, base_ref, step_ref) after whnf(target)
#   WHNF_REC_S ()                         after subst(step-app construction); loop
# COMPARE frames
#   CMP_W_L    (l_ref, r_ref)              after whnf(left); whnf right
#   CMP_W_R    (l_whnf_ref, r_ref)         after whnf(right); dispatch
#   CMP_LAM_T  (l_body_ref, r_body_ref)    after compare(binder tys); bodies
#   CMP_LAM_B  ()                          after compare(bodies); done
#   CMP_PI_D   (l_cod_ref, r_cod_ref)      after compare(domains); codomains
#   CMP_PI_C   ()                          after compare(codomains); done
#   CMP_APP_L  (l_arg_ref, r_arg_ref)      after compare(funcs); compare args
#   CMP_APP_R  ()                          after compare(args); done
#   CMP_SUCC   ()                          after compare(preds); done
#   CMP_REC_M  (r_ref)                     after compare(motive); compare base
#   CMP_REC_B  (r_ref)                     after compare(base); compare step
#   CMP_REC_S  (r_ref)                     after compare(step); compare target
#   CMP_REC_T  ()                          after compare(target); done
#   CMP_ETA    (lam_ref, neutral_ref)      after infer(neutral type); whnf it
#   CMP_ETA_W  (lam_ref, neutral_ref)      after whnf; must be Pi; defeq dom
#   CMP_ETA_D  (lam_ref, neutral_ref)      after defeq(binder, dom); bodies
#   CMP_ETA_B  (lam_body_ref, app_reg)     after building App(u', Var 0)
#   CMP_ETA_R  ()                          after compare(bodies); done
#   REC_M_D    (base, step, tgt) + mot     after defeq(mot dom, Nat)
#   REC_M_C    (base, step, tgt) + mot     after whnf(mot cod); must be Type
#   REC_BASE_B (base, step, tgt) + mot     after building App(mot, Zero)
#   REC_BASE   (step, tgt) + mot           after check(base, App(mot, Zero))
#   REC_STEP_B (step, tgt) + mot           after building the step type
#   REC_STEP   (tgt) + mot                 after check(step, step-type)
#   REC_TGT    () + mot                    after check(target, Nat)
#   REC_RES    () + mot                    after building App(mot, target)
#   ENTRY_VALID (term_ref)                 after infer(expected type); whnf it
#   ENTRY_VALID_W (term_ref)               after whnf; must be Type; check
#   ENTRY_DEFEQ ()                         after defeq entry; result is bool
FRAME_KINDS = [
    "LAM_TY", "LAM_TY_W", "LAM_BODY", "PI_DOM", "PI_DOM_W", "PI_BODY",
    "PI_BODY_W", "APP_F", "APP_F_W", "APP_A", "APP_SUBST",
    "REC_M", "REC_M_W", "REC_M_D", "REC_M_C", "REC_BASE_B", "REC_BASE",
    "REC_STEP_B", "REC_STEP", "REC_TGT", "REC_RES", "SUCC_T",
    "CHK_W", "CHK_LAM_D", "CHK_LAM_B", "CHK_INF", "CHK_DEFEQ",
    "WHNF_APP", "WHNF_APP_S", "WHNF_REC", "WHNF_REC_S",
    "CMP_W_L", "CMP_W_R", "CMP_LAM_T", "CMP_LAM_B", "CMP_PI_D", "CMP_PI_C",
    "CMP_APP_L", "CMP_APP_R", "CMP_SUCC", "CMP_REC_M", "CMP_REC_B",
    "CMP_REC_S", "CMP_REC_T", "CMP_ETA", "CMP_ETA_W", "CMP_ETA_D",
    "CMP_ETA_B", "CMP_ETA_R",
    "WALK_DONE", "ENTRY_VALID", "ENTRY_VALID_W", "ENTRY_DEFEQ",
]
FRAME_KIND = {k: i for i, k in enumerate(FRAME_KINDS)}
N_FRAME_KINDS = len(FRAME_KINDS)

# Path segment each frame kind descends on (for failure paths). A frame
# that continues a sub-computation inside a named subterm carries that
# subterm's name; the failure path is the concatenation of the segments
# on the frame stack. Frames that do not descend carry None.
FRAME_SEG = {
    "LAM_TY": "binder", "LAM_TY_W": "binder", "LAM_BODY": "body",
    "PI_DOM": "domain", "PI_DOM_W": "domain", "PI_BODY": "body",
    "PI_BODY_W": "body",
    "APP_F": "func", "APP_F_W": "func", "APP_A": "arg",
    "SUCC_T": "pred",
    "REC_M": "motive", "REC_M_W": "motive", "REC_M_D": "motive",
    "REC_M_C": "motive", "REC_BASE": "base", "REC_STEP": "step",
    "REC_TGT": "target",
    "CHK_LAM_D": "binder", "CHK_LAM_B": "body",
    "WHNF_APP": "func", "WHNF_REC": "target",
    "CMP_LAM_T": "binder", "CMP_LAM_B": "body",
    "CMP_PI_D": "domain", "CMP_PI_C": "body",
    "CMP_APP_L": "func", "CMP_APP_R": "arg", "CMP_SUCC": "pred",
    "CMP_REC_M": "motive", "CMP_REC_B": "base", "CMP_REC_S": "step",
    "CMP_REC_T": "target", "CMP_ETA_D": "binder", "CMP_ETA_B": "body",
}
FAIL_KINDS = ["none", "mismatch", "not-pi", "not-a-type", "unbound",
              "level", "fuel", "index"]
FAIL_KIND = {k: i for i, k in enumerate(FAIL_KINDS)}
N_FAIL_KINDS = len(FAIL_KINDS)

# Failure-path segment names (paths are up to 3 segments deep)
PATH_SEGS = ["", "binder", "body", "func", "arg", "domain", "motive",
             "base", "step", "target", "pred", "level"]
PATH_SEG = {s: i for i, s in enumerate(PATH_SEGS)}
N_PATH_SEGS = len(PATH_SEGS)
MAX_PATH_LEN = 3

# ---------------------------------------------------------------------------
# Slot-array tree representation (shared by RefMachine and MatrixMachine)
# A tree is a list of N_SLOTS nodes; node = (kind:str, level:int, idx:int).
# EMPTY nodes are ("EMPTY", 0, 0). level is meaningful for TYPE nodes,
# idx is meaningful for VAR nodes.
# ---------------------------------------------------------------------------

EMPTY_NODE = ("EMPTY", 0, 0)


def empty_tree():
    return [EMPTY_NODE] * N_SLOTS


def left_ptr(p):
    return 2 * p + 1


def right_ptr(p):
    return 2 * p + 2


def parent_ptr(p):
    return (p - 1) // 2 if p > 0 else -1


def slot_ok(p):
    return 0 <= p < N_SLOTS


def read_node(tree, p):
    return tree[p]


def write_node(tree, p, kind, level=0, idx=0):
    tree[p] = (kind, level, idx)


# ---------------------------------------------------------------------------
# Named kernel terms -> de Bruijn tuples -> slot arrays
# ---------------------------------------------------------------------------
# kernel_dt terms use named variables (strings). We convert to de Bruijn by
# walking the term with a binder stack (innermost binder = index 0). This
# conversion is in the HARNESS (not the machine): the machine only ever sees
# de Bruijn slot arrays. The conversion mirrors kernel_dt_independent's
# de Bruijn handling (names are only a surface syntax).

def named_to_debruijn(term, env=()):
    """env: tuple of binder names, innermost last."""
    t = term[0]
    if t == "var":
        name = term[1]
        # innermost binder is env[-1] -> de Bruijn index 0. Use the LAST
        # occurrence in env (shadowing): the innermost binder of that name.
        if name in env:
            return ("var", list(reversed(env)).index(name))
        # free variable: keep as a large index past the binders (the suites
        # have no free vars in checked terms; free vars appear only as
        # placeholder names inside types under binders, handled by callers)
        raise ValueError("free variable %r under env %r" % (name, env))
    if t == "lam":
        _, var, ty, body = term
        return ("lam", named_to_debruijn(ty, env),
                named_to_debruijn(body, env + (var,)))
    if t == "pi":
        _, var, ty, body = term
        return ("pi", named_to_debruijn(ty, env),
                named_to_debruijn(body, env + (var,)))
    if t == "app":
        return ("app", named_to_debruijn(term[1], env),
                named_to_debruijn(term[2], env))
    if t == "type":
        return ("type", term[1])
    if t == "nat":
        return ("nat",)
    if t == "zero":
        return ("zero",)
    if t == "succ":
        return ("succ", named_to_debruijn(term[1], env))
    if t == "rec":
        _, P, z, s, n = term
        return ("rec", named_to_debruijn(P, env), named_to_debruijn(z, env),
                named_to_debruijn(s, env), named_to_debruijn(n, env))
    raise ValueError("unknown term %r" % (term,))


def debruijn_to_named(term, env=()):
    """Inverse (for debugging / harness comparison). env: binder names."""
    t = term[0]
    if t == "var":
        n = term[1]
        if n < len(env):
            return ("var", env[len(env) - 1 - n])
        return ("var", "?free%d" % n)
    if t == "lam":
        _, ty, body = term
        name = "x%d" % len(env)
        return ("lam", name, debruijn_to_named(ty, env),
                debruijn_to_named(body, env + (name,)))
    if t == "pi":
        _, ty, body = term
        name = "x%d" % len(env)
        return ("pi", name, debruijn_to_named(ty, env),
                debruijn_to_named(body, env + (name,)))
    if t == "app":
        return ("app", debruijn_to_named(term[1], env),
                debruijn_to_named(term[2], env))
    if t == "type":
        return ("type", term[1])
    if t == "nat":
        return ("nat",)
    if t == "zero":
        return ("zero",)
    if t == "succ":
        return ("succ", debruijn_to_named(term[1], env))
    if t == "rec":
        return ("rec", debruijn_to_named(term[1], env),
                debruijn_to_named(term[2], env),
                debruijn_to_named(term[3], env),
                debruijn_to_named(term[4], env))
    raise ValueError("unknown de Bruijn term %r" % (term,))


def tree_depth_db(term):
    """Depth of a de Bruijn tuple under the binary layout (Rec chain +2)."""
    t = term[0]
    if t in ("var", "type", "nat", "zero"):
        return 0
    if t == "lam":
        return 1 + max(tree_depth_db(term[1]), tree_depth_db(term[2]))
    if t == "pi":
        return 1 + max(tree_depth_db(term[1]), tree_depth_db(term[2]))
    if t == "app":
        return 1 + max(tree_depth_db(term[1]), tree_depth_db(term[2]))
    if t == "succ":
        return 1 + tree_depth_db(term[1])
    if t == "rec":
        # REC holds motive + REC3; REC3 holds base + REC2; REC2 holds step + target
        return max(1 + tree_depth_db(term[1]),
                   2 + tree_depth_db(term[2]),
                   3 + tree_depth_db(term[3]),
                   3 + tree_depth_db(term[4]))
    raise ValueError("unknown de Bruijn term %r" % (term,))


def encode_db(term, tree=None, ptr=0):
    """Write a de Bruijn tuple into a slot array at ptr. Returns the tree."""
    if tree is None:
        tree = empty_tree()
    t = term[0]
    if t == "var":
        write_node(tree, ptr, "VAR", 0, term[1])
    elif t == "lam":
        write_node(tree, ptr, "LAM")
        encode_db(term[1], tree, left_ptr(ptr))
        encode_db(term[2], tree, right_ptr(ptr))
    elif t == "pi":
        write_node(tree, ptr, "PI")
        encode_db(term[1], tree, left_ptr(ptr))
        encode_db(term[2], tree, right_ptr(ptr))
    elif t == "app":
        write_node(tree, ptr, "APP")
        encode_db(term[1], tree, left_ptr(ptr))
        encode_db(term[2], tree, right_ptr(ptr))
    elif t == "type":
        write_node(tree, ptr, "TYPE", term[1], 0)
    elif t == "nat":
        write_node(tree, ptr, "NAT")
    elif t == "zero":
        write_node(tree, ptr, "ZERO")
    elif t == "succ":
        write_node(tree, ptr, "SUCC")
        encode_db(term[1], tree, left_ptr(ptr))
    elif t == "rec":
        # 3-node chain: REC(motive, REC3); REC3(base, REC2); REC2(step, target)
        write_node(tree, ptr, "REC")
        encode_db(term[1], tree, left_ptr(ptr))
        write_node(tree, right_ptr(ptr), "REC3")
        encode_db(term[2], tree, left_ptr(right_ptr(ptr)))
        write_node(tree, right_ptr(right_ptr(ptr)), "REC2")
        encode_db(term[3], tree, left_ptr(right_ptr(right_ptr(ptr))))
        encode_db(term[4], tree, right_ptr(right_ptr(right_ptr(ptr))))
    else:
        raise ValueError("unknown de Bruijn term %r" % (term,))
    return tree


def decode_tree(tree, ptr=0):
    """Read a slot array back into a de Bruijn tuple."""
    kind, level, idx = tree[ptr]
    if kind == "VAR":
        return ("var", idx)
    if kind == "LAM":
        return ("lam", decode_tree(tree, left_ptr(ptr)),
                decode_tree(tree, right_ptr(ptr)))
    if kind == "PI":
        return ("pi", decode_tree(tree, left_ptr(ptr)),
                decode_tree(tree, right_ptr(ptr)))
    if kind == "APP":
        return ("app", decode_tree(tree, left_ptr(ptr)),
                decode_tree(tree, right_ptr(ptr)))
    if kind == "TYPE":
        return ("type", level)
    if kind == "NAT":
        return ("nat",)
    if kind == "ZERO":
        return ("zero",)
    if kind == "SUCC":
        return ("succ", decode_tree(tree, left_ptr(ptr)))
    if kind == "REC":
        return ("rec", decode_tree(tree, left_ptr(ptr)),
                decode_tree(tree, left_ptr(right_ptr(ptr))),
                decode_tree(tree, left_ptr(right_ptr(right_ptr(ptr)))),
                decode_tree(tree, right_ptr(right_ptr(right_ptr(ptr)))))
    if kind == "EMPTY":
        raise ValueError("decode hit EMPTY slot at %d" % ptr)
    raise ValueError("unknown kind %r at slot %d" % (kind, ptr))


def subtree_depth(tree, ptr=0):
    """Depth of the subtree at ptr (0 for a leaf)."""
    kind, _, _ = tree[ptr]
    if kind in ("VAR", "TYPE", "NAT", "ZERO", "EMPTY"):
        return 0
    if kind in ("LAM", "PI", "APP", "SUCC"):
        d = 0
        if kind == "SUCC":
            return 1 + subtree_depth(tree, left_ptr(ptr))
        return 1 + max(subtree_depth(tree, left_ptr(ptr)),
                       subtree_depth(tree, right_ptr(ptr)))
    if kind == "REC":
        return max(1 + subtree_depth(tree, left_ptr(ptr)),
                   2 + subtree_depth(tree, left_ptr(right_ptr(ptr))),
                   3 + subtree_depth(tree, left_ptr(right_ptr(right_ptr(ptr)))),
                   3 + subtree_depth(tree, right_ptr(right_ptr(right_ptr(ptr)))))
    raise ValueError("unknown kind %r" % kind)


def count_nodes(tree, ptr=0):
    kind, _, _ = tree[ptr]
    if kind in ("VAR", "TYPE", "NAT", "ZERO", "EMPTY"):
        return 1 if kind != "EMPTY" else 0
    if kind in ("LAM", "PI", "APP"):
        return 1 + count_nodes(tree, left_ptr(ptr)) + count_nodes(tree, right_ptr(ptr))
    if kind == "SUCC":
        return 1 + count_nodes(tree, left_ptr(ptr))
    if kind == "REC":
        return (1 + count_nodes(tree, left_ptr(ptr))
                + 1 + count_nodes(tree, left_ptr(right_ptr(ptr)))
                + 1 + count_nodes(tree, left_ptr(right_ptr(right_ptr(ptr))))
                + count_nodes(tree, right_ptr(right_ptr(right_ptr(ptr)))))
    raise ValueError("unknown kind %r" % kind)


# ---------------------------------------------------------------------------
# Slot-array <-> one-hot vector (for the MatrixMachine)
# ---------------------------------------------------------------------------

def slot_onehot(node):
    """A node (kind, level, idx) -> a SLOT_DIM one-hot vector."""
    v = np.zeros(SLOT_DIM)
    kind, level, idx = node
    v[KIND_OFF + KIND[kind]] = 1.0
    v[LEVEL_OFF + level] = 1.0
    v[IDX_OFF + idx] = 1.0
    return v


def tree_to_vec(tree):
    """Slot array -> TREE_DIM one-hot vector."""
    v = np.zeros(TREE_DIM)
    for p, node in enumerate(tree):
        if node[0] == "EMPTY":
            continue
        v[p * SLOT_DIM:(p + 1) * SLOT_DIM] = slot_onehot(node)
    return v


def vec_to_tree(v):
    """TREE_DIM one-hot vector -> slot array (decoding; for lockstep)."""
    tree = []
    for p in range(N_SLOTS):
        seg = v[p * SLOT_DIM:(p + 1) * SLOT_DIM]
        if seg.sum() < 0.5:
            tree.append(EMPTY_NODE)
            continue
        k = int(np.argmax(seg[KIND_OFF:KIND_OFF + N_KINDS]))
        lv = int(np.argmax(seg[LEVEL_OFF:LEVEL_OFF + LEVELS]))
        ix = int(np.argmax(seg[IDX_OFF:IDX_OFF + VAR_IDX]))
        tree.append((KINDS[k], lv, ix))
    return tree


def ptr_onehot(p):
    v = np.zeros(N_SLOTS)
    v[p] = 1.0
    return v

# ---------------------------------------------------------------------------
# RefMachine — plain-logic state machine (slot-array lists)
#
# Design (refined from the spec; the register discipline is the part the
# spec left open, and it is the load-bearing choice):
#
#   * Trees live in a register file. Sources for a (sel, ptr) reference
#     are INPUT (the term under test, constant), AUX (the expected type
#     / second defeq term, constant), and R0..R{N_REG-1} (tree registers).
#   * Registers are a BOUNDED POOL with a free list and a
#     mark-and-sweep (2026-11-25, crux 1): _alloc pops a free
#     register; when the free list empties it sweeps, freeing every
#     register not in _live_regs() (plus the registers allocated in
#     the current step, which may not be stored in state yet). A freed
#     register's tree is reset to empty on re-allocation, so no stale
#     content survives. The pool size is POOL_SIZE (16): the complete
#     liveness measurement says at most 15 registers are live at once
#     (worst case "eta through check"), so 16 fits with one spare.
#     The register file is now 16 x 49,128 = 786,048 dims as a flat
#     state vector, down from 180 x 49,128 = 8,843,040. If the pool
#     ever exhausts, _alloc raises — a real signal that the liveness
#     function missed a source, or the pool is too small — it never
#     silently reuses a live register.
#   * The ctx is a stack of (sel, ptr) references (innermost first),
#     pushed/popped by index. Var(n) returns ctx[n] as a reference.
#   * The result of INFER/WHNF is a (sel, ptr) reference (res_sel,
#     res_ptr). COMPARE's result is res_bool. CHECK has no tree result.
#   * `dest` is the register the current INFER/WHNF writes its result
#     into. A caller allocates a fresh dest, pushes a continuation frame
#     saving the old dest, sets dest, and starts the sub-computation.
#     On RETURN the frame restores dest.
#   * The WALK engine (mode WALK) is the only thing that writes trees:
#     COPY (copy a subtree, shifting free vars) and SUBST (de Bruijn
#     substitution). Every construction (Pi, App rebuild, step type,
#     iota unfolding) is: write the fixed skeleton nodes into a fresh
#     register, push COPY/SUBST work items for the subtrees, push a
#     continuation frame, run the walk.
#   * Failure info: fail_path (tuple of segment names) and fail_kind.
#     The two motive regressions must land at ("motive",) / "not-a-type".
# ---------------------------------------------------------------------------

POOL_SIZE = 16  # bounded register file; measured max live is 15
N_REG = POOL_SIZE  # (kept name for the harness / status docs)


def rec_motive_ptr(p):
    return left_ptr(p)


def rec_base_ptr(p):
    return left_ptr(right_ptr(p))


def rec_step_ptr(p):
    return left_ptr(right_ptr(right_ptr(p)))


def rec_target_ptr(p):
    return right_ptr(right_ptr(right_ptr(p)))


class RefMachine:
    def __init__(self, term_db, ty_db=None, ctx_types=None, entry="check",
                 max_steps=200000, pool_size=POOL_SIZE):
        """term_db, ty_db: de Bruijn tuples. ctx_types: list of de Bruijn
        type tuples in insertion order (outermost first). entry: the
        public entry — "check" | "infer" | "defeq". pool_size: the
        bounded register file (default 16; measured max live is 15)."""
        self.pool_size = pool_size
        self.input_tree = encode_db(term_db)
        self.aux_tree = encode_db(ty_db) if ty_db is not None else empty_tree()
        self.regs = [empty_tree() for _ in range(pool_size)]
        self._free = list(range(pool_size - 1, -1, -1))  # LIFO free list
        self._free_set = set(self._free)
        self._step_allocs = set()  # registers allocated in the current step
        self.dest = -1            # register the current INFER/WHNF writes to
        self.ctx = []             # stack of (sel, ptr) refs, innermost first
        self.ctrl = []            # frames
        self.wstack = []          # walk work items
        # measurements (before any allocation)
        self.max_ctrl = 0
        self.max_reg = 0
        self.max_ctx = 0
        self.max_wstack = 0
        self.max_steps_used = 0
        for t in reversed(ctx_types or []):
            # context types are materialised into registers at entry;
            # the ctx stores (sel, ptr) references, not raw indices.
            r = self._alloc()
            self.regs[r] = encode_db(t)
            self.ctx.append((self._sel_of_reg(r), 0))
        self.max_ctx = len(self.ctx)
        self.mode = "INFER"
        self.fsel, self.fptr = 0, 0   # INPUT, 0
        self.asel, self.aptr = 1, 0   # AUX, 0
        self.res_sel, self.res_ptr = -1, 0
        self.res_bool = None
        self.res_level = None
        self.steps = 0
        self.max_steps = max_steps
        self.fail_path = ()
        self.fail_kind = "none"
        self._handler_seg = None  # seg of the frame whose handler runs
        self.entry = entry
        if entry == "check":
            # Public entry: validate the expected type is a type first
            # (infer it, whnf, must be a universe), then check.
            d = self._alloc()
            self._push_frame("ENTRY_VALID", ((0, 0), None, None), (0, 0))
            self.dest = d
            self.fsel, self.fptr = 1, 0   # AUX (the expected type)
            self.mode = "INFER"
        elif entry == "infer":
            d = self._alloc()
            self.dest = d
            self.fsel, self.fptr = 0, 0
            self.mode = "INFER"
        elif entry == "defeq":
            self.fsel, self.fptr = 0, 0
            self.asel, self.aptr = 1, 0
            self.mode = "COMPARE"
        else:
            raise ValueError("unknown entry %r" % entry)

    # -- state helpers ----------------------------------------------------

    def _alloc(self):
        """Allocate a tree register from the bounded pool (free list).
        When the free list empties, sweep: free every register not
        live in the machine state (plus the registers allocated in
        the current step, which may not be stored in state yet).
        The freed register's tree is reset to empty, so a reused
        register never carries stale content. Raises if the pool
        exhausts (live >= pool size): the liveness function missed
        a source, or the pool is too small. max_reg now counts the
        maximum SIMULTANEOUS allocations, not a monotonic total."""
        if not self._free:
            self._sweep()
            if not self._free:
                raise RuntimeError(
                    "register pool exhausted: live=%d pool=%d"
                    % (len(self._live_regs()), len(self.regs)))
        r = self._free.pop()
        self._free_set.discard(r)
        self.regs[r] = empty_tree()
        self._step_allocs.add(r)
        n_alloc = len(self.regs) - len(self._free)
        if n_alloc > self.max_reg:
            self.max_reg = n_alloc
        return r

    def _sweep(self):
        """Free every register not live in the machine state (and not
        allocated in the current step). Deterministic order: freed
        registers are pushed in ascending index order, so _alloc pops
        the highest free index first."""
        live = self._live_regs() | self._step_allocs
        for r in range(len(self.regs)):
            if r not in live and r not in self._free_set:
                self._free.append(r)
                self._free_set.add(r)

    def check_refs(self):
        """Debug invariant: no register referenced anywhere in the
        machine state is on the free list. A violation means the
        liveness function missed a source, or the sweep freed a live
        register. Raises AssertionError naming the register."""
        live = self._live_regs()
        bad = live & set(self._free)
        assert not bad, ("freed register(s) referenced in state: %s"
                         % sorted(bad))
        return live

    def _sel_of_reg(self, r):
        """Source index for register r (INPUT=0, AUX=1, Rr=2+r)."""
        return 2 + r

    def _reg_of_sel(self, s):
        return s - 2

    # Frame kinds whose smalls (s1, s2) carry a (sel, ptr) reference
    # rather than plain ints: the Rec frames carry the motive ref, and
    # WHNF_REC carries the target ref. Every other frame's smalls are
    # (0, 0) or a universe level (0..1), never a register sel.
    SMALLS_REF_FRAMES = frozenset([
        "REC_M", "REC_M_W", "REC_M_D", "REC_M_C", "REC_BASE_B",
        "REC_BASE", "REC_STEP_B", "REC_STEP", "REC_TGT", "REC_RES",
        "WHNF_REC",
    ])

    def _live_regs(self):
        """The set of register indices referenced anywhere in the
        machine state: frames (r1..r4, saved dest, walk dest, smalls
        refs), ctx entries, dest, res, focus (fsel/asel), and walk
        work items (source, destination, replacement). A register
        outside this set is dead: nothing in the state can read it.
        This is the liveness the bounded register file's sweep uses,
        and it is COMPLETE — the earlier measure_live_regs.py missed
        fsel/asel and the smalls refs, so its 14 was an undercount."""
        live = set()

        def add_sel(sel):
            if isinstance(sel, int) and sel >= 2:
                live.add(sel - 2)

        def add_ref(ref):
            if isinstance(ref, tuple) and len(ref) >= 1:
                add_sel(ref[0])

        for fr in self.ctrl:
            for ref in (fr[1], fr[2], fr[3], fr[4]):
                add_ref(ref)
            if isinstance(fr[7], int) and fr[7] >= 0:
                live.add(fr[7])          # saved dest
            if isinstance(fr[8], int) and fr[8] >= 0:
                live.add(fr[8])          # walk dest (a register index)
            if fr[0] in self.SMALLS_REF_FRAMES:
                add_sel(fr[5])           # smalls (sel, ptr) ref
        for ref in self.ctx:
            add_ref(ref)
        if isinstance(self.dest, int) and self.dest >= 0:
            live.add(self.dest)
        add_sel(self.res_sel)
        add_sel(self.fsel)
        add_sel(self.asel)
        for item in self.wstack:
            # (task, ssel, sptr, dsel, dptr, p1, p2, rsel, rptr)
            add_sel(item[1])
            add_sel(item[3])
            if isinstance(item[7], int) and item[7] >= 0:
                add_sel(item[7])
        return live

    def _get_tree(self, sel):
        if sel == 0:
            return self.input_tree
        if sel == 1:
            return self.aux_tree
        return self.regs[self._reg_of_sel(sel)]

    def _node(self, sel, ptr):
        return self._get_tree(sel)[ptr]

    def _kind(self, sel, ptr):
        return self._get_tree(sel)[ptr][0]

    def _push_frame(self, kind, refs, smalls, wdst=-1):
        """refs: (r1, r2, r3[, r4]) of (sel, ptr) or None. smalls: (s1, s2).
        wdst: the walk destination register for a frame that continues
        a WALK (-1 if this frame does not continue a walk). The frame
        is (kind, r1, r2, r3, r4, s1, s2, saved_dest, wdst)."""
        refs = tuple(refs) + (None,) * (4 - len(refs))
        fr = (kind, refs[0], refs[1], refs[2], refs[3], smalls[0],
              smalls[1], self.dest, wdst)
        self.ctrl.append(fr)
        if len(self.ctrl) > self.max_ctrl:
            self.max_ctrl = len(self.ctrl)
        return fr

    def _pop_frame(self):
        fr = self.ctrl.pop()
        self.dest = fr[7]   # restore the caller's dest
        return fr

    def _top_frame(self):
        return self.ctrl[-1] if self.ctrl else None

    def _error(self, kind, suffix=()):
        # Failure path = the segments of the frames on the stack
        # (outermost to innermost), plus the segment of the frame whose
        # handler is currently running (it was popped before dispatch),
        # plus any local suffix. FRAME_SEG names each frame kind's
        # descent; frames that do not descend contribute nothing.
        segs = [FRAME_SEG[fr[0]] for fr in self.ctrl
                if FRAME_SEG.get(fr[0])]
        if self._handler_seg:
            segs.append(self._handler_seg)
        self.fail_path = tuple(segs) + tuple(suffix)
        self.fail_kind = kind
        self.mode = "ERROR"

    def _push_walk(self, task, ssel, sptr, dsel, dptr, p1=0, p2=0,
                   rsel=-1, rptr=0):
        self.wstack.append((task, ssel, sptr, dsel, dptr, p1, p2,
                            rsel, rptr))
        if len(self.wstack) > self.max_wstack:
            self.max_wstack = len(self.wstack)

    def _start_walk(self, frame_kind, refs, smalls, work_items, wdst=-1):
        """Push a continuation frame, push the walk work items, enter
        WALK. wdst is the walk's destination register, recorded in the
        frame so the handler can find the built tree."""
        self._push_frame(frame_kind, refs, smalls, wdst=wdst)
        for w in work_items:
            self._push_walk(*w)
        self.mode = "WALK"

    def _return(self, res_sel=-1, res_ptr=0, res_bool=None, res_level=None):
        """A sub-computation finished: record its result, enter RETURN."""
        if res_sel >= 0:
            self.res_sel, self.res_ptr = res_sel, res_ptr
        if res_bool is not None:
            self.res_bool = res_bool
        if res_level is not None:
            self.res_level = res_level
        self.mode = "RETURN"

    # -- main loop ----------------------------------------------------------

    def step(self):
        self.steps += 1
        # Registers allocated in this step are protected from the sweep
        # until the handler stores them in state (see _alloc/_sweep).
        self._step_allocs = set()
        if self.steps > self.max_steps:
            self._error("fuel")
            return
        if self.steps > self.max_steps_used:
            self.max_steps_used = self.steps
        m = self.mode
        if m == "INFER":
            self._step_infer()
        elif m == "CHECK":
            self._step_check()
        elif m == "WHNF":
            self._step_whnf()
        elif m == "COMPARE":
            self._step_compare()
        elif m == "WALK":
            self._step_walk()
        elif m == "RETURN":
            self._step_return()
        elif m in ("HALT", "ERROR"):
            pass
        else:
            raise ValueError("unknown mode %r" % m)
        # A non-fuel error inside an eta branch is a caught CheckError:
        # the defeq that dispatched to eta is False.
        if self.mode == "ERROR":
            self._try_catch_error()

    def run(self):
        while self.mode not in ("HALT", "ERROR"):
            self.step()
        return self.mode

    # -- INFER --------------------------------------------------------------
    # Focus (fsel, fptr) is the term. Result goes into regs[dest]; on
    # completion res_sel/res_ptr = (sel_of(dest), 0) and mode = RETURN.

    def _step_infer(self):
        kind, level, idx = self._node(self.fsel, self.fptr)
        fsel, fptr = self.fsel, self.fptr
        if kind == "VAR":
            if idx < len(self.ctx):
                # The stored type's free variables are relative to the
                # context where the binder was introduced; in the
                # current context they sit (idx+1) binders further out
                # (the idx binders at 0..idx-1 plus this binder itself),
                # so the type must be shifted up by idx+1 to be read
                # here. A bare ctx[idx] would misread the binder as
                # the innermost variable (the a2 capture class, in de
                # Bruijn form). Copy it, shifted, into a fresh register.
                s, p = self.ctx[idx]
                d = self._alloc()
                self._start_walk("WALK_DONE", (None, None, None), (0, 0),
                                 [("COPY", s, p, self._sel_of_reg(d), 0,
                                   idx + 1, 0, -1, 0)], wdst=d)
            else:
                self._error("unbound")
            return
        if kind == "TYPE":
            write_node(self.regs[self.dest], 0, "TYPE", level + 1, 0)
            self._return(res_sel=self._sel_of_reg(self.dest), res_ptr=0)
            return
        if kind == "NAT":
            write_node(self.regs[self.dest], 0, "TYPE", 0, 0)
            self._return(res_sel=self._sel_of_reg(self.dest), res_ptr=0)
            return
        if kind == "ZERO":
            write_node(self.regs[self.dest], 0, "NAT")
            self._return(res_sel=self._sel_of_reg(self.dest), res_ptr=0)
            return
        if kind == "LAM":
            ty_ref = (fsel, left_ptr(fptr))
            body_ref = (fsel, right_ptr(fptr))
            d = self._alloc()
            self._push_frame("LAM_TY", (ty_ref, body_ref, None), (0, 0))
            self.dest = d
            self.fsel, self.fptr = ty_ref
            self.mode = "INFER"
            return
        if kind == "PI":
            ty_ref = (fsel, left_ptr(fptr))
            body_ref = (fsel, right_ptr(fptr))
            d = self._alloc()
            self._push_frame("PI_DOM", (ty_ref, body_ref, None), (0, 0))
            self.dest = d
            self.fsel, self.fptr = ty_ref
            self.mode = "INFER"
            return
        if kind == "APP":
            func_ref = (fsel, left_ptr(fptr))
            arg_ref = (fsel, right_ptr(fptr))
            d = self._alloc()
            self._push_frame("APP_F", (arg_ref, None, None), (0, 0))
            self.dest = d
            self.fsel, self.fptr = func_ref
            self.mode = "INFER"
            return
        if kind == "SUCC":
            pred_ref = (fsel, left_ptr(fptr))
            # check the predecessor against Nat: Nat is a one-node tree
            # in a fresh register; the check's continuation is SUCC_T.
            nr = self._alloc()
            write_node(self.regs[nr], 0, "NAT")
            self._push_frame("SUCC_T", (None, None, None), (0, 0))
            self._begin_check(pred_ref, (self._sel_of_reg(nr), 0))
            return
        if kind == "REC":
            mot_ref = (fsel, rec_motive_ptr(fptr))
            base_ref = (fsel, rec_base_ptr(fptr))
            step_ref = (fsel, rec_step_ptr(fptr))
            tgt_ref = (fsel, rec_target_ptr(fptr))
            d = self._alloc()
            # refs = (base, step, target); the motive ref rides in
            # smalls as (sel, ptr) — the frame has exactly 3 ref slots.
            self._push_frame("REC_M",
                             (base_ref, step_ref, tgt_ref),
                             (mot_ref[0], mot_ref[1]))
            self.dest = d
            self.fsel, self.fptr = mot_ref
            self.mode = "INFER"
            return
        self._error("unknown-term")

    # -- CHECK --------------------------------------------------------------
    # _begin_check(term_ref, type_ref) pushes the caller's continuation
    # (already on the stack) plus a CHK_W frame, copies the type into a
    # fresh register, and starts WHNF on it. The CHK_W handler then
    # dispatches: Lam-against-Pi, or infer + defeq.

    def _begin_check(self, term_ref, type_ref):
        # Push a CHK_W frame and start WHNF on the expected type. WHNF
        # allocates its own result register and reports it via res_*;
        # the CHK_W handler then dispatches (Lam-against-Pi, or
        # infer + defeq). No dest is allocated here.
        self._push_frame("CHK_W", (term_ref, type_ref, None), (0, 0))
        self.fsel, self.fptr = type_ref
        self.mode = "WHNF"

    def _step_check(self):
        # CHECK mode is only entered by a CHK_W handler (after the
        # type has been whnf'd); _step_check itself is not used.
        raise AssertionError("CHECK mode should not be stepped directly")

    # -- WHNF ---------------------------------------------------------------
    # Focus (fsel, fptr) is the subject. Result goes into regs[dest].
    # Head beta + head iota only; a neutral head is returned as-is
    # (rebuilt from its whnf'd head and original argument / target).

    def _step_whnf(self):
        kind, level, idx = self._node(self.fsel, self.fptr)
        fsel, fptr = self.fsel, self.fptr
        if kind == "APP":
            func_ref = (fsel, left_ptr(fptr))
            arg_ref = (fsel, right_ptr(fptr))
            self._push_frame("WHNF_APP", (arg_ref, None, None), (0, 0))
            self.fsel, self.fptr = func_ref
            self.mode = "WHNF"
            return
        if kind == "REC":
            mot_ref = (fsel, rec_motive_ptr(fptr))
            base_ref = (fsel, rec_base_ptr(fptr))
            step_ref = (fsel, rec_step_ptr(fptr))
            tgt_ref = (fsel, rec_target_ptr(fptr))
            # refs = (motive, base, step); target rides in smalls.
            self._push_frame("WHNF_REC",
                             (mot_ref, base_ref, step_ref),
                             (tgt_ref[0], tgt_ref[1]))
            self.fsel, self.fptr = tgt_ref
            self.mode = "WHNF"
            return
        # Not a redex head: the subject is already in whnf. Copy it
        # into a fresh register and return that reference.
        d = self._alloc()
        self._start_walk("WALK_DONE", (None, None, None), (0, 0),
                         [("COPY", fsel, fptr, self._sel_of_reg(d), 0,
                           0, 0, -1, 0)], wdst=d)

    # -- COMPARE (defeq) ----------------------------------------------------
    # Focus = left, aux = right. Result is res_bool. Mirrors
    # kernel_dt.defeq: whnf both sides, then structural dispatch.
    # The eta branch (exactly one side a Lam) infers the neutral's type;
    # an infer failure there is caught and yields False (the kernel
    # catches CheckError, except fuel, and returns False).

    def _step_compare(self):
        l_ref = (self.fsel, self.fptr)
        r_ref = (self.asel, self.aptr)
        self._push_frame("CMP_W_L", (l_ref, r_ref, None), (0, 0))
        self.fsel, self.fptr = l_ref
        self.mode = "WHNF"

    # -- WALK ----------------------------------------------------------------
    # One work item per step. When the stack empties, mode = RETURN.

    def _step_walk(self):
        if not self.wstack:
            self.mode = "RETURN"
            return
        (task, ssel, sptr, dsel, dptr, p1, p2, rsel, rptr) = \
            self.wstack.pop()
        kind, level, idx = self._node(ssel, sptr)
        dtree = self._get_tree(dsel)
        if task == "COPY":
            self._walk_copy(kind, level, idx, ssel, sptr, dsel, dptr,
                            p1, p2)
        elif task == "SUBST":
            self._walk_subst(kind, level, idx, ssel, sptr, dsel, dptr,
                             p1, p2, rsel, rptr)
        else:
            raise ValueError("unknown walk task %r" % task)

    def _walk_copy(self, kind, level, idx, ssel, sptr, dsel, dptr,
                   shift, cutoff):
        """COPY a subtree, shifting free vars >= cutoff by shift."""
        dtree = self._get_tree(dsel)
        if kind == "VAR":
            write_node(dtree, dptr, "VAR", 0,
                       idx + shift if idx >= cutoff else idx)
            return
        if kind in ("TYPE", "NAT", "ZERO"):
            write_node(dtree, dptr, kind, level, 0)
            return
        if kind in ("LAM", "PI"):
            write_node(dtree, dptr, kind)
            # binder type: same cutoff (not under the binder);
            # body: cutoff + 1 (under the binder).
            self._push_walk("COPY", ssel, left_ptr(sptr), dsel,
                            left_ptr(dptr), shift, cutoff, -1, 0)
            self._push_walk("COPY", ssel, right_ptr(sptr), dsel,
                            right_ptr(dptr), shift, cutoff + 1, -1, 0)
            return
        if kind == "APP":
            write_node(dtree, dptr, "APP")
            self._push_walk("COPY", ssel, left_ptr(sptr), dsel,
                            left_ptr(dptr), shift, cutoff, -1, 0)
            self._push_walk("COPY", ssel, right_ptr(sptr), dsel,
                            right_ptr(dptr), shift, cutoff, -1, 0)
            return
        if kind == "SUCC":
            write_node(dtree, dptr, "SUCC")
            self._push_walk("COPY", ssel, left_ptr(sptr), dsel,
                            left_ptr(dptr), shift, cutoff, -1, 0)
            return
        if kind == "REC":
            write_node(dtree, dptr, "REC")
            write_node(dtree, right_ptr(dptr), "REC3")
            write_node(dtree, right_ptr(right_ptr(dptr)), "REC2")
            self._push_walk("COPY", ssel, rec_motive_ptr(sptr), dsel,
                            rec_motive_ptr(dptr), shift, cutoff, -1, 0)
            self._push_walk("COPY", ssel, rec_base_ptr(sptr), dsel,
                            rec_base_ptr(dptr), shift, cutoff, -1, 0)
            self._push_walk("COPY", ssel, rec_step_ptr(sptr), dsel,
                            rec_step_ptr(dptr), shift, cutoff, -1, 0)
            self._push_walk("COPY", ssel, rec_target_ptr(sptr), dsel,
                            rec_target_ptr(dptr), shift, cutoff, -1, 0)
            return
        raise ValueError("COPY hit kind %r" % kind)

    def _walk_subst(self, kind, level, idx, ssel, sptr, dsel, dptr,
                    k, s, rsel, rptr):
        """SUBST: replace Var(k) by the replacement (rsel, rptr),
        shifted by s; free vars > k drop by 1; vars < k unchanged.
        Under a binder, k and s both increment by 1."""
        dtree = self._get_tree(dsel)
        if kind == "VAR":
            if idx == k:
                # graft the replacement, shifted by s
                self._push_walk("COPY", rsel, rptr, dsel, dptr, s, 0,
                                -1, 0)
            elif idx > k:
                write_node(dtree, dptr, "VAR", 0, idx - 1)
            else:
                write_node(dtree, dptr, "VAR", 0, idx)
            return
        if kind in ("TYPE", "NAT", "ZERO"):
            write_node(dtree, dptr, kind, level, 0)
            return
        if kind in ("LAM", "PI"):
            write_node(dtree, dptr, kind)
            # binder type: same (k, s); body: (k+1, s+1).
            self._push_walk("SUBST", ssel, left_ptr(sptr), dsel,
                            left_ptr(dptr), k, s, rsel, rptr)
            self._push_walk("SUBST", ssel, right_ptr(sptr), dsel,
                            right_ptr(dptr), k + 1, s + 1, rsel, rptr)
            return
        if kind == "APP":
            write_node(dtree, dptr, "APP")
            self._push_walk("SUBST", ssel, left_ptr(sptr), dsel,
                            left_ptr(dptr), k, s, rsel, rptr)
            self._push_walk("SUBST", ssel, right_ptr(sptr), dsel,
                            right_ptr(dptr), k, s, rsel, rptr)
            return
        if kind == "SUCC":
            write_node(dtree, dptr, "SUCC")
            self._push_walk("SUBST", ssel, left_ptr(sptr), dsel,
                            left_ptr(dptr), k, s, rsel, rptr)
            return
        if kind == "REC":
            write_node(dtree, dptr, "REC")
            write_node(dtree, right_ptr(dptr), "REC3")
            write_node(dtree, right_ptr(right_ptr(dptr)), "REC2")
            for src_p, dst_p in (
                    (rec_motive_ptr(sptr), rec_motive_ptr(dptr)),
                    (rec_base_ptr(sptr), rec_base_ptr(dptr)),
                    (rec_step_ptr(sptr), rec_step_ptr(dptr)),
                    (rec_target_ptr(sptr), rec_target_ptr(dptr))):
                self._push_walk("SUBST", ssel, src_p, dsel, dst_p,
                                k, s, rsel, rptr)
            return
        raise ValueError("SUBST hit kind %r" % kind)

    # -- RETURN --------------------------------------------------------------
    # Pop the top frame (restoring dest), dispatch on its kind. The
    # sub-computation's result is in res_sel/res_ptr (tree),
    # res_bool (defeq), or res_level. If the control stack is empty
    # after the pop, the top-level computation finished: HALT.

    def _step_return(self):
        if not self.ctrl:
            self.mode = "HALT"
            return
        fr = self._pop_frame()
        kind = fr[0]
        r1, r2, r3 = fr[1], fr[2], fr[3]
        s1, s2 = fr[5], fr[6]
        handler = getattr(self, "_ret_" + kind.lower(), None)
        if handler is None:
            raise ValueError("no RETURN handler for frame %r" % kind)
        # The handler runs after its frame was popped; an error it
        # raises is at that frame's segment, so remember it for _error.
        self._handler_seg = FRAME_SEG.get(kind)
        try:
            handler(fr, r1, r2, r3, s1, s2)
        finally:
            self._handler_seg = None

    # -- RETURN handlers -----------------------------------------------------

    def _ret_entry_valid(self, fr, r1, r2, r3, s1, s2):
        # res = the expected type's type. whnf it.
        self._push_frame("ENTRY_VALID_W", (r1, None, None), (0, 0))
        self.fsel, self.fptr = self.res_sel, self.res_ptr
        self.mode = "WHNF"

    def _ret_entry_valid_w(self, fr, r1, r2, r3, s1, s2):
        # res = whnf(expected type's type). Must be a universe.
        if self._kind(self.res_sel, self.res_ptr) != "TYPE":
            self._error("not-a-type")
            return
        term_ref = r1  # (INPUT, 0)
        self._begin_check(term_ref, (1, 0))   # type is AUX

    def _ret_lam_ty(self, fr, r1, r2, r3, s1, s2):
        # res = the binder type's type. whnf it.
        self._push_frame("LAM_TY_W", (r1, r2, None), (0, 0))
        self.fsel, self.fptr = self.res_sel, self.res_ptr
        self.mode = "WHNF"

    def _ret_lam_ty_w(self, fr, r1, r2, r3, s1, s2):
        # res = whnf(binder type's type). Must be a universe.
        if self._kind(self.res_sel, self.res_ptr) != "TYPE":
            self._error("not-a-type")
            return
        ty_ref, body_ref = r1, r2
        self.ctx.insert(0, ty_ref)
        if len(self.ctx) > self.max_ctx:
            self.max_ctx = len(self.ctx)
        d = self._alloc()
        self._push_frame("LAM_BODY", (body_ref, None, None), (0, 0))
        self.dest = d
        self.fsel, self.fptr = body_ref
        self.mode = "INFER"

    def _ret_lam_body(self, fr, r1, r2, r3, s1, s2):
        # res = the body's type. Build Pi(binder ty, body ty) into
        # regs[dest] (the caller's dest, restored on the pop).
        body_ty_ref = (self.res_sel, self.res_ptr)
        self.ctx.pop(0)
        # the binder type is the ctx entry we just popped — recover it:
        # it is the Lam's left child. The Lam node is the focus's
        # parent; r1 is the body ref, whose parent is the Lam node.
        # body_ref = (sel, right(lam_ptr)) so lam_ptr = parent(body).
        lam_ptr = parent_ptr(r1[1])
        ty_ref = (r1[0], left_ptr(lam_ptr))
        d = self.dest
        write_node(self.regs[d], 0, "PI")
        self._start_walk("WALK_DONE", (None, None, None), (0, 0),
                         [("COPY", ty_ref[0], ty_ref[1],
                           self._sel_of_reg(d), left_ptr(0), 0, 0, -1, 0),
                          ("COPY", body_ty_ref[0], body_ty_ref[1],
                           self._sel_of_reg(d), right_ptr(0), 0, 0, -1, 0)],
                         wdst=d)

    def _ret_pi_dom(self, fr, r1, r2, r3, s1, s2):
        # res = the domain's type. whnf it.
        self._push_frame("PI_DOM_W", (r1, r2, None), (0, 0))
        self.fsel, self.fptr = self.res_sel, self.res_ptr
        self.mode = "WHNF"

    def _ret_pi_dom_w(self, fr, r1, r2, r3, s1, s2):
        # res = whnf(domain type). Must be a universe at level i.
        if self._kind(self.res_sel, self.res_ptr) != "TYPE":
            self._error("not-a-type")
            return
        i = self._node(self.res_sel, self.res_ptr)[1]
        ty_ref, body_ref = r1, r2
        self.ctx.insert(0, ty_ref)
        if len(self.ctx) > self.max_ctx:
            self.max_ctx = len(self.ctx)
        d = self._alloc()
        self._push_frame("PI_BODY", (body_ref, None, None), (i, 0))
        self.dest = d
        self.fsel, self.fptr = body_ref
        self.mode = "INFER"

    def _ret_pi_body(self, fr, r1, r2, r3, s1, s2):
        # res = the codomain's type. whnf it.
        i = s1
        self._push_frame("PI_BODY_W", (None, None, None), (i, 0))
        self.fsel, self.fptr = self.res_sel, self.res_ptr
        self.mode = "WHNF"

    def _ret_pi_body_w(self, fr, r1, r2, r3, s1, s2):
        # res = whnf(codomain type). Must be a universe at level j.
        if self._kind(self.res_sel, self.res_ptr) != "TYPE":
            self._error("not-a-type")
            return
        j = self._node(self.res_sel, self.res_ptr)[1]
        self.ctx.pop(0)
        d = self.dest
        write_node(self.regs[d], 0, "TYPE", max(s1, j), 0)
        self._return(res_sel=self._sel_of_reg(d), res_ptr=0)

    def _ret_app_f(self, fr, r1, r2, r3, s1, s2):
        # res = the func's type. whnf it.
        self._push_frame("APP_F_W", (r1, None, None), (0, 0))
        self.fsel, self.fptr = self.res_sel, self.res_ptr
        self.mode = "WHNF"

    def _ret_app_f_w(self, fr, r1, r2, r3, s1, s2):
        # res = whnf(func type). Must be a Pi.
        if self._kind(self.res_sel, self.res_ptr) != "PI":
            self._error("not-pi")
            return
        arg_ref = r1
        dom_ref = (self.res_sel, left_ptr(self.res_ptr))
        cod_ref = (self.res_sel, right_ptr(self.res_ptr))
        self._push_frame("APP_A", (cod_ref, arg_ref, None), (0, 0))
        self._begin_check(arg_ref, dom_ref)

    def _ret_app_a(self, fr, r1, r2, r3, s1, s2):
        # The arg check succeeded. Result = subst(cod, 0, arg).
        cod_ref, arg_ref = r1, r2
        d = self._alloc()
        self._start_walk("APP_SUBST", (None, None, None), (0, 0),
                         [("SUBST", cod_ref[0], cod_ref[1],
                           self._sel_of_reg(d), 0, 0, 0,
                           arg_ref[0], arg_ref[1])], wdst=d)

    def _ret_app_subst(self, fr, r1, r2, r3, s1, s2):
        self._return(res_sel=self._sel_of_reg(fr[8]), res_ptr=0)

    def _ret_succ_t(self, fr, r1, r2, r3, s1, s2):
        # The predecessor check succeeded. Result is Nat.
        d = self.dest
        write_node(self.regs[d], 0, "NAT")
        self._return(res_sel=self._sel_of_reg(d), res_ptr=0)

    # -- Rec (the eliminator) ----------------------------------------------

    def _ret_rec_m(self, fr, r1, r2, r3, s1, s2):
        # res = the motive's type. whnf it.
        mot_ref = (s1, s2)
        self._push_frame("REC_M_W", (r1, r2, r3), (s1, s2))
        self.fsel, self.fptr = self.res_sel, self.res_ptr
        self.mode = "WHNF"

    def _ret_rec_m_w(self, fr, r1, r2, r3, s1, s2):
        # res = whnf(motive type). Must be a Pi.
        if self._kind(self.res_sel, self.res_ptr) != "PI":
            self._error("not-pi")
            return
        mot_ref = (s1, s2)
        dom_ref = (self.res_sel, left_ptr(self.res_ptr))
        # defeq(dom, Nat): Nat is a one-node tree in a fresh register.
        # The whnf'd motive type (a Pi) is carried in the frame's 4th
        # ref: the COMPARE below clobbers res_*, so _ret_rec_m_d cannot
        # recover the codomain from res.
        nr = self._alloc()
        write_node(self.regs[nr], 0, "NAT")
        self._push_frame("REC_M_D", (r1, r2, r3,
                                     (self.res_sel, self.res_ptr)),
                         (s1, s2))
        self.fsel, self.fptr = dom_ref
        self.asel, self.aptr = self._sel_of_reg(nr), 0
        self.mode = "COMPARE"

    def _ret_rec_m_d(self, fr, r1, r2, r3, s1, s2):
        # res_bool = defeq(motive domain, Nat).
        if not self.res_bool:
            self._error("mismatch")
            return
        mot_ref = (s1, s2)
        # The whnf'd motive type is in the frame's 4th ref (fr[4]);
        # its codomain is what must be a universe.
        mot_ty_ref = fr[4]
        cod_ref = (mot_ty_ref[0], right_ptr(mot_ty_ref[1]))
        self._push_frame("REC_M_C", (r1, r2, r3), (s1, s2))
        self.fsel, self.fptr = cod_ref
        self.mode = "WHNF"

    def _ret_rec_m_c(self, fr, r1, r2, r3, s1, s2):
        # res = whnf(motive codomain). Must BE a universe tag.
        if self._kind(self.res_sel, self.res_ptr) != "TYPE":
            self._error("not-a-type")
            return
        # Build App(motive, Zero) — the base's expected type.
        mot_ref = (s1, s2)
        d = self._alloc()
        write_node(self.regs[d], 0, "APP")
        write_node(self.regs[d], right_ptr(0), "ZERO")
        self._start_walk("REC_BASE_B", (r1, r2, r3), (s1, s2),
                         [("COPY", mot_ref[0], mot_ref[1],
                           self._sel_of_reg(d), left_ptr(0), 0, 0, -1, 0)],
                         wdst=d)

    def _ret_rec_base_b(self, fr, r1, r2, r3, s1, s2):
        # The App(motive, Zero) build finished; it is in regs[fr[8]].
        # Check the base against it.
        base_ref = r1
        app_ref = (self._sel_of_reg(fr[8]), 0)
        self._push_frame("REC_BASE", (r2, r3, None), (s1, s2))
        self._begin_check(base_ref, app_ref)

    def _ret_rec_base(self, fr, r1, r2, r3, s1, s2):
        # The base check succeeded. Build the step type:
        #   Pi n:Nat. Pi r:(App mot Var n). App mot (Succ (Var n))
        # under two binders, so the motive is copied with shift 2 and
        # Var n is index 1 inside the inner Pi.
        mot_ref = (s1, s2)
        step_ref, tgt_ref = r1, r2
        d = self._alloc()
        ds = self._sel_of_reg(d)
        # slots: 0=PI(outer), 1=NAT, 2=PI(inner), 5=APP(dom),
        # 11=motive, 12=VAR(1), 6=APP(cod), 13=motive, 14=SUCC, 29=VAR(1)
        write_node(self.regs[d], 0, "PI")
        write_node(self.regs[d], left_ptr(0), "NAT")
        write_node(self.regs[d], right_ptr(0), "PI")
        write_node(self.regs[d], left_ptr(right_ptr(0)), "APP")
        # In the inner Pi's DOMAIN, Var n is index 0 (the domain is
        # under the outer Pi's binder n only, not under r).
        write_node(self.regs[d], right_ptr(left_ptr(right_ptr(0))),
                   "VAR", 0, 0)
        write_node(self.regs[d], right_ptr(right_ptr(0)), "APP")
        write_node(self.regs[d], right_ptr(right_ptr(right_ptr(0))),
                   "SUCC")
        # In the inner Pi's CODOMAIN, Var n is index 1 (under n and r).
        write_node(self.regs[d],
                   left_ptr(right_ptr(right_ptr(right_ptr(0)))),
                   "VAR", 0, 1)
        self._start_walk(
            "REC_STEP_B", (step_ref, tgt_ref, None), (s1, s2),
            [("COPY", mot_ref[0], mot_ref[1], ds,
              left_ptr(left_ptr(right_ptr(0))), 1, 0, -1, 0),
             ("COPY", mot_ref[0], mot_ref[1], ds,
              left_ptr(right_ptr(right_ptr(0))), 2, 0, -1, 0)],
            wdst=d)

    def _ret_rec_step_b(self, fr, r1, r2, r3, s1, s2):
        # The step-type build finished; it is in regs[fr[8]].
        # Check the step against it.
        step_ref = r1
        ty_ref = (self._sel_of_reg(fr[8]), 0)
        self._push_frame("REC_STEP", (r2, None, None), (s1, s2))
        self._begin_check(step_ref, ty_ref)

    def _ret_rec_step(self, fr, r1, r2, r3, s1, s2):
        # The step check succeeded. Check the target against Nat.
        tgt_ref = r1
        nr = self._alloc()
        write_node(self.regs[nr], 0, "NAT")
        self._push_frame("REC_TGT", (tgt_ref, None, None), (s1, s2))
        self._begin_check(tgt_ref, (self._sel_of_reg(nr), 0))

    def _ret_rec_tgt(self, fr, r1, r2, r3, s1, s2):
        # The target check succeeded. Result = App(motive, target).
        mot_ref = (s1, s2)
        tgt_ref = r1
        d = self._alloc()
        write_node(self.regs[d], 0, "APP")
        self._start_walk("REC_RES", (None, None, None), (s1, s2),
                         [("COPY", mot_ref[0], mot_ref[1],
                           self._sel_of_reg(d), left_ptr(0), 0, 0, -1, 0),
                          ("COPY", tgt_ref[0], tgt_ref[1],
                           self._sel_of_reg(d), right_ptr(0), 0, 0, -1, 0)],
                         wdst=d)

    def _ret_rec_res(self, fr, r1, r2, r3, s1, s2):
        self._return(res_sel=self._sel_of_reg(fr[8]), res_ptr=0)

    # -- CHECK handlers --------------------------------------------------------

    def _ret_chk_w(self, fr, r1, r2, r3, s1, s2):
        # res = whnf(expected type). term = r1, expected type = r2.
        term_ref, type_ref = r1, r2
        term_kind = self._kind(term_ref[0], term_ref[1])
        type_kind = self._kind(self.res_sel, self.res_ptr)
        if term_kind == "LAM" and type_kind == "PI":
            binder_ty_ref = (term_ref[0], left_ptr(term_ref[1]))
            body_ref = (term_ref[0], right_ptr(term_ref[1]))
            dom_ref = (self.res_sel, left_ptr(self.res_ptr))
            cod_ref = (self.res_sel, right_ptr(self.res_ptr))
            self._push_frame("CHK_LAM_D", (body_ref, cod_ref, dom_ref),
                             (0, 0))
            self.fsel, self.fptr = binder_ty_ref
            self.asel, self.aptr = dom_ref
            self.mode = "COMPARE"
            return
        self._push_frame("CHK_INF", (type_ref, None, None), (0, 0))
        d = self._alloc()
        self.dest = d
        self.fsel, self.fptr = term_ref
        self.mode = "INFER"

    def _ret_chk_lam_d(self, fr, r1, r2, r3, s1, s2):
        # res_bool = defeq(binder type, domain).
        if not self.res_bool:
            self._error("mismatch")
            return
        body_ref, cod_ref, dom_ref = r1, r2, r3
        # The body is checked in the context extended with the domain.
        self.ctx.insert(0, dom_ref)
        if len(self.ctx) > self.max_ctx:
            self.max_ctx = len(self.ctx)
        self._push_frame("CHK_LAM_B", (None, None, None), (0, 0))
        self._begin_check(body_ref, cod_ref)

    def _ret_chk_lam_b(self, fr, r1, r2, r3, s1, s2):
        # The body check succeeded; the whole check succeeds.
        self.ctx.pop(0)
        self.mode = "RETURN"

    def _ret_chk_inf(self, fr, r1, r2, r3, s1, s2):
        # res = the inferred type. defeq it against the expected type.
        type_ref = r1
        self._push_frame("CHK_DEFEQ", (None, None, None), (0, 0))
        self.fsel, self.fptr = self.res_sel, self.res_ptr
        self.asel, self.aptr = type_ref
        self.mode = "COMPARE"

    def _ret_chk_defeq(self, fr, r1, r2, r3, s1, s2):
        if not self.res_bool:
            self._error("mismatch")
            return
        self.mode = "RETURN"

    # -- WHNF handlers ---------------------------------------------------------

    def _ret_whnf_app(self, fr, r1, r2, r3, s1, s2):
        # res = whnf(func). arg = r1.
        arg_ref = r1
        if self._kind(self.res_sel, self.res_ptr) == "LAM":
            body_ref = (self.res_sel, right_ptr(self.res_ptr))
            d = self._alloc()
            self._start_walk("WHNF_APP_S", (None, None, None), (0, 0),
                             [("SUBST", body_ref[0], body_ref[1],
                               self._sel_of_reg(d), 0, 0, 0,
                               arg_ref[0], arg_ref[1])], wdst=d)
            return
        # Neutral head: rebuild App(whnf func, arg) and return it.
        f_w = (self.res_sel, self.res_ptr)
        d = self._alloc()
        write_node(self.regs[d], 0, "APP")
        self._start_walk("WALK_DONE", (None, None, None), (0, 0),
                         [("COPY", f_w[0], f_w[1],
                           self._sel_of_reg(d), left_ptr(0), 0, 0, -1, 0),
                          ("COPY", arg_ref[0], arg_ref[1],
                           self._sel_of_reg(d), right_ptr(0), 0, 0, -1, 0)],
                         wdst=d)

    def _ret_whnf_app_s(self, fr, r1, r2, r3, s1, s2):
        # The subst finished; loop whnf on the result.
        self.fsel, self.fptr = self._sel_of_reg(fr[8]), 0
        self.mode = "WHNF"

    def _ret_whnf_rec(self, fr, r1, r2, r3, s1, s2):
        # res = whnf(target). motive/base/step in r1/r2/r3, target in s1/s2.
        mot_ref, base_ref, step_ref = r1, r2, r3
        tgt_kind = self._kind(self.res_sel, self.res_ptr)
        if tgt_kind == "ZERO":
            self._return(res_sel=base_ref[0], res_ptr=base_ref[1])
            return
        if tgt_kind == "SUCC":
            m_ref = (self.res_sel, left_ptr(self.res_ptr))
            d = self._alloc()
            ds = self._sel_of_reg(d)
            # App(App(step, m), Rec(mot, base, step, m))
            # 0=APP, 1=APP, 2=REC, 3=step, 4=m, 5=mot, 6=REC3,
            # 13=base, 14=REC2, 29=step, 30=m
            write_node(self.regs[d], 0, "APP")
            write_node(self.regs[d], left_ptr(0), "APP")
            write_node(self.regs[d], right_ptr(0), "REC")
            write_node(self.regs[d], right_ptr(right_ptr(0)), "REC3")
            write_node(self.regs[d],
                       right_ptr(right_ptr(right_ptr(0))), "REC2")
            self._start_walk(
                "WHNF_REC_S", (None, None, None), (0, 0),
                [("COPY", step_ref[0], step_ref[1], ds,
                  left_ptr(left_ptr(0)), 0, 0, -1, 0),
                 ("COPY", m_ref[0], m_ref[1], ds,
                  right_ptr(left_ptr(0)), 0, 0, -1, 0),
                 ("COPY", mot_ref[0], mot_ref[1], ds,
                  left_ptr(right_ptr(0)), 0, 0, -1, 0),
                 ("COPY", base_ref[0], base_ref[1], ds,
                  left_ptr(right_ptr(right_ptr(0))), 0, 0, -1, 0),
                 ("COPY", step_ref[0], step_ref[1], ds,
                  left_ptr(right_ptr(right_ptr(right_ptr(0)))),
                  0, 0, -1, 0),
                 ("COPY", m_ref[0], m_ref[1], ds,
                  right_ptr(right_ptr(right_ptr(right_ptr(0)))),
                  0, 0, -1, 0)],
                wdst=d)
            return
        # Neutral target: rebuild Rec(mot, base, step, whnf target).
        tgt_w = (self.res_sel, self.res_ptr)
        d = self._alloc()
        ds = self._sel_of_reg(d)
        write_node(self.regs[d], 0, "REC")
        write_node(self.regs[d], right_ptr(0), "REC3")
        write_node(self.regs[d], right_ptr(right_ptr(0)), "REC2")
        self._start_walk(
            "WALK_DONE", (None, None, None), (0, 0),
            [("COPY", mot_ref[0], mot_ref[1], ds,
              rec_motive_ptr(0), 0, 0, -1, 0),
             ("COPY", base_ref[0], base_ref[1], ds,
              rec_base_ptr(0), 0, 0, -1, 0),
             ("COPY", step_ref[0], step_ref[1], ds,
              rec_step_ptr(0), 0, 0, -1, 0),
             ("COPY", tgt_w[0], tgt_w[1], ds,
              rec_target_ptr(0), 0, 0, -1, 0)],
            wdst=d)

    def _ret_whnf_rec_s(self, fr, r1, r2, r3, s1, s2):
        # The iota unfolding finished; loop whnf on the result.
        self.fsel, self.fptr = self._sel_of_reg(fr[8]), 0
        self.mode = "WHNF"

    # -- COMPARE handlers ------------------------------------------------------

    def _ret_cmp_w_l(self, fr, r1, r2, r3, s1, s2):
        # res = whnf(left). Now whnf the right.
        left_w = (self.res_sel, self.res_ptr)
        self._push_frame("CMP_W_R", (left_w, r2, None), (0, 0))
        self.fsel, self.fptr = r2
        self.mode = "WHNF"

    def _ret_cmp_w_r(self, fr, r1, r2, r3, s1, s2):
        # res = whnf(right); left whnf is r1. Structural dispatch.
        lw = r1
        rw = (self.res_sel, self.res_ptr)
        lk = self._kind(lw[0], lw[1])
        rk = self._kind(rw[0], rw[1])
        if lk == "LAM" and rk == "LAM":
            self._push_frame("CMP_LAM_T", (lw, rw, None), (0, 0))
            self.fsel, self.fptr = (lw[0], left_ptr(lw[1]))
            self.asel, self.aptr = (rw[0], left_ptr(rw[1]))
            self.mode = "COMPARE"
            return
        if lk == "LAM" or rk == "LAM":
            # Eta: exactly one side is a Lam.
            if lk == "LAM":
                lam_ref, neutral_ref = lw, rw
            else:
                lam_ref, neutral_ref = rw, lw
            self._push_frame("CMP_ETA", (lam_ref, neutral_ref, None),
                             (0, 0))
            d = self._alloc()
            self.dest = d
            self.fsel, self.fptr = neutral_ref
            self.mode = "INFER"
            return
        if lk == "VAR" and rk == "VAR":
            li = self._node(lw[0], lw[1])[2]
            ri = self._node(rw[0], rw[1])[2]
            self._return(res_bool=(li == ri))
            return
        if lk == "TYPE" and rk == "TYPE":
            ll = self._node(lw[0], lw[1])[1]
            rl = self._node(rw[0], rw[1])[1]
            self._return(res_bool=(ll == rl))
            return
        if lk == "PI" and rk == "PI":
            self._push_frame("CMP_PI_D", (lw, rw, None), (0, 0))
            self.fsel, self.fptr = (lw[0], left_ptr(lw[1]))
            self.asel, self.aptr = (rw[0], left_ptr(rw[1]))
            self.mode = "COMPARE"
            return
        if lk == "APP" and rk == "APP":
            self._push_frame("CMP_APP_L", (lw, rw, None), (0, 0))
            self.fsel, self.fptr = (lw[0], left_ptr(lw[1]))
            self.asel, self.aptr = (rw[0], left_ptr(rw[1]))
            self.mode = "COMPARE"
            return
        if lk == "NAT" and rk == "NAT":
            self._return(res_bool=True)
            return
        if lk == "ZERO" and rk == "ZERO":
            self._return(res_bool=True)
            return
        if lk == "SUCC" and rk == "SUCC":
            self._push_frame("CMP_SUCC", (None, None, None), (0, 0))
            self.fsel, self.fptr = (lw[0], left_ptr(lw[1]))
            self.asel, self.aptr = (rw[0], left_ptr(rw[1]))
            self.mode = "COMPARE"
            return
        if lk == "REC" and rk == "REC":
            self._push_frame("CMP_REC_M", (lw, rw, None), (0, 0))
            self.fsel, self.fptr = (lw[0], rec_motive_ptr(lw[1]))
            self.asel, self.aptr = (rw[0], rec_motive_ptr(rw[1]))
            self.mode = "COMPARE"
            return
        self._return(res_bool=False)

    def _ret_cmp_lam_t(self, fr, r1, r2, r3, s1, s2):
        # res_bool = defeq(binder types).
        if not self.res_bool:
            self._return(res_bool=False)
            return
        lw, rw = r1, r2
        self.ctx.insert(0, (lw[0], left_ptr(lw[1])))
        if len(self.ctx) > self.max_ctx:
            self.max_ctx = len(self.ctx)
        self._push_frame("CMP_LAM_B", (lw, rw, None), (0, 0))
        self.fsel, self.fptr = (lw[0], right_ptr(lw[1]))
        self.asel, self.aptr = (rw[0], right_ptr(rw[1]))
        self.mode = "COMPARE"

    def _ret_cmp_lam_b(self, fr, r1, r2, r3, s1, s2):
        self.ctx.pop(0)
        self._return(res_bool=self.res_bool)

    def _ret_cmp_pi_d(self, fr, r1, r2, r3, s1, s2):
        if not self.res_bool:
            self._return(res_bool=False)
            return
        lw, rw = r1, r2
        self.ctx.insert(0, (lw[0], left_ptr(lw[1])))
        if len(self.ctx) > self.max_ctx:
            self.max_ctx = len(self.ctx)
        self._push_frame("CMP_PI_C", (lw, rw, None), (0, 0))
        self.fsel, self.fptr = (lw[0], right_ptr(lw[1]))
        self.asel, self.aptr = (rw[0], right_ptr(rw[1]))
        self.mode = "COMPARE"

    def _ret_cmp_pi_c(self, fr, r1, r2, r3, s1, s2):
        self.ctx.pop(0)
        self._return(res_bool=self.res_bool)

    def _ret_cmp_app_l(self, fr, r1, r2, r3, s1, s2):
        if not self.res_bool:
            self._return(res_bool=False)
            return
        lw, rw = r1, r2
        self._push_frame("CMP_APP_R", (lw, rw, None), (0, 0))
        self.fsel, self.fptr = (lw[0], right_ptr(lw[1]))
        self.asel, self.aptr = (rw[0], right_ptr(rw[1]))
        self.mode = "COMPARE"

    def _ret_cmp_app_r(self, fr, r1, r2, r3, s1, s2):
        self._return(res_bool=self.res_bool)

    def _ret_cmp_succ(self, fr, r1, r2, r3, s1, s2):
        self._return(res_bool=self.res_bool)

    def _ret_cmp_rec_m(self, fr, r1, r2, r3, s1, s2):
        if not self.res_bool:
            self._return(res_bool=False)
            return
        lw, rw = r1, r2
        self._push_frame("CMP_REC_B", (lw, rw, None), (0, 0))
        self.fsel, self.fptr = (lw[0], rec_base_ptr(lw[1]))
        self.asel, self.aptr = (rw[0], rec_base_ptr(rw[1]))
        self.mode = "COMPARE"

    def _ret_cmp_rec_b(self, fr, r1, r2, r3, s1, s2):
        if not self.res_bool:
            self._return(res_bool=False)
            return
        lw, rw = r1, r2
        self._push_frame("CMP_REC_S", (lw, rw, None), (0, 0))
        self.fsel, self.fptr = (lw[0], rec_step_ptr(lw[1]))
        self.asel, self.aptr = (rw[0], rec_step_ptr(rw[1]))
        self.mode = "COMPARE"

    def _ret_cmp_rec_s(self, fr, r1, r2, r3, s1, s2):
        if not self.res_bool:
            self._return(res_bool=False)
            return
        lw, rw = r1, r2
        self._push_frame("CMP_REC_T", (lw, rw, None), (0, 0))
        self.fsel, self.fptr = (lw[0], rec_target_ptr(lw[1]))
        self.asel, self.aptr = (rw[0], rec_target_ptr(rw[1]))
        self.mode = "COMPARE"

    def _ret_cmp_rec_t(self, fr, r1, r2, r3, s1, s2):
        self._return(res_bool=self.res_bool)

    def _ret_cmp_eta(self, fr, r1, r2, r3, s1, s2):
        # res = the neutral's type (inferred). whnf it.
        self._push_frame("CMP_ETA_W", (r1, r2, None), (0, 0))
        self.fsel, self.fptr = self.res_sel, self.res_ptr
        self.mode = "WHNF"

    def _ret_cmp_eta_w(self, fr, r1, r2, r3, s1, s2):
        # res = whnf(neutral type). Must be a Pi.
        if self._kind(self.res_sel, self.res_ptr) != "PI":
            self._return(res_bool=False)
            return
        lam_ref, neutral_ref = r1, r2
        dom_ref = (self.res_sel, left_ptr(self.res_ptr))
        lam_ty_ref = (lam_ref[0], left_ptr(lam_ref[1]))
        # The domain ref rides in the frame's 3rd ref: the COMPARE
        # below clobbers res_*, so _ret_cmp_eta_d cannot recover it
        # from res.
        self._push_frame("CMP_ETA_D", (lam_ref, neutral_ref, dom_ref),
                         (0, 0))
        self.fsel, self.fptr = lam_ty_ref
        self.asel, self.aptr = dom_ref
        self.mode = "COMPARE"

    def _ret_cmp_eta_d(self, fr, r1, r2, r3, s1, s2):
        # res_bool = defeq(lam binder type, domain).
        if not self.res_bool:
            self._return(res_bool=False)
            return
        lam_ref, neutral_ref, dom_ref = r1, r2, r3
        # The domain (r3) is the ctx entry for the fresh variable.
        self.ctx.insert(0, dom_ref)
        if len(self.ctx) > self.max_ctx:
            self.max_ctx = len(self.ctx)
        lam_body_ref = (lam_ref[0], right_ptr(lam_ref[1]))
        # Build App(shift(neutral, 1), Var 0) into a fresh register.
        d = self._alloc()
        write_node(self.regs[d], 0, "APP")
        write_node(self.regs[d], right_ptr(0), "VAR", 0, 0)
        self._start_walk("CMP_ETA_B", (lam_body_ref, None, None),
                         (0, 0),
                         [("COPY", neutral_ref[0], neutral_ref[1],
                           self._sel_of_reg(d), left_ptr(0), 1, 0, -1, 0)],
                         wdst=d)

    def _ret_cmp_eta_b(self, fr, r1, r2, r3, s1, s2):
        # The App build finished; compare the bodies.
        lam_body_ref = r1
        app_ref = (self._sel_of_reg(fr[8]), 0)
        self._push_frame("CMP_ETA_R", (None, None, None), (0, 0))
        self.fsel, self.fptr = lam_body_ref
        self.asel, self.aptr = app_ref
        self.mode = "COMPARE"

    def _ret_cmp_eta_r(self, fr, r1, r2, r3, s1, s2):
        self.ctx.pop(0)
        self._return(res_bool=self.res_bool)

    def _ret_walk_done(self, fr, r1, r2, r3, s1, s2):
        self._return(res_sel=self._sel_of_reg(fr[8]), res_ptr=0)

    # -- error catching (eta) -------------------------------------------------
    # A non-fuel error while a CMP_ETA frame is on the stack is the
    # kernel's caught CheckError: the defeq that dispatched to eta is
    # False. Unwind to that defeq's CMP_W_R frame and return False.

    def _try_catch_error(self):
        if self.mode != "ERROR" or self.fail_kind == "fuel":
            return False
        for i in range(len(self.ctrl) - 1, -1, -1):
            if self.ctrl[i][0] == "CMP_ETA":
                # The CMP_W_R that dispatched this eta is directly
                # below it; unwind to just below that CMP_W_R.
                j = i - 1
                while j >= 0 and self.ctrl[j][0] != "CMP_W_R":
                    j -= 1
                if j < 0:
                    return False
                del self.ctrl[j:]
                self.fail_path = ()
                self.fail_kind = "none"
                self.res_bool = False
                self.mode = "RETURN"
                return True
        return False

# ---------------------------------------------------------------------------
# Harness — kernel_dt class terms -> de Bruijn -> RefMachine, verdicts
# compared against kernel_dt.py (the sealed plain kernel).
# ---------------------------------------------------------------------------

def kdt_to_db(term, env=()):
    """kernel_dt class term -> de Bruijn tuple. env: binder names,
    innermost last. Free variables must be in env (the context names
    are passed as the initial env)."""
    if isinstance(term, kdt.Var):
        return ("var", list(reversed(env)).index(term.name))
    if isinstance(term, kdt.Lam):
        return ("lam", kdt_to_db(term.var_type, env),
                kdt_to_db(term.body, env + (term.var,)))
    if isinstance(term, kdt.Pi):
        return ("pi", kdt_to_db(term.var_type, env),
                kdt_to_db(term.body, env + (term.var,)))
    if isinstance(term, kdt.App):
        return ("app", kdt_to_db(term.func, env),
                kdt_to_db(term.arg, env))
    if isinstance(term, kdt.Type):
        return ("type", term.level)
    if isinstance(term, kdt.Nat):
        return ("nat",)
    if isinstance(term, kdt.Zero):
        return ("zero",)
    if isinstance(term, kdt.Succ):
        return ("succ", kdt_to_db(term.pred, env))
    if isinstance(term, kdt.Rec):
        return ("rec", kdt_to_db(term.motive, env),
                kdt_to_db(term.base, env), kdt_to_db(term.step, env),
                kdt_to_db(term.target, env))
    raise ValueError("unknown kernel_dt term %r" % (term,))


def convert_case(term, ty, ctx):
    """(term, expected type, ctx dict) -> (term_db, ty_db, ctx_types_db).
    ctx_types_db is in insertion order (outermost first); each context
    type is converted with only the names inserted before it in scope.
    The machine's ctx stack is the reverse (innermost first)."""
    names = list(ctx.keys())
    ctx_types = []
    for i, name in enumerate(names):
        ctx_types.append(kdt_to_db(ctx[name], tuple(names[:i])))
    env = tuple(names)
    term_db = kdt_to_db(term, env)
    ty_db = kdt_to_db(ty, env) if ty is not None else None
    return term_db, ty_db, ctx_types


def convert_defeq(t, u, ctx):
    names = list(ctx.keys())
    ctx_types = []
    for i, name in enumerate(names):
        ctx_types.append(kdt_to_db(ctx[name], tuple(names[:i])))
    env = tuple(names)
    return (kdt_to_db(t, env), kdt_to_db(u, env), ctx_types)


def run_ref_check(term_db, ty_db, ctx_types, max_steps=200000):
    m = RefMachine(term_db, ty_db, ctx_types, entry="check",
                   max_steps=max_steps)
    mode = m.run()
    if mode == "HALT":
        return None, m   # accepted
    return {"path": m.fail_path, "kind": m.fail_kind}, m


def run_ref_infer(term_db, ctx_types, max_steps=200000):
    m = RefMachine(term_db, None, ctx_types, entry="infer",
                   max_steps=max_steps)
    mode = m.run()
    if mode == "HALT":
        ty = decode_tree(m._get_tree(m.res_sel), m.res_ptr)
        return ty, None, m
    return None, {"path": m.fail_path, "kind": m.fail_kind}, m


def run_ref_defeq(t_db, u_db, ctx_types, max_steps=200000):
    m = RefMachine(t_db, u_db, ctx_types, entry="defeq",
                   max_steps=max_steps)
    mode = m.run()
    if mode == "HALT":
        return m.res_bool, None, m
    return None, {"path": m.fail_path, "kind": m.fail_kind}, m


# ---------------------------------------------------------------------------
# The suites, re-declared against the machine (mirroring
# test_kernel_dt.py and test_kernel_dt_nat.py line by line).
# ---------------------------------------------------------------------------

def build_cases():
    """Returns (check_cases, infer_cases, defeq_cases).
    check_case: (name, term, type, ctx, want) where want is None for
    accept or (path, kind) for reject. infer_case: (name, term, ctx,
    want) where want is (path, kind). defeq_case: (name, t, u, ctx, want)."""
    import test_kernel_dt as T1
    import test_kernel_dt_nat as T2
    k = kdt
    A, B, C = T1.A, T1.B, T1.C
    T0, T1u = T1.T0, T1.T1
    ctx0 = T1.ctx0
    ctx_eta = T1.ctx_eta
    ctx_eta2 = T1.ctx_eta2
    ctx_fam = T1.ctx_fam
    id_dep = T1.id_dep
    const_dep = T1.const_dep
    type_fn = T1.type_fn
    S = T1.S
    arr = T1.arr
    N, Z = T2.N, T2.Z
    add, double, ind, ind_ty = T2.add, T2.double, T2.ind, T2.ind_ty
    num = T2.num
    const_motive = T2.const_motive
    add_step = T2.add_step
    ctx_rec = T2.ctx_rec
    ctx_neutral = T2.ctx_neutral
    ctx_a = T2.ctx_a
    P = T2.P

    check_cases = [
        ("dependent identity",
         id_dep, k.Pi("A", T0, k.Pi("x", k.Var("A"), k.Var("A"))), ctx0,
         None),
        ("dependent const",
         const_dep,
         k.Pi("A", T0, k.Pi("B", T0,
                            k.Pi("x", k.Var("A"),
                                 k.Pi("y", k.Var("B"), k.Var("A"))))), ctx0,
         None),
        ("type-level function",
         type_fn, k.Pi("A", T0, T0), ctx0, None),
        ("conversion",
         k.Lam("x", B, k.Var("x")), k.App(type_fn, B), ctx0, None),
        ("(λA:Type 0. A -> A) B is a type",
         k.App(type_fn, B), T0, ctx0, None),
        ("eta through check",
         k.App(k.Var("K"), k.Var("a")),
         k.App(k.Var("H"), k.Lam("B", T0, k.App(k.Var("F"), k.Var("B")))),
         ctx_eta2, None),
        ("Type 0 : Type 1", T0, T1u, {}, None),
        ("Π A:Type 0. A -> A : Type 1",
         k.Pi("A", T0, arr(k.Var("A"), k.Var("A"))), T1u, {}, None),
        ("S", S,
         arr(arr(A, arr(B, C)), arr(arr(A, B), arr(A, C))), ctx0, None),
        ("applied dependent identity",
         k.App(k.App(id_dep, A), k.Var("x")), A,
         {**ctx0, "x": A}, None),
        ("Type 0 : Type 0 (Girard)", T0, T0, {},
         ((), "mismatch")),
        ("dependent application, wrong argument",
         k.App(k.App(id_dep, A), k.Var("y")), A,
         {**ctx0, "y": B}, (("arg",), "mismatch")),
        ("type-level self-application",
         k.Lam("A", T0, k.App(k.Var("A"), k.Var("A"))),
         k.Pi("A", T0, T0), {},
         (("body", "func"), "not-pi")),
        ("Π A:Type 0. A -> A : Type 0 (wrong universe)",
         k.Pi("A", T0, arr(k.Var("A"), k.Var("A"))), T0, {},
         ((), "mismatch")),
        ("capture a2",
         k.App(k.Lam("A", T0, k.Lam("x", A, k.Lam("A", T0, k.Var("x")))),
               B),
         k.Pi("x", B, k.Pi("A2", T0, B)), ctx0, None),
        ("capture family",
         k.Lam("x", A, k.App(k.Var("f"), k.Var("x"))),
         k.Pi("x", A, k.App(k.Var("F"), k.Var("x"))), ctx_fam, None),
        ("Nat : Type 0", N, T0, {}, None),
        ("Zero : Nat", Z, N, {}, None),
        ("Succ Zero : Nat", num(1), N, {}, None),
        ("Succ (Succ Zero) : Nat", num(2), N, {}, None),
        ("add : Nat -> Nat -> Nat", add, arr(N, arr(N, N)), {}, None),
        ("add 2 3 : Nat", k.App(k.App(add, num(2)), num(3)), N, {}, None),
        ("double : Nat -> Nat", double, arr(N, N), {}, None),
        ("induction principle", ind, ind_ty, {}, None),
        ("induction principle type : Type 1", ind_ty, T1u, {}, None),
        ("Succ of a non-Nat",
         k.Succ(k.Lam("x", A, k.Var("x"))), N, ctx_a,
         (("pred",), "mismatch")),
        ("Zero : Nat -> Nat", Z, arr(N, N), {}, ((), "mismatch")),
    ]
    infer_cases = [
        ("binder annotation is not a type",
         k.Lam("x", k.Var("f"), k.Var("x")),
         {**ctx0, "f": arr(A, A)}, (("binder",), "not-a-type")),
        ("motive is not a function",
         k.Rec(N, Z, add_step, Z), {},
         (("motive",), "not-pi")),
        ("motive codomain is not a universe",
         k.Rec(k.Lam("x", N, Z), Z, add_step, Z), {},
         (("motive",), "not-a-type")),
        ("motive codomain is neutral, not a universe",
         k.Rec(k.Var("P"), Z, add_step, Z), ctx_neutral,
         (("motive",), "not-a-type")),
        ("Rec on a non-Nat target",
         k.Rec(const_motive, Z, add_step, k.Lam("x", A, k.Var("x"))),
         ctx_a, (("target",), "mismatch")),
        ("Rec base of the wrong type",
         k.Rec(const_motive, k.Lam("x", A, k.Var("x")), add_step, Z),
         ctx_a, (("base",), "mismatch")),
    ]
    defeq_cases = [
        ("eta: f ≡ λx:A. f x", ctx_eta, k.Var("f"),
         k.Lam("x", A, k.App(k.Var("f"), k.Var("x"))), True),
        ("distinct variables: f ≢ g", ctx_eta, k.Var("f"), k.Var("g"),
         False),
        ("alpha: λx:A. x ≡ λy:A. y", ctx0,
         k.Lam("x", A, k.Var("x")), k.Lam("y", A, k.Var("y")), True),
        ("add 2 3 ≡ 5", {}, k.App(k.App(add, num(2)), num(3)), num(5),
         True),
        ("add 2 3 ≢ 4", {}, k.App(k.App(add, num(2)), num(3)), num(4),
         False),
        ("double 2 ≡ 4", {}, k.App(double, num(2)), num(4), True),
        ("iota on Zero", ctx_rec,
         k.Rec(P, k.Var("z"), k.Var("s"), Z), k.Var("z"), True),
        ("stuck Rec is itself", ctx_rec,
         k.Rec(P, k.Var("z"), k.Var("s"), k.Var("n")),
         k.Rec(P, k.Var("z"), k.Var("s"), k.Var("n")), True),
    ]
    return check_cases, infer_cases, defeq_cases


def run_ref_suites(verbose=True):
    """Run the RefMachine on all 41 suite cases; compare verdicts
    (and failure path/kind for rejections) against kernel_dt.py."""
    check_cases, infer_cases, defeq_cases = build_cases()
    results = []
    worst = {"ctrl": 0, "reg": 0, "ctx": 0, "wstack": 0, "steps": 0}
    for name, term, ty, ctx, want in check_cases:
        term_db, ty_db, ctx_types = convert_case(term, ty, ctx)
        fail, m = run_ref_check(term_db, ty_db, ctx_types)
        for kkey in ("ctrl", "reg", "ctx", "wstack", "steps"):
            v = {"ctrl": m.max_ctrl, "reg": m.max_reg,
                 "ctx": m.max_ctx, "wstack": m.max_wstack,
                 "steps": m.max_steps_used}[kkey]
            if v > worst[kkey]:
                worst[kkey] = v
        if want is None:
            ok = fail is None
            detail = "accepted" if ok else "rejected %r" % (fail,)
        else:
            ok = (fail is not None and tuple(fail["path"]) == tuple(want[0])
                  and fail["kind"] == want[1])
            detail = ("rejected %r" % (fail,) if fail is not None
                      else "accepted")
        results.append((ok, "check  %-38s %s" % (name, detail)))
    for name, term, ctx, want in infer_cases:
        term_db, _, ctx_types = convert_case(term, None, ctx)
        ty, fail, m = run_ref_infer(term_db, ctx_types)
        for kkey in ("ctrl", "reg", "ctx", "wstack", "steps"):
            v = {"ctrl": m.max_ctrl, "reg": m.max_reg,
                 "ctx": m.max_ctx, "wstack": m.max_wstack,
                 "steps": m.max_steps_used}[kkey]
            if v > worst[kkey]:
                worst[kkey] = v
        ok = (fail is not None and tuple(fail["path"]) == tuple(want[0])
              and fail["kind"] == want[1])
        detail = ("rejected %r" % (fail,) if fail is not None
                  else "inferred %r" % (ty,))
        results.append((ok, "infer  %-38s %s" % (name, detail)))
    for name, ctx, t, u, want in defeq_cases:
        t_db, u_db, ctx_types = convert_defeq(t, u, ctx)
        got, fail, m = run_ref_defeq(t_db, u_db, ctx_types)
        for kkey in ("ctrl", "reg", "ctx", "wstack", "steps"):
            v = {"ctrl": m.max_ctrl, "reg": m.max_reg,
                 "ctx": m.max_ctx, "wstack": m.max_wstack,
                 "steps": m.max_steps_used}[kkey]
            if v > worst[kkey]:
                worst[kkey] = v
        ok = (fail is None and got == want)
        detail = ("defeq=%r" % (got,) if fail is None
                  else "error %r" % (fail,))
        results.append((ok, "defeq  %-38s %s" % (name, detail)))
    if verbose:
        for ok, line in results:
            print(("PASS  " if ok else "FAIL  ") + line)
    passed = sum(1 for ok, _ in results if ok)
    if verbose:
        print("\nRefMachine: %d/%d passed" % (passed, len(results)))
        print("measured maxima: ctrl=%d reg=%d ctx=%d wstack=%d steps=%d"
              % (worst["ctrl"], worst["reg"], worst["ctx"],
                 worst["wstack"], worst["steps"]))
    return passed, len(results), worst, results


if __name__ == "__main__":
    p, n, worst, _ = run_ref_suites()
    if p != n:
        raise SystemExit(1)
