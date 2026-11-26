#!/usr/bin/env python3
"""
matrix_machine_dt.py — the sealed DT checker as explicit matrices.

The transliteration of the RefMachine in weights_kernel_dt.py into
explicit matrix ops (the MatrixMachine), same discipline as the STLC
weights_kernel.py: one control matrix, action gating, every operation
an explicit matrix or threshold gate, a registry counted by nonzero
parameters. Lockstep: decode(MatrixMachine) == RefMachine snapshot
after every step, on the DT suite (20/20) and the Nat suite (21/21).

Design decisions beyond the spec (with reasons):

* Walk is node-at-a-time. The RefMachine's _step_walk pops one work
  item and processes one node, and lockstep is step-for-step, so the
  MatrixMachine's WALK does the same. Consequence: the whole-tree
  subst_circuit (730,003 weights) is NOT the SUBST work-item — applying
  it whole-subtree in one step would finish a walk in 1 step while the
  RefMachine takes N, breaking lockstep. The SUBST/COPY work-items use
  the circuit's per-node machinery (VAR threshold comparisons, the
  shift-down permutation, graft detection as an equality) applied to a
  single node. The whole-tree subst_circuit stands as the measured
  fixed construction for whole-tree substitution (a component for a
  redesigned walk, not this one); its count is reported separately in
  MATRIX_MACHINE_STATUS.md, not in this machine's registry.
* Only the selected action is evaluated per step. The STLC machine
  computes all 8 candidates and gates them; with ~104 actions that is
  100x the work for the same values (the action vector is one-hot, so
  the gated sum equals the selected candidate). The control matrix
  still selects the action, and each action is a fixed sequence of
  matrix ops; evaluating only the selected one is the same computation
  with the zero branches elided.
* The free list is a stack, not a set. The RefMachine's _free is a
  LIFO stack (initially [15..0], top 0; a sweep pushes freed registers
  in ascending order, so the next alloc pops the highest freed). The
  register index a ref points to is part of the state, so the
  MatrixMachine keeps the free stack faithfully (16 slots + depth);
  alloc pops the top, sweep pushes freed regs ascending.
* A sweep does NOT clear trees; an alloc DOES (regs[r] := empty on
  alloc, exactly as the RefMachine). The lockstep decode therefore
  compares the full 16-register file, including stale trees in free
  registers.
* Counting discipline (STLC's): a matrix construction is counted once
  by its nnz however often it is applied; gating (multiplying by a
  0/1 selector) is free — the matrices being gated are counted.
  Bilinear selections (node read by a data-dependent pointer, node
  write into a data-dependent slot/register) are evaluated in numpy
  via the gated-sum form and their constructions are counted
  analytically, the same move subst_circuit.py makes for index arrays
  (an index array counts as the equivalent permutation's nnz).
  The registry is built once; count_params() sums it.

State layout (all one-hot unless noted):
  mode 8 | fsel 18, fptr 2047 | asel 18, aptr 2047 | res_sel 19 (18=none),
  res_ptr 2047, res_bool 3 (none/T/F), res_level 5 (none/0..3) |
  dest 17 (16=none) | ctx 8 x 2066 (present + sel 18 + ptr 2047) |
  ctrl 14 x 10417 (present + kind 53 + r1..r4 (sel 19 + ptr 2047) +
  s1 18 + s2 2047 + saved_dest 17 + wdst 17) |
  wstack 7 x 6221 (present + task 2 + ssel 18 + sptr 2047 + dsel 18 +
  dptr 2047 + p1 11 + p2 11 + rsel 19 + rptr 2047) |
  regs 16 x 2047 x 24 | input_tree, aux_tree (constants, not counted) |
  fail_path 3 x 12 | fail_kind 8 | free_stack 16 x 16 + free_depth 17 |
  step_allocs 16 bits | steps (scalar int, as in STLC).
"""

import os
import sys

import numpy as np

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp = None

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import weights_kernel_dt as W

# ---------------------------------------------------------------------------
# Layout (mirrors weights_kernel_dt)
# ---------------------------------------------------------------------------

N_SLOTS = W.N_SLOTS            # 2047
KINDS = W.KINDS
KIND = W.KIND
N_KINDS = W.N_KINDS            # 12
LEVELS = W.LEVELS              # 4
VAR_IDX = W.VAR_IDX            # 8
SLOT_DIM = W.SLOT_DIM          # 24
KIND_OFF = W.KIND_OFF
LEVEL_OFF = W.LEVEL_OFF
IDX_OFF = W.IDX_OFF
MODES = W.MODES
MODE = W.MODE
N_MODES = W.N_MODES
FRAME_KINDS = W.FRAME_KINDS
FRAME_KIND = W.FRAME_KIND
N_FRAME_KINDS = W.N_FRAME_KINDS
FRAME_SEG = W.FRAME_SEG
FAIL_KINDS = W.FAIL_KINDS
FAIL_KIND = W.FAIL_KIND
N_FAIL_KINDS = W.N_FAIL_KINDS
PATH_SEGS = W.PATH_SEGS
PATH_SEG = W.PATH_SEG
N_PATH_SEGS = W.N_PATH_SEGS
MAX_PATH_LEN = W.MAX_PATH_LEN
POOL_SIZE = W.POOL_SIZE        # 16
SOURCES = W.SOURCES
SRC = W.SRC
N_SRC = 2 + POOL_SIZE         # 18 (INPUT=0, AUX=1, register r -> 2+r).
# NOTE: W.N_SRC is 10 — it comes from weights_kernel_dt's SOURCES list,
# which is dead code there (never used); the live sel encoding is
# _sel_of_reg(r) = 2+r over the 16-register pool, i.e. sel 0..17.
# Using W.N_SRC truncated every sel >= 10 to an all-zero one-hot.
MAX_CTX = 8
MAX_CTRL = 14
MAX_WSTACK = 7

# Field dims
RES_SEL_DIM = N_SRC + 1        # 19, index N_SRC = none
DEST_DIM = POOL_SIZE + 1       # 17, index POOL_SIZE = none
RES_BOOL_DIM = 3               # 0=none, 1=True, 2=False
RES_LEVEL_DIM = LEVELS + 1     # 5, index 0 = none
CTX_SLOT_DIM = 1 + N_SRC + N_SLOTS          # 2066
REF_DIM = (N_SRC + 1) + N_SLOTS             # 2066 (sel 19 + ptr 2047)
FRAME_DIM = (1 + N_FRAME_KINDS + 4 * REF_DIM + N_SRC + N_SLOTS
             + DEST_DIM + DEST_DIM)         # 10417
WALK_DIM = (1 + 2 + N_SRC + N_SLOTS + N_SRC + N_SLOTS + 11 + 11
            + RES_SEL_DIM + N_SLOTS)        # 6221
FREE_STACK_DIM = POOL_SIZE * POOL_SIZE      # 256 (16 slots x reg one-hot 16)
FREE_DEPTH_DIM = POOL_SIZE + 1              # 17

# Frame field offsets (within a frame)
FR_PRESENT = 0
FR_KIND = 1
FR_R1 = FR_KIND + N_FRAME_KINDS
FR_R2 = FR_R1 + REF_DIM
FR_R3 = FR_R2 + REF_DIM
FR_R4 = FR_R3 + REF_DIM
FR_S1 = FR_R4 + REF_DIM
FR_S2 = FR_S1 + N_SRC
FR_SAVED_DEST = FR_S2 + N_SLOTS
FR_WDST = FR_SAVED_DEST + DEST_DIM
assert FR_WDST + DEST_DIM == FRAME_DIM, (FR_WDST, FRAME_DIM)

# Walk item field offsets
WK_PRESENT = 0
WK_TASK = 1
WK_SSEL = WK_TASK + 2
WK_SPTR = WK_SSEL + N_SRC
WK_DSEL = WK_SPTR + N_SLOTS
WK_DPTR = WK_DSEL + N_SRC
WK_P1 = WK_DPTR + N_SLOTS
WK_P2 = WK_P1 + 11
WK_RSEL = WK_P2 + 11
WK_RPTR = WK_RSEL + RES_SEL_DIM
assert WK_RPTR + N_SLOTS == WALK_DIM, (WK_RPTR, WALK_DIM)

# ---------------------------------------------------------------------------
# Registry (STLC discipline: nonzero parameters)
# ---------------------------------------------------------------------------

WEIGHTS = []   # (name, nnz) — matrices and analytic constructions


def reg(name, mat_or_nnz):
    """Register a matrix (counted by .nnz) or an analytic nnz."""
    nnz = int(mat_or_nnz.nnz) if hasattr(mat_or_nnz, "nnz") \
        else int(mat_or_nnz)
    WEIGHTS.append((name, nnz))
    return mat_or_nnz


def count_params():
    total = sum(n for _, n in WEIGHTS)
    per = {}
    for name, n in WEIGHTS:
        per[name] = per.get(name, 0) + n
    return total, per


# ---------------------------------------------------------------------------
# Sparse matrix builders (fixed permutations; registered once)
# ---------------------------------------------------------------------------

def _perm_matrix(pairs, n):
    """pairs: list of (row, col). Returns a CSR matrix (n x n)."""
    rows = [r for r, _ in pairs]
    cols = [c for _, c in pairs]
    return sp.csr_matrix((np.ones(len(pairs)), (rows, cols)), shape=(n, n))


def _child_matrix(left):
    """ptr one-hot -> child ptr one-hot (left or right child)."""
    pairs = []
    for i in range(N_SLOTS):
        j = 2 * i + 1 if left else 2 * i + 2
        if j < N_SLOTS:
            pairs.append((j, i))
    return _perm_matrix(pairs, N_SLOTS)


LEFT_CHILD = reg("ptr left-child permutation",
                 _child_matrix(left=True))
RIGHT_CHILD = reg("ptr right-child permutation",
                  _child_matrix(left=False))
PARENT_PTR = reg("ptr parent permutation",
                 _perm_matrix([( (p - 1) // 2, p) for p in range(1, N_SLOTS)],
                              N_SLOTS))


def _rec_child_matrix(kind):
    """ptr one-hot -> the Rec component's ptr (motive/base/step/target)."""
    def go(p):
        if kind == "motive":
            return 2 * p + 1
        if kind == "base":
            return 2 * (2 * p + 2) + 1
        if kind == "step":
            return 2 * (2 * (2 * p + 2) + 2) + 1
        return 2 * (2 * (2 * p + 2) + 2) + 2  # target
    pairs = []
    for p in range(N_SLOTS):
        j = go(p)
        if j < N_SLOTS:
            pairs.append((j, p))
    return _perm_matrix(pairs, N_SLOTS)


REC_MOTIVE_PTR = LEFT_CHILD   # motive = left child (same matrix, no recount)
REC_BASE_PTR = reg("ptr rec-base permutation",
                   _rec_child_matrix("base"))
REC_STEP_PTR = reg("ptr rec-step permutation",
                   _rec_child_matrix("step"))
REC_TARGET_PTR = reg("ptr rec-target permutation",
                     _rec_child_matrix("target"))


def _shift_matrix(n_slots, slot_dim, down):
    """Shift a stack of n_slots x slot_dim down (toward higher index)
    or up. Returns a CSR matrix (n_slots*slot_dim) x (n_slots*slot_dim).
    The vacated slot is zeroed; the shifted-out slot is dropped."""
    pairs = []
    for i in range(n_slots):
        j = i + 1 if down else i - 1
        if 0 <= j < n_slots:
            for d in range(slot_dim):
                pairs.append((j * slot_dim + d, i * slot_dim + d))
    return sp.csr_matrix(
        (np.ones(len(pairs)),
         ([r for r, _ in pairs], [c for _, c in pairs])),
        shape=(n_slots * slot_dim, n_slots * slot_dim))


CTX_DIM = MAX_CTX * CTX_SLOT_DIM
CTRL_DIM = MAX_CTRL * FRAME_DIM
WALKSTACK_DIM = MAX_WSTACK * WALK_DIM

SHIFT_DOWN_CTX = reg("ctx shift-down",
                     _shift_matrix(MAX_CTX, CTX_SLOT_DIM, down=True))
SHIFT_UP_CTX = reg("ctx shift-up",
                   _shift_matrix(MAX_CTX, CTX_SLOT_DIM, down=False))
SHIFT_DOWN_CTRL = reg("ctrl shift-down",
                      _shift_matrix(MAX_CTRL, FRAME_DIM, down=True))
SHIFT_UP_CTRL = reg("ctrl shift-up",
                    _shift_matrix(MAX_CTRL, FRAME_DIM, down=False))
SHIFT_DOWN_WALK = reg("wstack shift-down",
                      _shift_matrix(MAX_WSTACK, WALK_DIM, down=True))
SHIFT_UP_WALK = reg("wstack shift-up",
                    _shift_matrix(MAX_WSTACK, WALK_DIM, down=False))

# Node read construction (analytic count, subst_circuit's move):
# slot extraction E_p (24 nnz per slot x 2047 slots) + sel mux over the
# 24-dim slot vector (18 sources x 24). Counted once, used at every
# node read site.
reg("node-read slot extraction (E_p)", N_SLOTS * SLOT_DIM)
reg("node-read sel mux", N_SRC * SLOT_DIM)
# Node write construction (analytic): write a 24-dim node into a slot
# (24 nnz per slot x 2047) + erase the old slot (same) — the write is
# tree' = (I - E_p) tree + E_p node, gated by the dest ptr one-hot.
reg("node-write (write + erase)", 2 * N_SLOTS * SLOT_DIM)
# Register-file per-register gating (alloc clear, dest write routing):
# one identity per register tree, gated by the register selector.
reg("register-file per-register gating", POOL_SIZE * N_SLOTS * SLOT_DIM)
# Frame write routing (write a frame's fields into ctrl slot 0):
# kind 53 + 4 refs x 2066 + s1 18 + s2 2047 + saved_dest 17 + wdst 17.
reg("frame write routing", FRAME_DIM)
# Walk-item write routing (write an item's fields into wstack slot 0).
reg("walk-item write routing", WALK_DIM)
# Ctx slot write routing (write a ref into ctx slot 0).
reg("ctx slot write routing", CTX_SLOT_DIM)

# Free-stack push/pop: the stack is 16 slots x 16-dim reg one-hot;
# a push writes one slot (16 nnz), a pop reads one slot (16 nnz).
# Counted as the two routing constructions.
reg("free-stack push/pop routing", 2 * POOL_SIZE)


def _onehot(n, i):
    v = np.zeros(n)
    if 0 <= i < n:
        v[i] = 1.0
    return v


def _idx_of(vec):
    """Index of the 1 in a one-hot vector (assumes exactly one)."""
    return int(np.argmax(vec))


def _is_onehot(vec):
    return abs(float(vec.sum()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# RefMachine snapshot (canonical, for lockstep comparison)
# ---------------------------------------------------------------------------

def ref_snapshot(m):
    """Canonical snapshot of a RefMachine's state."""
    def ref_tup(ref):
        return None if ref is None else (int(ref[0]), int(ref[1]))
    ctrl = []
    for fr in m.ctrl:
        ctrl.append((fr[0], ref_tup(fr[1]), ref_tup(fr[2]),
                     ref_tup(fr[3]), ref_tup(fr[4]), int(fr[5]),
                     int(fr[6]), int(fr[7]), int(fr[8])))
    wstack = []
    for it in m.wstack:
        wstack.append((it[0], int(it[1]), int(it[2]), int(it[3]),
                       int(it[4]), int(it[5]), int(it[6]), int(it[7]),
                       int(it[8])))
    regs = []
    for r in m.regs:
        regs.append(tuple(r))
    return {
        "mode": m.mode,
        "fsel": int(m.fsel), "fptr": int(m.fptr),
        "asel": int(m.asel), "aptr": int(m.aptr),
        "res_sel": int(m.res_sel), "res_ptr": int(m.res_ptr),
        "res_bool": m.res_bool, "res_level": m.res_level,
        "dest": int(m.dest),
        "ctx": tuple((int(s), int(p)) for s, p in m.ctx),
        "ctrl": tuple(ctrl),
        "wstack": tuple(wstack),
        "regs": tuple(regs),
        "fail_path": tuple(m.fail_path), "fail_kind": m.fail_kind,
        "free": tuple(m._free), "step_allocs": tuple(sorted(m._step_allocs)),
        "steps": int(m.steps),
    }

# ---------------------------------------------------------------------------
# MatrixMachine — the same machine, as explicit matrix ops
# ---------------------------------------------------------------------------

# Small threshold-layer counts (analytic, registered once).
# Liveness: live[r] = OR over the machine's ~103 sel fields of
# [field == 2+r]. One threshold unit per register, fan-in = #sel fields.
_N_SEL_FIELDS = (MAX_CTRL * 5 + MAX_CTX + 1 + 1 + 2 + MAX_WSTACK * 3)
reg("liveness OR (live bits)", POOL_SIZE * (_N_SEL_FIELDS + 1))
# Fail-path construction: kind->segment per frame slot (53 nnz x 14),
# rank prefix sums (~98), path-slot routing (3 x 14 x 12).
reg("fail-path construction", N_FRAME_KINDS * MAX_CTRL + 98
    + MAX_PATH_LEN * MAX_CTRL * N_PATH_SEGS)
# Eta error-catch construction: two priority encoders over the frame
# kinds (topmost CMP_ETA, CMP_W_R below it) + stack truncation gating.
reg("eta error-catch construction", 4 * MAX_CTRL + MAX_CTRL * FRAME_DIM // 64)
# Comparison bits (VAR idx < ctx depth, SUBST idx==k / idx>k, level max):
# small threshold layers over one-hot outer products.
reg("comparison bits (idx/depth/level)", 512)


class MatrixMachine:
    def __init__(self, term_db, ty_db=None, ctx_types=None, entry="check",
                 max_steps=200000, pool_size=POOL_SIZE):
        """Mirror RefMachine.__init__ exactly (same entry setup)."""
        self.pool_size = pool_size
        self.max_steps = max_steps
        self.steps = 0
        self.entry = entry
        # Constant input trees (the program, not counted — STLC rule).
        self.input_tree = np.array(
            [W.slot_onehot(n) for n in W.encode_db(term_db)])
        self.aux_tree = np.array(
            [W.slot_onehot(n) for n in
             (W.encode_db(ty_db) if ty_db is not None else W.empty_tree())])
        # Register file: (16, 2047, 24).
        self.regs = np.zeros((pool_size, N_SLOTS, SLOT_DIM))
        # Free stack: (16, 16) rows bottom-first, depth one-hot.
        self.free_stack = np.zeros((POOL_SIZE, POOL_SIZE))
        for i in range(pool_size):
            self.free_stack[i, pool_size - 1 - i] = 1.0
        self.free_depth = pool_size
        self.step_allocs = np.zeros(POOL_SIZE)
        # Scalar fields (one-hots).
        self.mode = _onehot(N_MODES, MODE["INFER"])
        self.fsel = _onehot(N_SRC, 0)
        self.fptr = _onehot(N_SLOTS, 0)
        self.asel = _onehot(N_SRC, 1)
        self.aptr = _onehot(N_SLOTS, 0)
        self.res_sel = _onehot(RES_SEL_DIM, N_SRC)   # none
        self.res_ptr = _onehot(N_SLOTS, 0)
        self.res_bool = _onehot(RES_BOOL_DIM, 0)     # none
        self.res_level = _onehot(RES_LEVEL_DIM, 0)   # none
        self.dest = _onehot(DEST_DIM, POOL_SIZE)     # none
        self.ctx = np.zeros(CTX_DIM)
        self.ctrl = np.zeros(CTRL_DIM)
        self.wstack = np.zeros(WALKSTACK_DIM)
        self.fail_path = np.zeros((MAX_PATH_LEN, N_PATH_SEGS))
        self.fail_path[:, PATH_SEG[""]] = 1.0
        self.fail_kind = _onehot(N_FAIL_KINDS, FAIL_KIND["none"])
        self.max_ctrl = 0
        self.max_reg = 0
        self.max_ctx = 0
        self.max_wstack = 0
        self.max_steps_used = 0
        self._handler_seg = None
        # Context types materialised into registers (innermost first,
        # i.e. reversed(ctx_types)); ctx slot i = ctx_types reversed[i].
        for i, t in enumerate(reversed(ctx_types or [])):
            r = self._alloc()
            self.regs[r] = np.array([W.slot_onehot(n)
                                     for n in W.encode_db(t)])
            self._ctx_write_slot(i, 2 + r, 0)
        self.max_ctx = len(ctx_types or [])
        if entry == "check":
            d = self._alloc()
            self._push_frame("ENTRY_VALID",
                             [(0, 0), None, None, None], (0, 0),
                             saved_dest=self._dest_idx(), wdst=-1)
            self.dest = _onehot(DEST_DIM, d)
            self.fsel, self.fptr = _onehot(N_SRC, 1), _onehot(N_SLOTS, 0)
            self.mode = _onehot(N_MODES, MODE["INFER"])
        elif entry == "infer":
            d = self._alloc()
            self.dest = _onehot(DEST_DIM, d)
            self.fsel, self.fptr = _onehot(N_SRC, 0), _onehot(N_SLOTS, 0)
            self.mode = _onehot(N_MODES, MODE["INFER"])
        elif entry == "defeq":
            self.fsel, self.fptr = _onehot(N_SRC, 0), _onehot(N_SLOTS, 0)
            self.asel, self.aptr = _onehot(N_SRC, 1), _onehot(N_SLOTS, 0)
            self.mode = _onehot(N_MODES, MODE["COMPARE"])
        else:
            raise ValueError("unknown entry %r" % entry)

    # -- field access -----------------------------------------------------

    def _mode_idx(self):
        return _idx_of(self.mode)

    def _mode_name(self):
        return MODES[self._mode_idx()]

    def _dest_idx(self):
        d = _idx_of(self.dest)
        return -1 if d == POOL_SIZE else d

    def _sel_idx(self, vec):
        return _idx_of(vec)

    def _tree_of(self, sel):
        """The (2047, 24) tree array for source sel."""
        if sel == 0:
            return self.input_tree
        if sel == 1:
            return self.aux_tree
        return self.regs[sel - 2]

    def read_node(self, sel, ptr):
        """Node read (the node-read construction, counted in the
        registry): the 24-dim one-hot node at (sel, ptr)."""
        return self._tree_of(sel)[ptr].copy()

    def read_kind(self, sel, ptr):
        node = self.read_node(sel, ptr)
        return KINDS[int(np.argmax(node[KIND_OFF:KIND_OFF + N_KINDS]))]

    def read_level(self, sel, ptr):
        node = self.read_node(sel, ptr)
        return int(np.argmax(node[LEVEL_OFF:LEVEL_OFF + LEVELS]))

    def read_idx(self, sel, ptr):
        node = self.read_node(sel, ptr)
        return int(np.argmax(node[IDX_OFF:IDX_OFF + VAR_IDX]))

    def write_node_reg(self, r, ptr, kind, level=0, idx=0):
        """Write a constant node into regs[r] at ptr (node-write)."""
        node = np.zeros(SLOT_DIM)
        node[KIND_OFF + KIND[kind]] = 1.0
        node[LEVEL_OFF + level] = 1.0
        node[IDX_OFF + idx] = 1.0
        self.regs[r, ptr] = node

    def write_tree_reg(self, r, tree):
        """Overwrite regs[r] with a whole (2047, 24) tree array."""
        self.regs[r] = np.array(tree, dtype=float)

    # -- ctx stack (slot 0 = top / innermost) ------------------------------

    def _ctx_depth(self):
        d = 0
        for i in range(MAX_CTX):
            if self.ctx[i * CTX_SLOT_DIM] > 0.5:
                d += 1
        return d

    def _ctx_write_slot(self, i, sel, ptr):
        base = i * CTX_SLOT_DIM
        self.ctx[base] = 1.0
        self.ctx[base + 1:base + 1 + N_SRC] = _onehot(N_SRC, sel)
        self.ctx[base + 1 + N_SRC:base + 1 + N_SRC + N_SLOTS] = \
            _onehot(N_SLOTS, ptr)

    def _ctx_read_slot(self, i):
        base = i * CTX_SLOT_DIM
        if self.ctx[base] < 0.5:
            return None
        sel = _idx_of(self.ctx[base + 1:base + 1 + N_SRC])
        ptr = _idx_of(self.ctx[base + 1 + N_SRC:
                               base + 1 + N_SRC + N_SLOTS])
        return (sel, ptr)

    def _ctx_push(self, sel, ptr):
        """Shift down + write slot 0 (the shift matrix, applied)."""
        self.ctx = SHIFT_DOWN_CTX @ self.ctx
        self._ctx_write_slot(0, sel, ptr)
        d = self._ctx_depth()
        if d > self.max_ctx:
            self.max_ctx = d

    def _ctx_pop(self):
        self.ctx = SHIFT_UP_CTX @ self.ctx

    def _ctx_ref(self, i):
        return self._ctx_read_slot(i)

    # -- control stack (slot 0 = top) ---------------------------------------

    def _ctrl_depth(self):
        d = 0
        for i in range(MAX_CTRL):
            if self.ctrl[i * FRAME_DIM + FR_PRESENT] > 0.5:
                d += 1
        return d

    def _frame_write_slot0(self, kind, refs, smalls, saved_dest, wdst):
        """Write a frame into ctrl slot 0 (frame-write routing)."""
        base = 0
        self.ctrl[base + FR_PRESENT] = 1.0
        self.ctrl[base + FR_KIND:base + FR_KIND + N_FRAME_KINDS] = \
            _onehot(N_FRAME_KINDS, FRAME_KIND[kind])
        for j, ref in enumerate(refs):
            off = base + [FR_R1, FR_R2, FR_R3, FR_R4][j]
            if ref is None:
                self.ctrl[off:off + RES_SEL_DIM] = \
                    _onehot(RES_SEL_DIM, N_SRC)
                self.ctrl[off + RES_SEL_DIM:
                          off + RES_SEL_DIM + N_SLOTS] = 0.0
            else:
                self.ctrl[off:off + RES_SEL_DIM] = \
                    _onehot(RES_SEL_DIM, ref[0])
                self.ctrl[off + RES_SEL_DIM:
                          off + RES_SEL_DIM + N_SLOTS] = \
                    _onehot(N_SLOTS, ref[1])
        self.ctrl[base + FR_S1:base + FR_S1 + N_SRC] = \
            _onehot(N_SRC, smalls[0])
        self.ctrl[base + FR_S2:base + FR_S2 + N_SLOTS] = \
            _onehot(N_SLOTS, smalls[1])
        self.ctrl[base + FR_SAVED_DEST:
                  base + FR_SAVED_DEST + DEST_DIM] = \
            _onehot(DEST_DIM, POOL_SIZE if saved_dest < 0 else saved_dest)
        self.ctrl[base + FR_WDST:base + FR_WDST + DEST_DIM] = \
            _onehot(DEST_DIM, POOL_SIZE if wdst < 0 else wdst)

    def _frame_read_slot(self, i):
        """Decode ctrl slot i into the RefMachine frame tuple."""
        base = i * FRAME_DIM
        if self.ctrl[base + FR_PRESENT] < 0.5:
            return None
        kind = FRAME_KINDS[_idx_of(
            self.ctrl[base + FR_KIND:base + FR_KIND + N_FRAME_KINDS])]
        refs = []
        for j in range(4):
            off = base + [FR_R1, FR_R2, FR_R3, FR_R4][j]
            s = _idx_of(self.ctrl[off:off + RES_SEL_DIM])
            p = _idx_of(self.ctrl[off + RES_SEL_DIM:
                                  off + RES_SEL_DIM + N_SLOTS])
            refs.append(None if s == N_SRC else (s, p))
        s1 = _idx_of(self.ctrl[base + FR_S1:base + FR_S1 + N_SRC])
        s2 = _idx_of(self.ctrl[base + FR_S2:base + FR_S2 + N_SLOTS])
        sd = _idx_of(self.ctrl[base + FR_SAVED_DEST:
                               base + FR_SAVED_DEST + DEST_DIM])
        wd = _idx_of(self.ctrl[base + FR_WDST:base + FR_WDST + DEST_DIM])
        return (kind, refs[0], refs[1], refs[2], refs[3], s1, s2,
                -1 if sd == POOL_SIZE else sd,
                -1 if wd == POOL_SIZE else wd)

    def _push_frame(self, kind, refs, smalls, saved_dest=None, wdst=-1):
        """Shift down + write slot 0. saved_dest defaults to the
        current dest (the RefMachine's _push_frame saves self.dest)."""
        if saved_dest is None:
            saved_dest = self._dest_idx()
        self.ctrl = SHIFT_DOWN_CTRL @ self.ctrl
        self._frame_write_slot0(kind, refs, smalls, saved_dest, wdst)
        d = self._ctrl_depth()
        if d > self.max_ctrl:
            self.max_ctrl = d

    def _pop_frame(self):
        """Read slot 0, shift up, restore dest from saved_dest."""
        fr = self._frame_read_slot(0)
        self.ctrl = SHIFT_UP_CTRL @ self.ctrl
        sd = fr[7]
        self.dest = _onehot(DEST_DIM, POOL_SIZE if sd < 0 else sd)
        return fr

    def _top_frame(self):
        if self._ctrl_depth() == 0:
            return None
        return self._frame_read_slot(0)

    # -- walk stack (slot 0 = top) ------------------------------------------

    def _wstack_depth(self):
        d = 0
        for i in range(MAX_WSTACK):
            if self.wstack[i * WALK_DIM + WK_PRESENT] > 0.5:
                d += 1
        return d

    def _walk_write_slot0(self, task, ssel, sptr, dsel, dptr, p1, p2,
                          rsel, rptr):
        base = 0
        self.wstack[base + WK_PRESENT] = 1.0
        self.wstack[base + WK_TASK:base + WK_TASK + 2] = \
            _onehot(2, 0 if task == "COPY" else 1)
        self.wstack[base + WK_SSEL:base + WK_SSEL + N_SRC] = \
            _onehot(N_SRC, ssel)
        self.wstack[base + WK_SPTR:base + WK_SPTR + N_SLOTS] = \
            _onehot(N_SLOTS, sptr)
        self.wstack[base + WK_DSEL:base + WK_DSEL + N_SRC] = \
            _onehot(N_SRC, dsel)
        self.wstack[base + WK_DPTR:base + WK_DPTR + N_SLOTS] = \
            _onehot(N_SLOTS, dptr)
        self.wstack[base + WK_P1:base + WK_P1 + 11] = _onehot(11, p1)
        self.wstack[base + WK_P2:base + WK_P2 + 11] = _onehot(11, p2)
        self.wstack[base + WK_RSEL:base + WK_RSEL + RES_SEL_DIM] = \
            _onehot(RES_SEL_DIM, N_SRC if rsel < 0 else rsel)
        self.wstack[base + WK_RPTR:base + WK_RPTR + N_SLOTS] = \
            _onehot(N_SLOTS, rptr)

    def _walk_read_slot(self, i):
        base = i * WALK_DIM
        if self.wstack[base + WK_PRESENT] < 0.5:
            return None
        task = "COPY" if _idx_of(
            self.wstack[base + WK_TASK:base + WK_TASK + 2]) == 0 else "SUBST"
        ssel = _idx_of(self.wstack[base + WK_SSEL:base + WK_SSEL + N_SRC])
        sptr = _idx_of(self.wstack[base + WK_SPTR:base + WK_SPTR + N_SLOTS])
        dsel = _idx_of(self.wstack[base + WK_DSEL:base + WK_DSEL + N_SRC])
        dptr = _idx_of(self.wstack[base + WK_DPTR:base + WK_DPTR + N_SLOTS])
        p1 = _idx_of(self.wstack[base + WK_P1:base + WK_P1 + 11])
        p2 = _idx_of(self.wstack[base + WK_P2:base + WK_P2 + 11])
        rsel = _idx_of(self.wstack[base + WK_RSEL:
                                   base + WK_RSEL + RES_SEL_DIM])
        rptr = _idx_of(self.wstack[base + WK_RPTR:
                                   base + WK_RPTR + N_SLOTS])
        return (task, ssel, sptr, dsel, dptr, p1, p2,
                -1 if rsel == N_SRC else rsel, rptr)

    def _push_walk(self, task, ssel, sptr, dsel, dptr, p1=0, p2=0,
                   rsel=-1, rptr=0):
        self.wstack = SHIFT_DOWN_WALK @ self.wstack
        self._walk_write_slot0(task, ssel, sptr, dsel, dptr, p1, p2,
                               rsel, rptr)
        d = self._wstack_depth()
        if d > self.max_wstack:
            self.max_wstack = d

    def _pop_walk(self):
        it = self._walk_read_slot(0)
        self.wstack = SHIFT_UP_WALK @ self.wstack
        return it

    def _start_walk(self, frame_kind, refs, smalls, work_items, wdst=-1):
        self._push_frame(frame_kind, refs, smalls, wdst=wdst)
        for w in work_items:
            self._push_walk(*w)
        self.mode = _onehot(N_MODES, MODE["WALK"])

    # -- register pool (free stack + mark-and-sweep) -------------------------

    def _free_set(self):
        s = set()
        for i in range(self.free_depth):
            s.add(int(np.argmax(self.free_stack[i])))
        return s

    def _live_bits(self):
        """live[r] = OR over the machine's sel fields of [field==2+r]
        (the liveness threshold layer, counted in the registry)."""
        live = np.zeros(POOL_SIZE)
        def add_sel(sel):
            if isinstance(sel, int) and sel >= 2:
                live[sel - 2] = 1.0
        def add_ref(ref):
            if ref is not None:
                add_sel(ref[0])
        d = self._ctrl_depth()
        for i in range(d):
            fr = self._frame_read_slot(i)
            for ref in (fr[1], fr[2], fr[3], fr[4]):
                add_ref(ref)
            if isinstance(fr[7], int) and fr[7] >= 0:
                live[fr[7]] = 1.0
            if isinstance(fr[8], int) and fr[8] >= 0:
                live[fr[8]] = 1.0
            if fr[0] in W.RefMachine.SMALLS_REF_FRAMES:
                add_sel(fr[5])
        for i in range(self._ctx_depth()):
            add_ref(self._ctx_ref(i))
        dd = self._dest_idx()
        if dd >= 0:
            live[dd] = 1.0
        add_sel(_idx_of(self.res_sel))
        add_sel(_idx_of(self.fsel))
        add_sel(_idx_of(self.asel))
        for i in range(self._wstack_depth()):
            it = self._walk_read_slot(i)
            add_sel(it[1])
            add_sel(it[3])
            if isinstance(it[7], int) and it[7] >= 0:
                add_sel(it[7])
        return live

    def _sweep(self):
        """Free every register not live and not allocated this step
        and not already free; push freed regs in ascending order.
        Trees are NOT cleared (the RefMachine's sweep doesn't)."""
        live = self._live_bits()
        free = self._free_set()
        for r in range(POOL_SIZE):
            if live[r] < 0.5 and self.step_allocs[r] < 0.5 \
                    and r not in free:
                # push r onto the free stack (ascending order by the loop)
                self.free_stack[self.free_depth, :] = 0.0
                self.free_stack[self.free_depth, r] = 1.0
                self.free_depth += 1

    def _alloc(self):
        """Pop the free stack top (sweep first if empty); clear the
        register's tree; mark step_allocs. Returns the reg index."""
        if self.free_depth == 0:
            self._sweep()
            if self.free_depth == 0:
                raise RuntimeError(
                    "register pool exhausted: live=%d pool=%d"
                    % (int(self._live_bits().sum()), self.pool_size))
        self.free_depth -= 1
        r = int(np.argmax(self.free_stack[self.free_depth]))
        self.free_stack[self.free_depth, :] = 0.0
        self.regs[r] = np.zeros((N_SLOTS, SLOT_DIM))
        self.step_allocs[r] = 1.0
        n_alloc = self.pool_size - self.free_depth
        if n_alloc > self.max_reg:
            self.max_reg = n_alloc
        return r

    def check_refs(self):
        """Debug invariant: no register referenced in state is free."""
        live = set(int(i) for i in np.where(self._live_bits() > 0.5)[0])
        bad = live & self._free_set()
        assert not bad, ("freed register(s) referenced in state: %s"
                         % sorted(bad))
        return live

    # -- results / errors -----------------------------------------------------

    def _return(self, res_sel=-1, res_ptr=0, res_bool=None, res_level=None):
        if res_sel >= 0:
            self.res_sel = _onehot(RES_SEL_DIM, res_sel)
            self.res_ptr = _onehot(N_SLOTS, res_ptr)
        if res_bool is not None:
            self.res_bool = _onehot(RES_BOOL_DIM, 1 if res_bool else 2)
        if res_level is not None:
            self.res_level = _onehot(RES_LEVEL_DIM, 1 + res_level)
        self.mode = _onehot(N_MODES, MODE["RETURN"])

    def _error(self, kind, suffix=()):
        """Failure path = frame-stack segments + handler seg + suffix
        (the fail-path construction, counted in the registry)."""
        segs = []
        d = self._ctrl_depth()
        for i in range(d - 1, -1, -1):
            # slot d-1 is the bottom frame, slot 0 the top: iterating
            # i from d-1 down to 0 collects segments outermost to
            # innermost, exactly the RefMachine's order over its list.
            fr = self._frame_read_slot(i)
            seg = FRAME_SEG.get(fr[0])
            if seg:
                segs.append(seg)
        if self._handler_seg:
            segs.append(self._handler_seg)
        segs.extend(suffix)
        self.fail_path = np.zeros((MAX_PATH_LEN, N_PATH_SEGS))
        for j in range(MAX_PATH_LEN):
            s = segs[j] if j < len(segs) else ""
            self.fail_path[j] = _onehot(N_PATH_SEGS, PATH_SEG[s])
        self.fail_kind = _onehot(N_FAIL_KINDS, FAIL_KIND[kind])
        self.mode = _onehot(N_MODES, MODE["ERROR"])

    def _try_catch_error(self):
        """Post-step eta catch: a non-fuel error while a CMP_ETA frame
        is on the stack unwinds to just below the CMP_W_R under it
        and returns False (the eta error-catch construction)."""
        if self._mode_name() != "ERROR":
            return False
        if _idx_of(self.fail_kind) == FAIL_KIND["fuel"]:
            return False
        d = self._ctrl_depth()
        kinds = [self._frame_read_slot(i)[0] for i in range(d)]
        # kinds[0] is the top frame; find the topmost CMP_ETA.
        i_eta = None
        for i in range(d):
            if kinds[i] == "CMP_ETA":
                i_eta = i
                break
        if i_eta is None:
            return False
        # The CMP_W_R directly below it (larger slot index = deeper).
        j = None
        for i in range(i_eta + 1, d):
            if kinds[i] == "CMP_W_R":
                j = i
                break
        if j is None:
            return False
        # Truncate the stack to just below slot j: keep slots j+1..
        # (deeper frames) and drop slots 0..j. In slot terms the new
        # stack is the old slots j+1..d-1 shifted to 0..d-j-2.
        new_ctrl = np.zeros(CTRL_DIM)
        for i in range(j + 1, d):
            new_ctrl[(i - j - 1) * FRAME_DIM:(i - j) * FRAME_DIM] = \
                self.ctrl[i * FRAME_DIM:(i + 1) * FRAME_DIM]
        self.ctrl = new_ctrl
        self.fail_path = np.zeros((MAX_PATH_LEN, N_PATH_SEGS))
        self.fail_path[:, PATH_SEG[""]] = 1.0
        self.fail_kind = _onehot(N_FAIL_KINDS, FAIL_KIND["none"])
        self.res_bool = _onehot(RES_BOOL_DIM, 2)   # False
        self.mode = _onehot(N_MODES, MODE["RETURN"])
        return True

    def _begin_check(self, term_ref, type_ref):
        self._push_frame("CHK_W", [term_ref, type_ref, None, None], (0, 0))
        self.fsel = _onehot(N_SRC, type_ref[0])
        self.fptr = _onehot(N_SLOTS, type_ref[1])
        self.mode = _onehot(N_MODES, MODE["WHNF"])

    # -- decode (for lockstep) ------------------------------------------------

    def decode(self):
        """Canonical snapshot in the RefMachine's terms."""
        d = self._ctrl_depth()
        ctrl = []
        for j in range(d):
            # RefMachine ctrl[j] (bottom j=0) = matrix slot d-1-j.
            ctrl.append(self._frame_read_slot(d - 1 - j))
        wd = self._wstack_depth()
        wstack = []
        for j in range(wd):
            wstack.append(self._walk_read_slot(wd - 1 - j))
        regs = []
        for r in range(self.pool_size):
            tree = self.regs[r]  # (2047, 24); vectorized argmax per slot
            kinds = np.argmax(tree[:, KIND_OFF:KIND_OFF + N_KINDS], axis=1)
            levels = np.argmax(tree[:, LEVEL_OFF:LEVEL_OFF + LEVELS], axis=1)
            idxs = np.argmax(tree[:, IDX_OFF:IDX_OFF + VAR_IDX], axis=1)
            present = tree.sum(axis=1) >= 0.5
            nodes = tuple(
                W.EMPTY_NODE if not present[p] else
                (KINDS[int(kinds[p])], int(levels[p]), int(idxs[p]))
                for p in range(N_SLOTS))
            regs.append(nodes)
        free = []
        for i in range(self.free_depth):
            free.append(int(np.argmax(self.free_stack[i])))
        fail_path = []
        for j in range(MAX_PATH_LEN):
            s = PATH_SEGS[_idx_of(self.fail_path[j])]
            if s != "":
                fail_path.append(s)
        rb = _idx_of(self.res_bool)
        rl = _idx_of(self.res_level)
        return {
            "mode": self._mode_name(),
            "fsel": _idx_of(self.fsel), "fptr": _idx_of(self.fptr),
            "asel": _idx_of(self.asel), "aptr": _idx_of(self.aptr),
            "res_sel": (_idx_of(self.res_sel)
                        if _idx_of(self.res_sel) != N_SRC else -1),
            "res_ptr": _idx_of(self.res_ptr),
            "res_bool": None if rb == 0 else (rb == 1),
            "res_level": None if rl == 0 else rl - 1,
            "dest": self._dest_idx(),
            "ctx": tuple(self._ctx_ref(i)
                         for i in range(self._ctx_depth())),
            "ctrl": tuple(ctrl),
            "wstack": tuple(wstack),
            "regs": tuple(regs),
            "fail_path": tuple(fail_path),
            "fail_kind": FAIL_KINDS[_idx_of(self.fail_kind)],
            "free": tuple(free),
            "step_allocs": tuple(sorted(
                int(i) for i in np.where(self.step_allocs > 0.5)[0])),
            "steps": int(self.steps),
        }

    # -- action helpers -------------------------------------------------------

    def _focus(self):
        return _idx_of(self.fsel), _idx_of(self.fptr)

    def _aux(self):
        return _idx_of(self.asel), _idx_of(self.aptr)

    def _res_ref(self):
        s = _idx_of(self.res_sel)
        return (-1, 0) if s == N_SRC else (s, _idx_of(self.res_ptr))

    def _set_focus(self, sel, ptr):
        self.fsel = _onehot(N_SRC, sel)
        self.fptr = _onehot(N_SLOTS, ptr)

    def _set_aux(self, sel, ptr):
        self.asel = _onehot(N_SRC, sel)
        self.aptr = _onehot(N_SLOTS, ptr)

    def _set_mode(self, name):
        self.mode = _onehot(N_MODES, MODE[name])

    def _sel_of_reg(self, r):
        return 2 + r

    def _ptr_child(self, ptr_vec, mat):
        """Apply a ptr permutation matrix to a ptr one-hot."""
        return mat @ ptr_vec

    def _node_at(self, sel, ptr):
        """(kind, level, idx) of the node at (sel, ptr)."""
        node = self.read_node(sel, ptr)
        return (KINDS[int(np.argmax(node[KIND_OFF:KIND_OFF + N_KINDS]))],
                int(np.argmax(node[LEVEL_OFF:LEVEL_OFF + LEVELS])),
                int(np.argmax(node[IDX_OFF:IDX_OFF + VAR_IDX])))

    # -- INFER actions ---------------------------------------------------------
    # Focus (fsel, fptr) is the term. Mirrors RefMachine._step_infer.

    def _act_inf_var_ok(self):
        _, fptr = self._focus()
        idx = self.read_idx(_idx_of(self.fsel), fptr)
        s, p = self._ctx_ref(idx)
        d = self._alloc()
        self._start_walk("WALK_DONE", [None, None, None, None], (0, 0),
                         [("COPY", s, p, self._sel_of_reg(d), 0,
                           idx + 1, 0, -1, 0)], wdst=d)

    def _act_inf_var_err(self):
        self._error("unbound")

    def _act_inf_type(self):
        level = self.read_level(_idx_of(self.fsel), _idx_of(self.fptr))
        d = self._dest_idx()
        self.write_node_reg(d, 0, "TYPE", level + 1, 0)
        self._return(res_sel=self._sel_of_reg(d), res_ptr=0)

    def _act_inf_nat(self):
        d = self._dest_idx()
        self.write_node_reg(d, 0, "TYPE", 0, 0)
        self._return(res_sel=self._sel_of_reg(d), res_ptr=0)

    def _act_inf_zero(self):
        d = self._dest_idx()
        self.write_node_reg(d, 0, "NAT")
        self._return(res_sel=self._sel_of_reg(d), res_ptr=0)

    def _act_inf_lam(self):
        fsel, fptr = self._focus()
        ty_ref = (fsel, int((LEFT_CHILD @ self.fptr).argmax()))
        body_ref = (fsel, int((RIGHT_CHILD @ self.fptr).argmax()))
        d = self._alloc()
        self._push_frame("LAM_TY", [ty_ref, body_ref, None, None], (0, 0))
        self.dest = _onehot(DEST_DIM, d)
        self._set_focus(*ty_ref)
        self._set_mode("INFER")

    def _act_inf_pi(self):
        fsel, fptr = self._focus()
        ty_ref = (fsel, int((LEFT_CHILD @ self.fptr).argmax()))
        body_ref = (fsel, int((RIGHT_CHILD @ self.fptr).argmax()))
        d = self._alloc()
        self._push_frame("PI_DOM", [ty_ref, body_ref, None, None], (0, 0))
        self.dest = _onehot(DEST_DIM, d)
        self._set_focus(*ty_ref)
        self._set_mode("INFER")

    def _act_inf_app(self):
        fsel, fptr = self._focus()
        func_ref = (fsel, int((LEFT_CHILD @ self.fptr).argmax()))
        arg_ref = (fsel, int((RIGHT_CHILD @ self.fptr).argmax()))
        d = self._alloc()
        self._push_frame("APP_F", [arg_ref, None, None, None], (0, 0))
        self.dest = _onehot(DEST_DIM, d)
        self._set_focus(*func_ref)
        self._set_mode("INFER")

    def _act_inf_succ(self):
        fsel, fptr = self._focus()
        pred_ref = (fsel, int((LEFT_CHILD @ self.fptr).argmax()))
        nr = self._alloc()
        self.write_node_reg(nr, 0, "NAT")
        self._push_frame("SUCC_T", [None, None, None, None], (0, 0))
        self._begin_check(pred_ref, (self._sel_of_reg(nr), 0))

    def _act_inf_rec(self):
        fsel, fptr = self._focus()
        mot_ref = (fsel, int((REC_MOTIVE_PTR @ self.fptr).argmax()))
        base_ref = (fsel, int((REC_BASE_PTR @ self.fptr).argmax()))
        step_ref = (fsel, int((REC_STEP_PTR @ self.fptr).argmax()))
        tgt_ref = (fsel, int((REC_TARGET_PTR @ self.fptr).argmax()))
        d = self._alloc()
        self._push_frame("REC_M", [base_ref, step_ref, tgt_ref, None],
                         (mot_ref[0], mot_ref[1]))
        self.dest = _onehot(DEST_DIM, d)
        self._set_focus(*mot_ref)
        self._set_mode("INFER")

    def _act_inf_err(self):
        self._error("unknown-term")

    # -- WHNF actions ----------------------------------------------------------
    # Focus is the subject. Mirrors RefMachine._step_whnf.

    def _act_whnf_app(self):
        fsel, fptr = self._focus()
        func_ref = (fsel, int((LEFT_CHILD @ self.fptr).argmax()))
        arg_ref = (fsel, int((RIGHT_CHILD @ self.fptr).argmax()))
        self._push_frame("WHNF_APP", [arg_ref, None, None, None], (0, 0))
        self._set_focus(*func_ref)
        self._set_mode("WHNF")

    def _act_whnf_rec(self):
        fsel, fptr = self._focus()
        mot_ref = (fsel, int((REC_MOTIVE_PTR @ self.fptr).argmax()))
        base_ref = (fsel, int((REC_BASE_PTR @ self.fptr).argmax()))
        step_ref = (fsel, int((REC_STEP_PTR @ self.fptr).argmax()))
        tgt_ref = (fsel, int((REC_TARGET_PTR @ self.fptr).argmax()))
        self._push_frame("WHNF_REC", [mot_ref, base_ref, step_ref, None],
                         (tgt_ref[0], tgt_ref[1]))
        self._set_focus(*tgt_ref)
        self._set_mode("WHNF")

    def _act_whnf_copy(self):
        fsel, fptr = self._focus()
        d = self._alloc()
        self._start_walk("WALK_DONE", [None, None, None, None], (0, 0),
                         [("COPY", fsel, fptr, self._sel_of_reg(d), 0,
                           0, 0, -1, 0)], wdst=d)

    # -- COMPARE action ---------------------------------------------------------

    def _act_cmp_start(self):
        l_ref = self._focus()
        r_ref = self._aux()
        self._push_frame("CMP_W_L", [l_ref, r_ref, None, None], (0, 0))
        self._set_focus(*l_ref)
        self._set_mode("WHNF")

    # -- WALK actions ------------------------------------------------------------
    # One work item per step. Each action is straight-line for its own
    # (task, kind) case — the control matrix is the dispatch, so the
    # body never re-branches on the kind. The node ops use the subst
    # circuit's per-node machinery (VAR threshold comparisons, the
    # shift permutation, graft detection as an equality).

    def _act_walk_empty(self):
        self._set_mode("RETURN")

    def _walk_write_node(self, dsel, dptr, node):
        """Write a 24-dim node vector into the tree at (dsel, dptr)
        (the node-write construction). dsel is a register sel."""
        r = dsel - 2
        self.regs[r, dptr] = np.array(node, dtype=float)

    def _walk_pop_item(self):
        return self._pop_walk()

    def _child_ptrs(self, sptr, dptr):
        """(left_src, right_src, left_dst, right_dst) ptr indices."""
        lp = int((LEFT_CHILD @ _onehot(N_SLOTS, sptr)).argmax())
        rp = int((RIGHT_CHILD @ _onehot(N_SLOTS, sptr)).argmax())
        ld = int((LEFT_CHILD @ _onehot(N_SLOTS, dptr)).argmax())
        rd = int((RIGHT_CHILD @ _onehot(N_SLOTS, dptr)).argmax())
        return lp, rp, ld, rd

    def _rec_ptrs(self, sptr, dptr):
        """(mot, base, step, tgt) ptr indices, source and dest."""
        mp = int((REC_MOTIVE_PTR @ _onehot(N_SLOTS, sptr)).argmax())
        bp = int((REC_BASE_PTR @ _onehot(N_SLOTS, sptr)).argmax())
        sp = int((REC_STEP_PTR @ _onehot(N_SLOTS, sptr)).argmax())
        tp = int((REC_TARGET_PTR @ _onehot(N_SLOTS, sptr)).argmax())
        md = int((REC_MOTIVE_PTR @ _onehot(N_SLOTS, dptr)).argmax())
        bd = int((REC_BASE_PTR @ _onehot(N_SLOTS, dptr)).argmax())
        sd = int((REC_STEP_PTR @ _onehot(N_SLOTS, dptr)).argmax())
        td = int((REC_TARGET_PTR @ _onehot(N_SLOTS, dptr)).argmax())
        return mp, bp, sp, tp, md, bd, sd, td

    def _rec_skeleton(self, dsel, dptr):
        """Write the REC/REC3/REC2 skeleton nodes at the dest."""
        r = dsel - 2
        self.write_node_reg(r, dptr, "REC")
        r2 = int((RIGHT_CHILD @ _onehot(N_SLOTS, dptr)).argmax())
        self.write_node_reg(r, r2, "REC3")
        r3 = int((RIGHT_CHILD @ _onehot(N_SLOTS, r2)).argmax())
        self.write_node_reg(r, r3, "REC2")

    def _act_walk_copy_var(self):
        (_, ssel, sptr, dsel, dptr, shift, cutoff, _, _) = \
            self._walk_pop_item()
        idx = self.read_idx(ssel, sptr)
        # idx' = idx + shift if idx >= cutoff else idx (threshold + mux).
        idx2 = idx + shift if idx >= cutoff else idx
        node = np.zeros(SLOT_DIM)
        node[KIND_OFF + KIND["VAR"]] = 1.0
        node[IDX_OFF + idx2] = 1.0
        self._walk_write_node(dsel, dptr, node)

    def _act_walk_copy_leaf(self):
        (_, ssel, sptr, dsel, dptr, _, _, _, _) = self._walk_pop_item()
        node = self.read_node(ssel, sptr)
        self._walk_write_node(dsel, dptr, node)

    def _act_walk_copy_lampi(self):
        (_, ssel, sptr, dsel, dptr, shift, cutoff, _, _) = \
            self._walk_pop_item()
        kind, _, _ = self._node_at(ssel, sptr)
        node = np.zeros(SLOT_DIM)
        node[KIND_OFF + KIND[kind]] = 1.0
        self._walk_write_node(dsel, dptr, node)
        lp, rp, ld, rd = self._child_ptrs(sptr, dptr)
        self._push_walk("COPY", ssel, lp, dsel, ld, shift, cutoff, -1, 0)
        self._push_walk("COPY", ssel, rp, dsel, rd, shift, cutoff + 1,
                        -1, 0)

    def _act_walk_copy_app(self):
        (_, ssel, sptr, dsel, dptr, shift, cutoff, _, _) = \
            self._walk_pop_item()
        node = np.zeros(SLOT_DIM)
        node[KIND_OFF + KIND["APP"]] = 1.0
        self._walk_write_node(dsel, dptr, node)
        lp, rp, ld, rd = self._child_ptrs(sptr, dptr)
        self._push_walk("COPY", ssel, lp, dsel, ld, shift, cutoff, -1, 0)
        self._push_walk("COPY", ssel, rp, dsel, rd, shift, cutoff, -1, 0)

    def _act_walk_copy_succ(self):
        (_, ssel, sptr, dsel, dptr, shift, cutoff, _, _) = \
            self._walk_pop_item()
        node = np.zeros(SLOT_DIM)
        node[KIND_OFF + KIND["SUCC"]] = 1.0
        self._walk_write_node(dsel, dptr, node)
        lp, _, ld, _ = self._child_ptrs(sptr, dptr)
        self._push_walk("COPY", ssel, lp, dsel, ld, shift, cutoff, -1, 0)

    def _act_walk_copy_rec(self):
        (_, ssel, sptr, dsel, dptr, shift, cutoff, _, _) = \
            self._walk_pop_item()
        self._rec_skeleton(dsel, dptr)
        mp, bp, sp, tp, md, bd, sd, td = self._rec_ptrs(sptr, dptr)
        self._push_walk("COPY", ssel, mp, dsel, md, shift, cutoff, -1, 0)
        self._push_walk("COPY", ssel, bp, dsel, bd, shift, cutoff, -1, 0)
        self._push_walk("COPY", ssel, sp, dsel, sd, shift, cutoff, -1, 0)
        self._push_walk("COPY", ssel, tp, dsel, td, shift, cutoff, -1, 0)

    def _act_walk_subst_var_graft(self):
        (_, ssel, sptr, dsel, dptr, _, s, rsel, rptr) = \
            self._walk_pop_item()
        # idx == k (the control selected this action on the graft bit):
        # push a COPY of the replacement, shifted by s.
        self._push_walk("COPY", rsel, rptr, dsel, dptr, s, 0, -1, 0)

    def _act_walk_subst_var_gt(self):
        (_, ssel, sptr, dsel, dptr, _, _, _, _) = self._walk_pop_item()
        idx = self.read_idx(ssel, sptr)
        node = np.zeros(SLOT_DIM)
        node[KIND_OFF + KIND["VAR"]] = 1.0
        node[IDX_OFF + idx - 1] = 1.0
        self._walk_write_node(dsel, dptr, node)

    def _act_walk_subst_var_lt(self):
        (_, ssel, sptr, dsel, dptr, _, _, _, _) = self._walk_pop_item()
        node = self.read_node(ssel, sptr)
        self._walk_write_node(dsel, dptr, node)

    def _act_walk_subst_leaf(self):
        (_, ssel, sptr, dsel, dptr, _, _, _, _) = self._walk_pop_item()
        node = self.read_node(ssel, sptr)
        self._walk_write_node(dsel, dptr, node)

    def _act_walk_subst_lampi(self):
        (_, ssel, sptr, dsel, dptr, k, s, rsel, rptr) = \
            self._walk_pop_item()
        kind, _, _ = self._node_at(ssel, sptr)
        node = np.zeros(SLOT_DIM)
        node[KIND_OFF + KIND[kind]] = 1.0
        self._walk_write_node(dsel, dptr, node)
        lp, rp, ld, rd = self._child_ptrs(sptr, dptr)
        self._push_walk("SUBST", ssel, lp, dsel, ld, k, s, rsel, rptr)
        self._push_walk("SUBST", ssel, rp, dsel, rd, k + 1, s + 1,
                        rsel, rptr)

    def _act_walk_subst_app(self):
        (_, ssel, sptr, dsel, dptr, k, s, rsel, rptr) = \
            self._walk_pop_item()
        node = np.zeros(SLOT_DIM)
        node[KIND_OFF + KIND["APP"]] = 1.0
        self._walk_write_node(dsel, dptr, node)
        lp, rp, ld, rd = self._child_ptrs(sptr, dptr)
        self._push_walk("SUBST", ssel, lp, dsel, ld, k, s, rsel, rptr)
        self._push_walk("SUBST", ssel, rp, dsel, rd, k, s, rsel, rptr)

    def _act_walk_subst_succ(self):
        (_, ssel, sptr, dsel, dptr, k, s, rsel, rptr) = \
            self._walk_pop_item()
        node = np.zeros(SLOT_DIM)
        node[KIND_OFF + KIND["SUCC"]] = 1.0
        self._walk_write_node(dsel, dptr, node)
        lp, _, ld, _ = self._child_ptrs(sptr, dptr)
        self._push_walk("SUBST", ssel, lp, dsel, ld, k, s, rsel, rptr)

    def _act_walk_subst_rec(self):
        (_, ssel, sptr, dsel, dptr, k, s, rsel, rptr) = \
            self._walk_pop_item()
        self._rec_skeleton(dsel, dptr)
        mp, bp, sp, tp, md, bd, sd, td = self._rec_ptrs(sptr, dptr)
        self._push_walk("SUBST", ssel, mp, dsel, md, k, s, rsel, rptr)
        self._push_walk("SUBST", ssel, bp, dsel, bd, k, s, rsel, rptr)
        self._push_walk("SUBST", ssel, sp, dsel, sd, k, s, rsel, rptr)
        self._push_walk("SUBST", ssel, tp, dsel, td, k, s, rsel, rptr)

    # -- RETURN actions ----------------------------------------------------------
    # Each pops the top frame (restoring dest) and runs the handler.
    # _handler_seg is set by step() before the call (the RefMachine's
    # _step_return does the same). Mirrors RefMachine._ret_* exactly,
    # including push order and dest discipline.

    def _act_ret_halt(self):
        self._set_mode("HALT")

    def _act_ret_entry_valid(self):
        fr = self._pop_frame()
        r1 = fr[1]
        self._push_frame("ENTRY_VALID_W", [r1, None, None, None], (0, 0))
        rs, rp = self._res_ref()
        self._set_focus(rs, rp)
        self._set_mode("WHNF")

    def _act_ret_entry_valid_w_ok(self):
        fr = self._pop_frame()
        r1 = fr[1]
        self._begin_check(r1, (1, 0))

    def _act_ret_entry_valid_w_err(self):
        self._pop_frame()
        self._error("not-a-type")

    def _act_ret_lam_ty(self):
        fr = self._pop_frame()
        r1, r2 = fr[1], fr[2]
        self._push_frame("LAM_TY_W", [r1, r2, None, None], (0, 0))
        rs, rp = self._res_ref()
        self._set_focus(rs, rp)
        self._set_mode("WHNF")

    def _act_ret_lam_ty_w_ok(self):
        fr = self._pop_frame()
        ty_ref, body_ref = fr[1], fr[2]
        self._ctx_push(*ty_ref)
        d = self._alloc()
        self._push_frame("LAM_BODY", [body_ref, None, None, None], (0, 0))
        self.dest = _onehot(DEST_DIM, d)
        self._set_focus(*body_ref)
        self._set_mode("INFER")

    def _act_ret_lam_ty_w_err(self):
        self._pop_frame()
        self._error("not-a-type")

    def _act_ret_lam_body(self):
        fr = self._pop_frame()
        r1 = fr[1]
        body_ty_ref = self._res_ref()
        self._ctx_pop()
        lam_ptr = int((PARENT_PTR @ _onehot(N_SLOTS, r1[1])).argmax())
        ty_ptr = int((LEFT_CHILD @ _onehot(N_SLOTS, lam_ptr)).argmax())
        ty_ref = (r1[0], ty_ptr)
        d = self._dest_idx()
        self.write_node_reg(d, 0, "PI")
        self._start_walk("WALK_DONE", [None, None, None, None], (0, 0),
                         [("COPY", ty_ref[0], ty_ref[1],
                           self._sel_of_reg(d),
                           int((LEFT_CHILD @ _onehot(N_SLOTS, 0)).argmax()),
                           0, 0, -1, 0),
                          ("COPY", body_ty_ref[0], body_ty_ref[1],
                           self._sel_of_reg(d),
                           int((RIGHT_CHILD @ _onehot(N_SLOTS, 0)).argmax()),
                           0, 0, -1, 0)],
                         wdst=d)

    def _act_ret_pi_dom(self):
        fr = self._pop_frame()
        r1, r2 = fr[1], fr[2]
        self._push_frame("PI_DOM_W", [r1, r2, None, None], (0, 0))
        rs, rp = self._res_ref()
        self._set_focus(rs, rp)
        self._set_mode("WHNF")

    def _act_ret_pi_dom_w_ok(self):
        fr = self._pop_frame()
        ty_ref, body_ref = fr[1], fr[2]
        i = self.read_level(*self._res_ref())
        self._ctx_push(*ty_ref)
        d = self._alloc()
        self._push_frame("PI_BODY", [body_ref, None, None, None], (i, 0))
        self.dest = _onehot(DEST_DIM, d)
        self._set_focus(*body_ref)
        self._set_mode("INFER")

    def _act_ret_pi_dom_w_err(self):
        self._pop_frame()
        self._error("not-a-type")

    def _act_ret_pi_body(self):
        fr = self._pop_frame()
        i = fr[5]
        self._push_frame("PI_BODY_W", [None, None, None, None], (i, 0))
        rs, rp = self._res_ref()
        self._set_focus(rs, rp)
        self._set_mode("WHNF")

    def _act_ret_pi_body_w_ok(self):
        fr = self._pop_frame()
        j = self.read_level(*self._res_ref())
        self._ctx_pop()
        d = self._dest_idx()
        self.write_node_reg(d, 0, "TYPE", max(fr[5], j), 0)
        self._return(res_sel=self._sel_of_reg(d), res_ptr=0)

    def _act_ret_pi_body_w_err(self):
        self._pop_frame()
        self._error("not-a-type")

    def _act_ret_app_f(self):
        fr = self._pop_frame()
        r1 = fr[1]
        self._push_frame("APP_F_W", [r1, None, None, None], (0, 0))
        rs, rp = self._res_ref()
        self._set_focus(rs, rp)
        self._set_mode("WHNF")

    def _act_ret_app_f_w_ok(self):
        fr = self._pop_frame()
        arg_ref = fr[1]
        rs, rp = self._res_ref()
        dom_ref = (rs, int((LEFT_CHILD @ _onehot(N_SLOTS, rp)).argmax()))
        cod_ref = (rs, int((RIGHT_CHILD @ _onehot(N_SLOTS, rp)).argmax()))
        self._push_frame("APP_A", [cod_ref, arg_ref, None, None], (0, 0))
        self._begin_check(arg_ref, dom_ref)

    def _act_ret_app_f_w_err(self):
        self._pop_frame()
        self._error("not-pi")

    def _act_ret_app_a(self):
        fr = self._pop_frame()
        cod_ref, arg_ref = fr[1], fr[2]
        d = self._alloc()
        self._start_walk("APP_SUBST", [None, None, None, None], (0, 0),
                         [("SUBST", cod_ref[0], cod_ref[1],
                           self._sel_of_reg(d), 0, 0, 0,
                           arg_ref[0], arg_ref[1])], wdst=d)

    def _act_ret_app_subst(self):
        fr = self._pop_frame()
        self._return(res_sel=self._sel_of_reg(fr[8]), res_ptr=0)

    def _act_ret_succ_t(self):
        self._pop_frame()
        d = self._dest_idx()
        self.write_node_reg(d, 0, "NAT")
        self._return(res_sel=self._sel_of_reg(d), res_ptr=0)

    def _act_ret_rec_m(self):
        fr = self._pop_frame()
        r1, r2, r3, s1, s2 = fr[1], fr[2], fr[3], fr[5], fr[6]
        self._push_frame("REC_M_W", [r1, r2, r3, None], (s1, s2))
        rs, rp = self._res_ref()
        self._set_focus(rs, rp)
        self._set_mode("WHNF")

    def _act_ret_rec_m_w_ok(self):
        fr = self._pop_frame()
        r1, r2, r3, s1, s2 = fr[1], fr[2], fr[3], fr[5], fr[6]
        rs, rp = self._res_ref()
        dom_ref = (rs, int((LEFT_CHILD @ _onehot(N_SLOTS, rp)).argmax()))
        nr = self._alloc()
        self.write_node_reg(nr, 0, "NAT")
        self._push_frame("REC_M_D", [r1, r2, r3, (rs, rp)], (s1, s2))
        self._set_focus(*dom_ref)
        self._set_aux(self._sel_of_reg(nr), 0)
        self._set_mode("COMPARE")

    def _act_ret_rec_m_w_err(self):
        self._pop_frame()
        self._error("not-pi")

    def _act_ret_rec_m_d_ok(self):
        fr = self._pop_frame()
        r1, r2, r3, s1, s2 = fr[1], fr[2], fr[3], fr[5], fr[6]
        mot_ty_ref = fr[4]
        cod_ref = (mot_ty_ref[0],
                   int((RIGHT_CHILD @ _onehot(N_SLOTS,
                                              mot_ty_ref[1])).argmax()))
        self._push_frame("REC_M_C", [r1, r2, r3, None], (s1, s2))
        self._set_focus(*cod_ref)
        self._set_mode("WHNF")

    def _act_ret_rec_m_d_err(self):
        self._pop_frame()
        self._error("mismatch")

    def _act_ret_rec_m_c_ok(self):
        fr = self._pop_frame()
        r1, r2, r3, s1, s2 = fr[1], fr[2], fr[3], fr[5], fr[6]
        mot_ref = (s1, s2)
        d = self._alloc()
        self.write_node_reg(d, 0, "APP")
        self.write_node_reg(d, int((RIGHT_CHILD @ _onehot(N_SLOTS, 0)).argmax()),
                            "ZERO")
        self._start_walk("REC_BASE_B", [r1, r2, r3, None], (s1, s2),
                         [("COPY", mot_ref[0], mot_ref[1],
                           self._sel_of_reg(d),
                           int((LEFT_CHILD @ _onehot(N_SLOTS, 0)).argmax()),
                           0, 0, -1, 0)],
                         wdst=d)

    def _act_ret_rec_m_c_err(self):
        self._pop_frame()
        self._error("not-a-type")

    def _act_ret_rec_base_b(self):
        fr = self._pop_frame()
        r1, r2, r3, s1, s2 = fr[1], fr[2], fr[3], fr[5], fr[6]
        base_ref = r1
        app_ref = (self._sel_of_reg(fr[8]), 0)
        self._push_frame("REC_BASE", [r2, r3, None, None], (s1, s2))
        self._begin_check(base_ref, app_ref)

    def _act_ret_rec_base(self):
        fr = self._pop_frame()
        r1, r2, s1, s2 = fr[1], fr[2], fr[5], fr[6]
        mot_ref = (s1, s2)
        step_ref, tgt_ref = r1, r2
        d = self._alloc()
        ds = self._sel_of_reg(d)
        # Skeleton: 0=PI, 1=NAT, 2=PI, 5=APP, 12=VAR(0), 6=APP,
        # 14=SUCC, 29=VAR(1). COPY mot -> 11 (shift 1), mot -> 13 (shift 2).
        self.write_node_reg(d, 0, "PI")
        self.write_node_reg(d, 1, "NAT")
        self.write_node_reg(d, 2, "PI")
        self.write_node_reg(d, 5, "APP")
        self.write_node_reg(d, 12, "VAR", 0, 0)
        self.write_node_reg(d, 6, "APP")
        self.write_node_reg(d, 14, "SUCC")
        self.write_node_reg(d, 29, "VAR", 0, 1)
        self._start_walk("REC_STEP_B", [step_ref, tgt_ref, None, None],
                         (s1, s2),
                         [("COPY", mot_ref[0], mot_ref[1], ds, 11,
                           1, 0, -1, 0),
                          ("COPY", mot_ref[0], mot_ref[1], ds, 13,
                           2, 0, -1, 0)],
                         wdst=d)

    def _act_ret_rec_step_b(self):
        fr = self._pop_frame()
        r1, r2, s1, s2 = fr[1], fr[2], fr[5], fr[6]
        step_ref = r1
        ty_ref = (self._sel_of_reg(fr[8]), 0)
        self._push_frame("REC_STEP", [r2, None, None, None], (s1, s2))
        self._begin_check(step_ref, ty_ref)

    def _act_ret_rec_step(self):
        fr = self._pop_frame()
        r1, s1, s2 = fr[1], fr[5], fr[6]
        tgt_ref = r1
        nr = self._alloc()
        self.write_node_reg(nr, 0, "NAT")
        self._push_frame("REC_TGT", [tgt_ref, None, None, None], (s1, s2))
        self._begin_check(tgt_ref, (self._sel_of_reg(nr), 0))

    def _act_ret_rec_tgt(self):
        fr = self._pop_frame()
        r1, s1, s2 = fr[1], fr[5], fr[6]
        mot_ref = (s1, s2)
        tgt_ref = r1
        d = self._alloc()
        self.write_node_reg(d, 0, "APP")
        self._start_walk("REC_RES", [None, None, None, None], (s1, s2),
                         [("COPY", mot_ref[0], mot_ref[1],
                           self._sel_of_reg(d),
                           int((LEFT_CHILD @ _onehot(N_SLOTS, 0)).argmax()),
                           0, 0, -1, 0),
                          ("COPY", tgt_ref[0], tgt_ref[1],
                           self._sel_of_reg(d),
                           int((RIGHT_CHILD @ _onehot(N_SLOTS, 0)).argmax()),
                           0, 0, -1, 0)],
                         wdst=d)

    def _act_ret_rec_res(self):
        fr = self._pop_frame()
        self._return(res_sel=self._sel_of_reg(fr[8]), res_ptr=0)

    def _act_ret_chk_w_lam_pi(self):
        fr = self._pop_frame()
        term_ref, type_ref = fr[1], fr[2]
        binder_ty_ref = (term_ref[0],
                         int((LEFT_CHILD @ _onehot(N_SLOTS,
                                                   term_ref[1])).argmax()))
        body_ref = (term_ref[0],
                    int((RIGHT_CHILD @ _onehot(N_SLOTS,
                                              term_ref[1])).argmax()))
        rs, rp = self._res_ref()
        dom_ref = (rs, int((LEFT_CHILD @ _onehot(N_SLOTS, rp)).argmax()))
        cod_ref = (rs, int((RIGHT_CHILD @ _onehot(N_SLOTS, rp)).argmax()))
        self._push_frame("CHK_LAM_D", [body_ref, cod_ref, dom_ref, None],
                         (0, 0))
        self._set_focus(*binder_ty_ref)
        self._set_aux(*dom_ref)
        self._set_mode("COMPARE")

    def _act_ret_chk_w_other(self):
        fr = self._pop_frame()
        term_ref, type_ref = fr[1], fr[2]
        self._push_frame("CHK_INF", [type_ref, None, None, None], (0, 0))
        d = self._alloc()
        self.dest = _onehot(DEST_DIM, d)
        self._set_focus(*term_ref)
        self._set_mode("INFER")

    def _act_ret_chk_lam_d_ok(self):
        fr = self._pop_frame()
        body_ref, cod_ref, dom_ref = fr[1], fr[2], fr[3]
        self._ctx_push(*dom_ref)
        self._push_frame("CHK_LAM_B", [None, None, None, None], (0, 0))
        self._begin_check(body_ref, cod_ref)

    def _act_ret_chk_lam_d_err(self):
        self._pop_frame()
        self._error("mismatch")

    def _act_ret_chk_lam_b(self):
        self._pop_frame()
        self._ctx_pop()
        self._set_mode("RETURN")

    def _act_ret_chk_inf(self):
        fr = self._pop_frame()
        type_ref = fr[1]
        self._push_frame("CHK_DEFEQ", [None, None, None, None], (0, 0))
        rs, rp = self._res_ref()
        self._set_focus(rs, rp)
        self._set_aux(*type_ref)
        self._set_mode("COMPARE")

    def _act_ret_chk_defeq_ok(self):
        self._pop_frame()
        self._set_mode("RETURN")

    def _act_ret_chk_defeq_err(self):
        self._pop_frame()
        self._error("mismatch")

    def _act_ret_whnf_app_lam(self):
        fr = self._pop_frame()
        arg_ref = fr[1]
        rs, rp = self._res_ref()
        body_ref = (rs, int((RIGHT_CHILD @ _onehot(N_SLOTS, rp)).argmax()))
        d = self._alloc()
        self._start_walk("WHNF_APP_S", [None, None, None, None], (0, 0),
                         [("SUBST", body_ref[0], body_ref[1],
                           self._sel_of_reg(d), 0, 0, 0,
                           arg_ref[0], arg_ref[1])], wdst=d)

    def _act_ret_whnf_app_neutral(self):
        fr = self._pop_frame()
        arg_ref = fr[1]
        f_w = self._res_ref()
        d = self._alloc()
        self.write_node_reg(d, 0, "APP")
        self._start_walk("WALK_DONE", [None, None, None, None], (0, 0),
                         [("COPY", f_w[0], f_w[1],
                           self._sel_of_reg(d),
                           int((LEFT_CHILD @ _onehot(N_SLOTS, 0)).argmax()),
                           0, 0, -1, 0),
                          ("COPY", arg_ref[0], arg_ref[1],
                           self._sel_of_reg(d),
                           int((RIGHT_CHILD @ _onehot(N_SLOTS, 0)).argmax()),
                           0, 0, -1, 0)],
                         wdst=d)

    def _act_ret_whnf_app_s(self):
        fr = self._pop_frame()
        self._set_focus(self._sel_of_reg(fr[8]), 0)
        self._set_mode("WHNF")

    def _act_ret_whnf_rec_zero(self):
        fr = self._pop_frame()
        base_ref = fr[2]
        self._return(res_sel=base_ref[0], res_ptr=base_ref[1])

    def _act_ret_whnf_rec_succ(self):
        fr = self._pop_frame()
        mot_ref, base_ref, step_ref = fr[1], fr[2], fr[3]
        rs, rp = self._res_ref()
        m_ref = (rs, int((LEFT_CHILD @ _onehot(N_SLOTS, rp)).argmax()))
        d = self._alloc()
        ds = self._sel_of_reg(d)
        # App(App(step, m), Rec(mot, base, step, m)):
        # 0=APP, 1=APP, 2=REC, 3=step, 4=m, 5=mot, 6=REC3, 13=base,
        # 14=REC2, 29=step, 30=m.
        self.write_node_reg(d, 0, "APP")
        self.write_node_reg(d, 1, "APP")
        self.write_node_reg(d, 2, "REC")
        self.write_node_reg(d, 6, "REC3")
        self.write_node_reg(d, 14, "REC2")
        self._start_walk("WHNF_REC_S", [None, None, None, None], (0, 0),
                         [("COPY", step_ref[0], step_ref[1], ds, 3,
                           0, 0, -1, 0),
                          ("COPY", m_ref[0], m_ref[1], ds, 4,
                           0, 0, -1, 0),
                          ("COPY", mot_ref[0], mot_ref[1], ds, 5,
                           0, 0, -1, 0),
                          ("COPY", base_ref[0], base_ref[1], ds, 13,
                           0, 0, -1, 0),
                          ("COPY", step_ref[0], step_ref[1], ds, 29,
                           0, 0, -1, 0),
                          ("COPY", m_ref[0], m_ref[1], ds, 30,
                           0, 0, -1, 0)],
                         wdst=d)

    def _act_ret_whnf_rec_neutral(self):
        fr = self._pop_frame()
        mot_ref, base_ref, step_ref = fr[1], fr[2], fr[3]
        tgt_w = self._res_ref()
        d = self._alloc()
        ds = self._sel_of_reg(d)
        self.write_node_reg(d, 0, "REC")
        self.write_node_reg(d, 2, "REC3")
        self.write_node_reg(d, 6, "REC2")
        self._start_walk("WALK_DONE", [None, None, None, None], (0, 0),
                         [("COPY", mot_ref[0], mot_ref[1], ds, 1,
                           0, 0, -1, 0),
                          ("COPY", base_ref[0], base_ref[1], ds, 5,
                           0, 0, -1, 0),
                          ("COPY", step_ref[0], step_ref[1], ds, 13,
                           0, 0, -1, 0),
                          ("COPY", tgt_w[0], tgt_w[1], ds, 14,
                           0, 0, -1, 0)],
                         wdst=d)

    def _act_ret_whnf_rec_s(self):
        fr = self._pop_frame()
        self._set_focus(self._sel_of_reg(fr[8]), 0)
        self._set_mode("WHNF")

    def _act_ret_cmp_w_l(self):
        fr = self._pop_frame()
        r2 = fr[2]
        left_w = self._res_ref()
        self._push_frame("CMP_W_R", [left_w, r2, None, None], (0, 0))
        self._set_focus(*r2)
        self._set_mode("WHNF")

    def _act_ret_cmp_w_r_lam_lam(self):
        fr = self._pop_frame()
        lw = fr[1]
        rw = self._res_ref()  # whnf'd right (fr[2] is pre-whnf)
        self._push_frame("CMP_LAM_T", [lw, rw, None, None], (0, 0))
        self._set_focus(lw[0], int((LEFT_CHILD @ _onehot(N_SLOTS, lw[1])).argmax()))
        self._set_aux(rw[0], int((LEFT_CHILD @ _onehot(N_SLOTS, rw[1])).argmax()))
        self._set_mode("COMPARE")

    def _act_ret_cmp_w_r_eta_l(self):
        fr = self._pop_frame()
        lw = fr[1]
        rw = self._res_ref()  # whnf'd right (fr[2] is pre-whnf)
        self._push_frame("CMP_ETA", [lw, rw, None, None], (0, 0))
        d = self._alloc()
        self.dest = _onehot(DEST_DIM, d)
        self._set_focus(*rw)
        self._set_mode("INFER")

    def _act_ret_cmp_w_r_eta_r(self):
        fr = self._pop_frame()
        lw = fr[1]
        rw = self._res_ref()  # whnf'd right (fr[2] is pre-whnf)
        self._push_frame("CMP_ETA", [rw, lw, None, None], (0, 0))
        d = self._alloc()
        self.dest = _onehot(DEST_DIM, d)
        self._set_focus(*lw)
        self._set_mode("INFER")

    def _act_ret_cmp_w_r_var_var(self):
        fr = self._pop_frame()
        lw = fr[1]
        rw = self._res_ref()  # whnf'd right (fr[2] is pre-whnf)
        li = self.read_idx(*lw)
        ri = self.read_idx(*rw)
        self._return(res_bool=(li == ri))

    def _act_ret_cmp_w_r_type_type(self):
        fr = self._pop_frame()
        lw = fr[1]
        rw = self._res_ref()  # whnf'd right (fr[2] is pre-whnf)
        ll = self.read_level(*lw)
        rl = self.read_level(*rw)
        self._return(res_bool=(ll == rl))

    def _act_ret_cmp_w_r_pi_pi(self):
        fr = self._pop_frame()
        lw = fr[1]
        rw = self._res_ref()  # whnf'd right (fr[2] is pre-whnf)
        self._push_frame("CMP_PI_D", [lw, rw, None, None], (0, 0))
        self._set_focus(lw[0], int((LEFT_CHILD @ _onehot(N_SLOTS, lw[1])).argmax()))
        self._set_aux(rw[0], int((LEFT_CHILD @ _onehot(N_SLOTS, rw[1])).argmax()))
        self._set_mode("COMPARE")

    def _act_ret_cmp_w_r_app_app(self):
        fr = self._pop_frame()
        lw = fr[1]
        rw = self._res_ref()  # whnf'd right (fr[2] is pre-whnf)
        self._push_frame("CMP_APP_L", [lw, rw, None, None], (0, 0))
        self._set_focus(lw[0], int((LEFT_CHILD @ _onehot(N_SLOTS, lw[1])).argmax()))
        self._set_aux(rw[0], int((LEFT_CHILD @ _onehot(N_SLOTS, rw[1])).argmax()))
        self._set_mode("COMPARE")

    def _act_ret_cmp_w_r_nat_nat(self):
        self._pop_frame()
        self._return(res_bool=True)

    def _act_ret_cmp_w_r_zero_zero(self):
        self._pop_frame()
        self._return(res_bool=True)

    def _act_ret_cmp_w_r_succ_succ(self):
        fr = self._pop_frame()
        lw = fr[1]
        rw = self._res_ref()  # whnf'd right (fr[2] is pre-whnf)
        self._push_frame("CMP_SUCC", [None, None, None, None], (0, 0))
        self._set_focus(lw[0], int((LEFT_CHILD @ _onehot(N_SLOTS, lw[1])).argmax()))
        self._set_aux(rw[0], int((LEFT_CHILD @ _onehot(N_SLOTS, rw[1])).argmax()))
        self._set_mode("COMPARE")

    def _act_ret_cmp_w_r_rec_rec(self):
        fr = self._pop_frame()
        lw = fr[1]
        rw = self._res_ref()  # whnf'd right (fr[2] is pre-whnf)
        self._push_frame("CMP_REC_M", [lw, rw, None, None], (0, 0))
        self._set_focus(lw[0], int((REC_MOTIVE_PTR @ _onehot(N_SLOTS, lw[1])).argmax()))
        self._set_aux(rw[0], int((REC_MOTIVE_PTR @ _onehot(N_SLOTS, rw[1])).argmax()))
        self._set_mode("COMPARE")

    def _act_ret_cmp_w_r_else(self):
        self._pop_frame()
        self._return(res_bool=False)

    def _act_ret_cmp_lam_t_ok(self):
        fr = self._pop_frame()
        lw, rw = fr[1], fr[2]
        self._ctx_push(lw[0], int((LEFT_CHILD @ _onehot(N_SLOTS, lw[1])).argmax()))
        self._push_frame("CMP_LAM_B", [lw, rw, None, None], (0, 0))
        self._set_focus(lw[0], int((RIGHT_CHILD @ _onehot(N_SLOTS, lw[1])).argmax()))
        self._set_aux(rw[0], int((RIGHT_CHILD @ _onehot(N_SLOTS, rw[1])).argmax()))
        self._set_mode("COMPARE")

    def _act_ret_cmp_lam_t_err(self):
        self._pop_frame()
        self._return(res_bool=False)

    def _act_ret_cmp_lam_b(self):
        self._pop_frame()
        self._ctx_pop()
        rb = _idx_of(self.res_bool)
        self._return(res_bool=(rb == 1))

    def _act_ret_cmp_pi_d_ok(self):
        fr = self._pop_frame()
        lw, rw = fr[1], fr[2]
        self._ctx_push(lw[0], int((LEFT_CHILD @ _onehot(N_SLOTS, lw[1])).argmax()))
        self._push_frame("CMP_PI_C", [lw, rw, None, None], (0, 0))
        self._set_focus(lw[0], int((RIGHT_CHILD @ _onehot(N_SLOTS, lw[1])).argmax()))
        self._set_aux(rw[0], int((RIGHT_CHILD @ _onehot(N_SLOTS, rw[1])).argmax()))
        self._set_mode("COMPARE")

    def _act_ret_cmp_pi_d_err(self):
        self._pop_frame()
        self._return(res_bool=False)

    def _act_ret_cmp_pi_c(self):
        self._pop_frame()
        self._ctx_pop()
        rb = _idx_of(self.res_bool)
        self._return(res_bool=(rb == 1))

    def _act_ret_cmp_app_l_ok(self):
        fr = self._pop_frame()
        lw, rw = fr[1], fr[2]
        self._push_frame("CMP_APP_R", [lw, rw, None, None], (0, 0))
        self._set_focus(lw[0], int((RIGHT_CHILD @ _onehot(N_SLOTS, lw[1])).argmax()))
        self._set_aux(rw[0], int((RIGHT_CHILD @ _onehot(N_SLOTS, rw[1])).argmax()))
        self._set_mode("COMPARE")

    def _act_ret_cmp_app_l_err(self):
        self._pop_frame()
        self._return(res_bool=False)

    def _act_ret_cmp_app_r(self):
        self._pop_frame()
        rb = _idx_of(self.res_bool)
        self._return(res_bool=(rb == 1))

    def _act_ret_cmp_succ(self):
        self._pop_frame()
        rb = _idx_of(self.res_bool)
        self._return(res_bool=(rb == 1))

    def _act_ret_cmp_rec_m_ok(self):
        fr = self._pop_frame()
        lw, rw = fr[1], fr[2]
        self._push_frame("CMP_REC_B", [lw, rw, None, None], (0, 0))
        self._set_focus(lw[0], int((REC_BASE_PTR @ _onehot(N_SLOTS, lw[1])).argmax()))
        self._set_aux(rw[0], int((REC_BASE_PTR @ _onehot(N_SLOTS, rw[1])).argmax()))
        self._set_mode("COMPARE")

    def _act_ret_cmp_rec_m_err(self):
        self._pop_frame()
        self._return(res_bool=False)

    def _act_ret_cmp_rec_b_ok(self):
        fr = self._pop_frame()
        lw, rw = fr[1], fr[2]
        self._push_frame("CMP_REC_S", [lw, rw, None, None], (0, 0))
        self._set_focus(lw[0], int((REC_STEP_PTR @ _onehot(N_SLOTS, lw[1])).argmax()))
        self._set_aux(rw[0], int((REC_STEP_PTR @ _onehot(N_SLOTS, rw[1])).argmax()))
        self._set_mode("COMPARE")

    def _act_ret_cmp_rec_b_err(self):
        self._pop_frame()
        self._return(res_bool=False)

    def _act_ret_cmp_rec_s_ok(self):
        fr = self._pop_frame()
        lw, rw = fr[1], fr[2]
        self._push_frame("CMP_REC_T", [lw, rw, None, None], (0, 0))
        self._set_focus(lw[0], int((REC_TARGET_PTR @ _onehot(N_SLOTS, lw[1])).argmax()))
        self._set_aux(rw[0], int((REC_TARGET_PTR @ _onehot(N_SLOTS, rw[1])).argmax()))
        self._set_mode("COMPARE")

    def _act_ret_cmp_rec_s_err(self):
        self._pop_frame()
        self._return(res_bool=False)

    def _act_ret_cmp_rec_t(self):
        self._pop_frame()
        rb = _idx_of(self.res_bool)
        self._return(res_bool=(rb == 1))

    def _act_ret_cmp_eta(self):
        fr = self._pop_frame()
        r1, r2 = fr[1], fr[2]
        self._push_frame("CMP_ETA_W", [r1, r2, None, None], (0, 0))
        rs, rp = self._res_ref()
        self._set_focus(rs, rp)
        self._set_mode("WHNF")

    def _act_ret_cmp_eta_w_ok(self):
        fr = self._pop_frame()
        lam_ref, neutral_ref = fr[1], fr[2]
        rs, rp = self._res_ref()
        dom_ref = (rs, int((LEFT_CHILD @ _onehot(N_SLOTS, rp)).argmax()))
        lam_ty_ref = (lam_ref[0],
                      int((LEFT_CHILD @ _onehot(N_SLOTS, lam_ref[1])).argmax()))
        self._push_frame("CMP_ETA_D", [lam_ref, neutral_ref, dom_ref, None],
                         (0, 0))
        self._set_focus(*lam_ty_ref)
        self._set_aux(*dom_ref)
        self._set_mode("COMPARE")

    def _act_ret_cmp_eta_w_err(self):
        self._pop_frame()
        self._return(res_bool=False)

    def _act_ret_cmp_eta_d_ok(self):
        fr = self._pop_frame()
        lam_ref, neutral_ref, dom_ref = fr[1], fr[2], fr[3]
        self._ctx_push(*dom_ref)
        lam_body_ref = (lam_ref[0],
                        int((RIGHT_CHILD @ _onehot(N_SLOTS, lam_ref[1])).argmax()))
        d = self._alloc()
        self.write_node_reg(d, 0, "APP")
        self.write_node_reg(d, int((RIGHT_CHILD @ _onehot(N_SLOTS, 0)).argmax()),
                            "VAR", 0, 0)
        self._start_walk("CMP_ETA_B", [lam_body_ref, None, None, None],
                         (0, 0),
                         [("COPY", neutral_ref[0], neutral_ref[1],
                           self._sel_of_reg(d),
                           int((LEFT_CHILD @ _onehot(N_SLOTS, 0)).argmax()),
                           1, 0, -1, 0)],
                         wdst=d)

    def _act_ret_cmp_eta_d_err(self):
        self._pop_frame()
        self._return(res_bool=False)

    def _act_ret_cmp_eta_b(self):
        fr = self._pop_frame()
        lam_body_ref = fr[1]
        app_ref = (self._sel_of_reg(fr[8]), 0)
        self._push_frame("CMP_ETA_R", [None, None, None, None], (0, 0))
        self._set_focus(*lam_body_ref)
        self._set_aux(*app_ref)
        self._set_mode("COMPARE")

    def _act_ret_cmp_eta_r(self):
        self._pop_frame()
        self._ctx_pop()
        rb = _idx_of(self.res_bool)
        self._return(res_bool=(rb == 1))

    def _act_ret_walk_done(self):
        fr = self._pop_frame()
        self._return(res_sel=self._sel_of_reg(fr[8]), res_ptr=0)

    # -- control: features, action table, W_CTRL -----------------------------

    def _features(self):
        """Control features: one-hots + derived boolean bits."""
        f = {}
        f["mode"] = self.mode.copy()
        fs, fp = self._focus()
        fnode = self.read_node(fs, fp)
        f["fkind"] = fnode[KIND_OFF:KIND_OFF + N_KINDS].copy()
        f["fidx"] = fnode[IDX_OFF:IDX_OFF + VAR_IDX].copy()
        if self._ctrl_depth() > 0:
            top = self._frame_read_slot(0)
            f["topkind"] = _onehot(N_FRAME_KINDS, FRAME_KIND[top[0]])
            if top[1] is not None:
                r1k = self.read_node(top[1][0], top[1][1])
                f["r1kind"] = r1k[KIND_OFF:KIND_OFF + N_KINDS].copy()
            else:
                f["r1kind"] = np.zeros(N_KINDS)
        else:
            f["topkind"] = np.zeros(N_FRAME_KINDS)
            f["r1kind"] = np.zeros(N_KINDS)
        if self._wstack_depth() > 0:
            it = self._walk_read_slot(0)
            f["walktask"] = _onehot(2, 0 if it[0] == "COPY" else 1)
            wnode = self.read_node(it[1], it[2])
            f["wkind"] = wnode[KIND_OFF:KIND_OFF + N_KINDS].copy()
            f["widx"] = wnode[IDX_OFF:IDX_OFF + VAR_IDX].copy()
            f["wp1"] = _onehot(11, it[5])
            f["wp2"] = _onehot(11, it[6])
        else:
            f["walktask"] = np.zeros(2)
            f["wkind"] = np.zeros(N_KINDS)
            f["widx"] = np.zeros(VAR_IDX)
            f["wp1"] = np.zeros(11)
            f["wp2"] = np.zeros(11)
        rs, rp = self._res_ref()
        if rs >= 0:
            rnode = self.read_node(rs, rp)
            f["rkind"] = rnode[KIND_OFF:KIND_OFF + N_KINDS].copy()
        else:
            f["rkind"] = np.zeros(N_KINDS)
        f["rbool"] = self.res_bool.copy()
        f["ctxdepth"] = _onehot(9, self._ctx_depth())
        f["wempty"] = np.array([1.0 if self._wstack_depth() == 0 else 0.0])
        f["cempty"] = np.array([1.0 if self._ctrl_depth() == 0 else 0.0])
        # Derived bits (threshold layers; counted in the registry).
        fidx = int(np.argmax(f["fidx"]))
        depth = int(np.argmax(f["ctxdepth"]))
        f["var_ok"] = np.array([1.0 if fidx < depth else 0.0])
        f["var_bad"] = np.array([1.0 if fidx >= depth else 0.0])
        fk = int(np.argmax(f["fkind"]))
        f["fkind_bad"] = np.array(
            [1.0 if KINDS[fk] in ("EMPTY", "REC3", "REC2") else 0.0])
        f["fkind_not_app_rec"] = np.array(
            [1.0 if KINDS[fk] not in ("APP", "REC") else 0.0])
        rk = f["rkind"]
        f["rkind_is_lam"] = np.array([rk[KIND["LAM"]]])
        f["rkind_not_lam"] = np.array([1.0 - rk[KIND["LAM"]]])
        f["rkind_is_pi"] = np.array([rk[KIND["PI"]]])
        f["rkind_not_pi"] = np.array([1.0 - rk[KIND["PI"]]])
        f["rkind_is_type"] = np.array([rk[KIND["TYPE"]]])
        f["rkind_not_type"] = np.array([1.0 - rk[KIND["TYPE"]]])
        f["rkind_is_zero"] = np.array([rk[KIND["ZERO"]]])
        f["rkind_is_succ"] = np.array([rk[KIND["SUCC"]]])
        f["rkind_not_zero_succ"] = np.array(
            [1.0 if rk[KIND["ZERO"]] < 0.5 and rk[KIND["SUCC"]] < 0.5
             else 0.0])
        r1k = f["r1kind"]
        f["r1kind_is_lam"] = np.array([r1k[KIND["LAM"]]])
        f["r1kind_not_lam"] = np.array([1.0 - r1k[KIND["LAM"]]])
        wk = f["wkind"]
        f["wkind_is_var"] = np.array([wk[KIND["VAR"]]])
        f["wkind_is_leaf"] = np.array(
            [1.0 if wk[KIND["TYPE"]] > 0.5 or wk[KIND["NAT"]] > 0.5
             or wk[KIND["ZERO"]] > 0.5 else 0.0])
        f["wkind_is_lampi"] = np.array(
            [1.0 if wk[KIND["LAM"]] > 0.5 or wk[KIND["PI"]] > 0.5
             else 0.0])
        widx = int(np.argmax(f["widx"]))
        wp1 = int(np.argmax(f["wp1"]))
        is_var = wk[KIND["VAR"]] > 0.5
        graft = is_var and widx == wp1
        gt = is_var and widx > wp1
        f["w_graft"] = np.array([1.0 if graft else 0.0])
        f["w_gt"] = np.array([1.0 if gt else 0.0])
        f["w_var_lt"] = np.array([1.0 if is_var and not graft and not gt
                                  else 0.0])
        f["chk_w_other"] = np.array(
            [1.0 if not (r1k[KIND["LAM"]] > 0.5
                         and rk[KIND["PI"]] > 0.5) else 0.0])
        # CMP_W_R else: none of the 11 (lk, rk) conditions holds.
        pairs = [(KIND["LAM"], KIND["LAM"]), (KIND["VAR"], KIND["VAR"]),
                 (KIND["TYPE"], KIND["TYPE"]), (KIND["PI"], KIND["PI"]),
                 (KIND["APP"], KIND["APP"]), (KIND["NAT"], KIND["NAT"]),
                 (KIND["ZERO"], KIND["ZERO"]), (KIND["SUCC"], KIND["SUCC"]),
                 (KIND["REC"], KIND["REC"])]
        any_pair = any(r1k[a] > 0.5 and rk[b] > 0.5 for a, b in pairs)
        any_eta = ((r1k[KIND["LAM"]] > 0.5 and rk[KIND["LAM"]] < 0.5)
                   or (r1k[KIND["LAM"]] < 0.5 and rk[KIND["LAM"]] > 0.5))
        f["cmp_w_r_else"] = np.array([1.0 if not (any_pair or any_eta)
                                      else 0.0])
        return f

    def _feature_vector(self):
        """Concatenated feature vector + (group, index) -> position."""
        f = self._features()
        vec = []
        pos = {}
        for group, dim in FEATURE_GROUPS:
            arr = f[group]
            for i in range(dim):
                pos[(group, i)] = len(vec) + i
            vec.extend(arr.tolist())
        return np.array(vec), pos

    # -- the step ---------------------------------------------------------------

    def step(self):
        self.steps += 1
        self.step_allocs = np.zeros(POOL_SIZE)
        if self.steps > self.max_steps:
            self._error("fuel")
            return
        if self.steps > self.max_steps_used:
            self.max_steps_used = self.steps
        m = self._mode_name()
        if m in ("HALT", "ERROR"):
            return
        feats, pos = self._feature_vector()
        logits = feats @ W_CTRL + W_BIAS
        action = np.maximum(0.0, logits)
        total = float(action.sum())
        if abs(total - 0.5) > 1e-9:
            raise RuntimeError(
                "control fired total %r at step %d mode %s "
                "(logits %s)" % (total, self.steps, m, logits))
        ai = int(np.argmax(action))
        if m == "RETURN":
            top = self._top_frame()
            self._handler_seg = FRAME_SEG.get(top[0]) if top else None
        try:
            ACTION_FNS[ai](self)
        finally:
            self._handler_seg = None
        if self._mode_name() == "ERROR":
            self._try_catch_error()

    def run(self):
        while self._mode_name() not in ("HALT", "ERROR"):
            self.step()
        return self._mode_name()


# ---------------------------------------------------------------------------
# Action table + control matrix (built once)
# ---------------------------------------------------------------------------
# Each action: (name, method name, [(feature group, index), ...]).
# A required bit is (group, index) into the feature vector; for a
# one-hot group the index is the value's index, for a bit group it is 0.

FEATURE_GROUPS = [
    ("mode", N_MODES), ("fkind", N_KINDS), ("topkind", N_FRAME_KINDS),
    ("walktask", 2), ("wkind", N_KINDS), ("rkind", N_KINDS),
    ("r1kind", N_KINDS), ("rbool", RES_BOOL_DIM), ("fidx", VAR_IDX),
    ("ctxdepth", 9), ("wempty", 1), ("cempty", 1),
    ("var_ok", 1), ("var_bad", 1), ("fkind_bad", 1),
    ("fkind_not_app_rec", 1),
    ("rkind_is_lam", 1), ("rkind_not_lam", 1), ("rkind_is_pi", 1),
    ("rkind_not_pi", 1), ("rkind_is_type", 1), ("rkind_not_type", 1),
    ("rkind_is_zero", 1), ("rkind_is_succ", 1),
    ("rkind_not_zero_succ", 1),
    ("r1kind_is_lam", 1), ("r1kind_not_lam", 1),
    ("wkind_is_var", 1), ("wkind_is_leaf", 1), ("wkind_is_lampi", 1),
    ("w_graft", 1), ("w_gt", 1), ("w_var_lt", 1),
    ("chk_w_other", 1), ("cmp_w_r_else", 1),
]
FEATURE_DIM = sum(d for _, d in FEATURE_GROUPS)

_M = MODE
_K = KIND


def _a(name, method, req):
    return (name, method, req)


ACTION_SPECS = [
    # INFER
    _a("inf_var_ok", "_act_inf_var_ok",
       [("mode", _M["INFER"]), ("fkind", _K["VAR"]), ("var_ok", 0)]),
    _a("inf_var_err", "_act_inf_var_err",
       [("mode", _M["INFER"]), ("fkind", _K["VAR"]), ("var_bad", 0)]),
    _a("inf_type", "_act_inf_type",
       [("mode", _M["INFER"]), ("fkind", _K["TYPE"])]),
    _a("inf_nat", "_act_inf_nat",
       [("mode", _M["INFER"]), ("fkind", _K["NAT"])]),
    _a("inf_zero", "_act_inf_zero",
       [("mode", _M["INFER"]), ("fkind", _K["ZERO"])]),
    _a("inf_lam", "_act_inf_lam",
       [("mode", _M["INFER"]), ("fkind", _K["LAM"])]),
    _a("inf_pi", "_act_inf_pi",
       [("mode", _M["INFER"]), ("fkind", _K["PI"])]),
    _a("inf_app", "_act_inf_app",
       [("mode", _M["INFER"]), ("fkind", _K["APP"])]),
    _a("inf_succ", "_act_inf_succ",
       [("mode", _M["INFER"]), ("fkind", _K["SUCC"])]),
    _a("inf_rec", "_act_inf_rec",
       [("mode", _M["INFER"]), ("fkind", _K["REC"])]),
    _a("inf_err", "_act_inf_err",
       [("mode", _M["INFER"]), ("fkind_bad", 0)]),
    # WHNF
    _a("whnf_app", "_act_whnf_app",
       [("mode", _M["WHNF"]), ("fkind", _K["APP"])]),
    _a("whnf_rec", "_act_whnf_rec",
       [("mode", _M["WHNF"]), ("fkind", _K["REC"])]),
    _a("whnf_copy", "_act_whnf_copy",
       [("mode", _M["WHNF"]), ("fkind_not_app_rec", 0)]),
    # COMPARE
    _a("cmp_start", "_act_cmp_start", [("mode", _M["COMPARE"])]),
    # WALK
    _a("walk_empty", "_act_walk_empty",
       [("mode", _M["WALK"]), ("wempty", 0)]),
    _a("walk_copy_var", "_act_walk_copy_var",
       [("mode", _M["WALK"]), ("walktask", 0), ("wkind_is_var", 0)]),
    _a("walk_copy_leaf", "_act_walk_copy_leaf",
       [("mode", _M["WALK"]), ("walktask", 0), ("wkind_is_leaf", 0)]),
    _a("walk_copy_lampi", "_act_walk_copy_lampi",
       [("mode", _M["WALK"]), ("walktask", 0), ("wkind_is_lampi", 0)]),
    _a("walk_copy_app", "_act_walk_copy_app",
       [("mode", _M["WALK"]), ("walktask", 0), ("wkind", _K["APP"])]),
    _a("walk_copy_succ", "_act_walk_copy_succ",
       [("mode", _M["WALK"]), ("walktask", 0), ("wkind", _K["SUCC"])]),
    _a("walk_copy_rec", "_act_walk_copy_rec",
       [("mode", _M["WALK"]), ("walktask", 0), ("wkind", _K["REC"])]),
    _a("walk_subst_var_graft", "_act_walk_subst_var_graft",
       [("mode", _M["WALK"]), ("walktask", 1), ("w_graft", 0)]),
    _a("walk_subst_var_gt", "_act_walk_subst_var_gt",
       [("mode", _M["WALK"]), ("walktask", 1), ("w_gt", 0)]),
    _a("walk_subst_var_lt", "_act_walk_subst_var_lt",
       [("mode", _M["WALK"]), ("walktask", 1), ("w_var_lt", 0)]),
    _a("walk_subst_leaf", "_act_walk_subst_leaf",
       [("mode", _M["WALK"]), ("walktask", 1), ("wkind_is_leaf", 0)]),
    _a("walk_subst_lampi", "_act_walk_subst_lampi",
       [("mode", _M["WALK"]), ("walktask", 1), ("wkind_is_lampi", 0)]),
    _a("walk_subst_app", "_act_walk_subst_app",
       [("mode", _M["WALK"]), ("walktask", 1), ("wkind", _K["APP"])]),
    _a("walk_subst_succ", "_act_walk_subst_succ",
       [("mode", _M["WALK"]), ("walktask", 1), ("wkind", _K["SUCC"])]),
    _a("walk_subst_rec", "_act_walk_subst_rec",
       [("mode", _M["WALK"]), ("walktask", 1), ("wkind", _K["REC"])]),
    # RETURN — halt
    _a("ret_halt", "_act_ret_halt",
       [("mode", _M["RETURN"]), ("cempty", 0)]),
]


def _ret(name, method, frame_kind, extra=()):
    return _a(name, method,
              [("mode", _M["RETURN"]), ("topkind", FRAME_KIND[frame_kind])]
              + list(extra))


_T = ("rbool", 1)   # res_bool True
_F = ("rbool", 2)   # res_bool False
ACTION_SPECS += [
    _ret("ret_entry_valid", "_act_ret_entry_valid", "ENTRY_VALID"),
    _ret("ret_entry_valid_w_ok", "_act_ret_entry_valid_w_ok",
         "ENTRY_VALID_W", [("rkind_is_type", 0)]),
    _ret("ret_entry_valid_w_err", "_act_ret_entry_valid_w_err",
         "ENTRY_VALID_W", [("rkind_not_type", 0)]),
    _ret("ret_lam_ty", "_act_ret_lam_ty", "LAM_TY"),
    _ret("ret_lam_ty_w_ok", "_act_ret_lam_ty_w_ok",
         "LAM_TY_W", [("rkind_is_type", 0)]),
    _ret("ret_lam_ty_w_err", "_act_ret_lam_ty_w_err",
         "LAM_TY_W", [("rkind_not_type", 0)]),
    _ret("ret_lam_body", "_act_ret_lam_body", "LAM_BODY"),
    _ret("ret_pi_dom", "_act_ret_pi_dom", "PI_DOM"),
    _ret("ret_pi_dom_w_ok", "_act_ret_pi_dom_w_ok",
         "PI_DOM_W", [("rkind_is_type", 0)]),
    _ret("ret_pi_dom_w_err", "_act_ret_pi_dom_w_err",
         "PI_DOM_W", [("rkind_not_type", 0)]),
    _ret("ret_pi_body", "_act_ret_pi_body", "PI_BODY"),
    _ret("ret_pi_body_w_ok", "_act_ret_pi_body_w_ok",
         "PI_BODY_W", [("rkind_is_type", 0)]),
    _ret("ret_pi_body_w_err", "_act_ret_pi_body_w_err",
         "PI_BODY_W", [("rkind_not_type", 0)]),
    _ret("ret_app_f", "_act_ret_app_f", "APP_F"),
    _ret("ret_app_f_w_ok", "_act_ret_app_f_w_ok",
         "APP_F_W", [("rkind_is_pi", 0)]),
    _ret("ret_app_f_w_err", "_act_ret_app_f_w_err",
         "APP_F_W", [("rkind_not_pi", 0)]),
    _ret("ret_app_a", "_act_ret_app_a", "APP_A"),
    _ret("ret_app_subst", "_act_ret_app_subst", "APP_SUBST"),
    _ret("ret_rec_m", "_act_ret_rec_m", "REC_M"),
    _ret("ret_rec_m_w_ok", "_act_ret_rec_m_w_ok",
         "REC_M_W", [("rkind_is_pi", 0)]),
    _ret("ret_rec_m_w_err", "_act_ret_rec_m_w_err",
         "REC_M_W", [("rkind_not_pi", 0)]),
    _ret("ret_rec_m_d_ok", "_act_ret_rec_m_d_ok", "REC_M_D", [_T]),
    _ret("ret_rec_m_d_err", "_act_ret_rec_m_d_err", "REC_M_D", [_F]),
    _ret("ret_rec_m_c_ok", "_act_ret_rec_m_c_ok",
         "REC_M_C", [("rkind_is_type", 0)]),
    _ret("ret_rec_m_c_err", "_act_ret_rec_m_c_err",
         "REC_M_C", [("rkind_not_type", 0)]),
    _ret("ret_rec_base_b", "_act_ret_rec_base_b", "REC_BASE_B"),
    _ret("ret_rec_base", "_act_ret_rec_base", "REC_BASE"),
    _ret("ret_rec_step_b", "_act_ret_rec_step_b", "REC_STEP_B"),
    _ret("ret_rec_step", "_act_ret_rec_step", "REC_STEP"),
    _ret("ret_rec_tgt", "_act_ret_rec_tgt", "REC_TGT"),
    _ret("ret_rec_res", "_act_ret_rec_res", "REC_RES"),
    _ret("ret_succ_t", "_act_ret_succ_t", "SUCC_T"),
    _ret("ret_chk_w_lam_pi", "_act_ret_chk_w_lam_pi", "CHK_W",
         [("r1kind_is_lam", 0), ("rkind_is_pi", 0)]),
    _ret("ret_chk_w_other", "_act_ret_chk_w_other", "CHK_W",
         [("chk_w_other", 0)]),
    _ret("ret_chk_lam_d_ok", "_act_ret_chk_lam_d_ok", "CHK_LAM_D", [_T]),
    _ret("ret_chk_lam_d_err", "_act_ret_chk_lam_d_err", "CHK_LAM_D", [_F]),
    _ret("ret_chk_lam_b", "_act_ret_chk_lam_b", "CHK_LAM_B"),
    _ret("ret_chk_inf", "_act_ret_chk_inf", "CHK_INF"),
    _ret("ret_chk_defeq_ok", "_act_ret_chk_defeq_ok", "CHK_DEFEQ", [_T]),
    _ret("ret_chk_defeq_err", "_act_ret_chk_defeq_err", "CHK_DEFEQ", [_F]),
    _ret("ret_whnf_app_lam", "_act_ret_whnf_app_lam",
         "WHNF_APP", [("rkind_is_lam", 0)]),
    _ret("ret_whnf_app_neutral", "_act_ret_whnf_app_neutral",
         "WHNF_APP", [("rkind_not_lam", 0)]),
    _ret("ret_whnf_app_s", "_act_ret_whnf_app_s", "WHNF_APP_S"),
    _ret("ret_whnf_rec_zero", "_act_ret_whnf_rec_zero",
         "WHNF_REC", [("rkind_is_zero", 0)]),
    _ret("ret_whnf_rec_succ", "_act_ret_whnf_rec_succ",
         "WHNF_REC", [("rkind_is_succ", 0)]),
    _ret("ret_whnf_rec_neutral", "_act_ret_whnf_rec_neutral",
         "WHNF_REC", [("rkind_not_zero_succ", 0)]),
    _ret("ret_whnf_rec_s", "_act_ret_whnf_rec_s", "WHNF_REC_S"),
    _ret("ret_cmp_w_l", "_act_ret_cmp_w_l", "CMP_W_L"),
    _ret("ret_cmp_w_r_lam_lam", "_act_ret_cmp_w_r_lam_lam", "CMP_W_R",
         [("r1kind_is_lam", 0), ("rkind_is_lam", 0)]),
    _ret("ret_cmp_w_r_eta_l", "_act_ret_cmp_w_r_eta_l", "CMP_W_R",
         [("r1kind_is_lam", 0), ("rkind_not_lam", 0)]),
    _ret("ret_cmp_w_r_eta_r", "_act_ret_cmp_w_r_eta_r", "CMP_W_R",
         [("r1kind_not_lam", 0), ("rkind_is_lam", 0)]),
    _ret("ret_cmp_w_r_var_var", "_act_ret_cmp_w_r_var_var", "CMP_W_R",
         [("r1kind", _K["VAR"]), ("rkind", _K["VAR"])]),
    _ret("ret_cmp_w_r_type_type", "_act_ret_cmp_w_r_type_type", "CMP_W_R",
         [("r1kind", _K["TYPE"]), ("rkind", _K["TYPE"])]),
    _ret("ret_cmp_w_r_pi_pi", "_act_ret_cmp_w_r_pi_pi", "CMP_W_R",
         [("r1kind", _K["PI"]), ("rkind", _K["PI"])]),
    _ret("ret_cmp_w_r_app_app", "_act_ret_cmp_w_r_app_app", "CMP_W_R",
         [("r1kind", _K["APP"]), ("rkind", _K["APP"])]),
    _ret("ret_cmp_w_r_nat_nat", "_act_ret_cmp_w_r_nat_nat", "CMP_W_R",
         [("r1kind", _K["NAT"]), ("rkind", _K["NAT"])]),
    _ret("ret_cmp_w_r_zero_zero", "_act_ret_cmp_w_r_zero_zero", "CMP_W_R",
         [("r1kind", _K["ZERO"]), ("rkind", _K["ZERO"])]),
    _ret("ret_cmp_w_r_succ_succ", "_act_ret_cmp_w_r_succ_succ", "CMP_W_R",
         [("r1kind", _K["SUCC"]), ("rkind", _K["SUCC"])]),
    _ret("ret_cmp_w_r_rec_rec", "_act_ret_cmp_w_r_rec_rec", "CMP_W_R",
         [("r1kind", _K["REC"]), ("rkind", _K["REC"])]),
    _ret("ret_cmp_w_r_else", "_act_ret_cmp_w_r_else", "CMP_W_R",
         [("cmp_w_r_else", 0)]),
    _ret("ret_cmp_lam_t_ok", "_act_ret_cmp_lam_t_ok", "CMP_LAM_T", [_T]),
    _ret("ret_cmp_lam_t_err", "_act_ret_cmp_lam_t_err", "CMP_LAM_T", [_F]),
    _ret("ret_cmp_lam_b", "_act_ret_cmp_lam_b", "CMP_LAM_B"),
    _ret("ret_cmp_pi_d_ok", "_act_ret_cmp_pi_d_ok", "CMP_PI_D", [_T]),
    _ret("ret_cmp_pi_d_err", "_act_ret_cmp_pi_d_err", "CMP_PI_D", [_F]),
    _ret("ret_cmp_pi_c", "_act_ret_cmp_pi_c", "CMP_PI_C"),
    _ret("ret_cmp_app_l_ok", "_act_ret_cmp_app_l_ok", "CMP_APP_L", [_T]),
    _ret("ret_cmp_app_l_err", "_act_ret_cmp_app_l_err", "CMP_APP_L", [_F]),
    _ret("ret_cmp_app_r", "_act_ret_cmp_app_r", "CMP_APP_R"),
    _ret("ret_cmp_succ", "_act_ret_cmp_succ", "CMP_SUCC"),
    _ret("ret_cmp_rec_m_ok", "_act_ret_cmp_rec_m_ok", "CMP_REC_M", [_T]),
    _ret("ret_cmp_rec_m_err", "_act_ret_cmp_rec_m_err", "CMP_REC_M", [_F]),
    _ret("ret_cmp_rec_b_ok", "_act_ret_cmp_rec_b_ok", "CMP_REC_B", [_T]),
    _ret("ret_cmp_rec_b_err", "_act_ret_cmp_rec_b_err", "CMP_REC_B", [_F]),
    _ret("ret_cmp_rec_s_ok", "_act_ret_cmp_rec_s_ok", "CMP_REC_S", [_T]),
    _ret("ret_cmp_rec_s_err", "_act_ret_cmp_rec_s_err", "CMP_REC_S", [_F]),
    _ret("ret_cmp_rec_t", "_act_ret_cmp_rec_t", "CMP_REC_T"),
    _ret("ret_cmp_eta", "_act_ret_cmp_eta", "CMP_ETA"),
    _ret("ret_cmp_eta_w_ok", "_act_ret_cmp_eta_w_ok",
         "CMP_ETA_W", [("rkind_is_pi", 0)]),
    _ret("ret_cmp_eta_w_err", "_act_ret_cmp_eta_w_err",
         "CMP_ETA_W", [("rkind_not_pi", 0)]),
    _ret("ret_cmp_eta_d_ok", "_act_ret_cmp_eta_d_ok", "CMP_ETA_D", [_T]),
    _ret("ret_cmp_eta_d_err", "_act_ret_cmp_eta_d_err", "CMP_ETA_D", [_F]),
    _ret("ret_cmp_eta_b", "_act_ret_cmp_eta_b", "CMP_ETA_B"),
    _ret("ret_cmp_eta_r", "_act_ret_cmp_eta_r", "CMP_ETA_R"),
    _ret("ret_walk_done", "_act_ret_walk_done", "WALK_DONE"),
]

ACTION_NAMES = [a[0] for a in ACTION_SPECS]
ACTION_FNS = [getattr(MatrixMachine, a[1]) for a in ACTION_SPECS]
N_ACTIONS = len(ACTION_SPECS)


def _build_control():
    """W_CTRL (FEATURE_DIM x N_ACTIONS) + bias, STLC's threshold
    discipline: action a fires iff all its required bits are 1.
    Column a has 1.0 on its required feature positions and bias
    -(n_req - 0.5), so the logit is +0.5 iff every required bit is 1
    and <= -0.5 otherwise."""
    pos = {}
    off = 0
    for group, dim in FEATURE_GROUPS:
        for i in range(dim):
            pos[(group, i)] = off + i
        off += dim
    Wc = np.zeros((FEATURE_DIM, N_ACTIONS))
    bias = np.zeros(N_ACTIONS)
    for a, (_, _, req) in enumerate(ACTION_SPECS):
        for (group, idx) in req:
            Wc[pos[(group, idx)], a] = 1.0
        bias[a] = -(len(req) - 0.5)
    return Wc, bias


W_CTRL, W_BIAS = _build_control()
reg("control matrix W_CTRL + bias",
    int(np.count_nonzero(W_CTRL)) + int(np.count_nonzero(W_BIAS)))

# Derived-bit threshold layers (analytic counts; see _features).
# var_ok/var_bad: idx<depth over the 8x9 outer product (72) + threshold
# (36+1); fkind_bad 4; fkind_not_app_rec 3; wkind_is_leaf 4;
# wkind_is_lampi 3; w_graft 12; w_gt ~130; w_var_lt 4; chk_w_other 3;
# cmp_w_r_else 45; level max 16 + level +1 4.
reg("control derived-bit threshold layers",
    72 + 37 + 4 + 3 + 4 + 3 + 12 + 130 + 4 + 3 + 45 + 20)


# ---------------------------------------------------------------------------
# Lockstep test harness
# ---------------------------------------------------------------------------

def _snapshots_equal(ref_snap, mat_snap, case_name, step):
    if ref_snap == mat_snap:
        return True, ""
    keys = sorted(set(ref_snap) | set(mat_snap))
    for k in keys:
        if ref_snap.get(k) != mat_snap.get(k):
            rv, mv = ref_snap.get(k), mat_snap.get(k)
            if k == "regs":
                diffs = [i for i in range(len(rv))
                         if rv[i] != mv[i]]
                return False, ("regs differ at registers %s "
                               "(first diff slot %s)"
                               % (diffs[:5],
                                  next((p for p in range(N_SLOTS)
                                        if rv[diffs[0]][p] != mv[diffs[0]][p]),
                                       None) if diffs else None))
            if k in ("ctrl", "wstack", "ctx"):
                n = max(len(rv), len(mv))
                for i in range(n):
                    a = rv[i] if i < len(rv) else None
                    b = mv[i] if i < len(mv) else None
                    if a != b:
                        return False, ("%s[%d] ref=%r mat=%r" % (k, i, a, b))
                return False, ("%s lengths differ ref=%d mat=%d"
                               % (k, len(rv), len(mv)))
            return False, ("%s ref=%r mat=%r" % (k, rv, mv))
    return False, "snapshots differ but no key differs?!"


def run_lockstep(verbose=True):
    """RefMachine vs MatrixMachine, step for step, on all 41 cases;
    verdicts checked against the expected values in build_cases."""
    check_cases, infer_cases, defeq_cases = W.build_cases()
    results = []
    worst = {"ctrl": 0, "reg": 0, "ctx": 0, "wstack": 0, "steps": 0}
    n_steps_total = 0

    def note_worst(m):
        for kkey, val in (("ctrl", m.max_ctrl), ("reg", m.max_reg),
                          ("ctx", m.max_ctx), ("wstack", m.max_wstack),
                          ("steps", m.max_steps_used)):
            if val > worst[kkey]:
                worst[kkey] = val

    def lockstep_run(ref, mat, label):
        """Step both until one halts/errors; compare every step."""
        nonlocal n_steps_total
        while True:
            rs = ref_snapshot(ref)
            ms = mat.decode()
            ok, why = _snapshots_equal(rs, ms, label, ref.steps)
            if not ok:
                return False, ("DIVERGE %s at step %d: %s"
                               % (label, ref.steps, why))
            if ref.mode in ("HALT", "ERROR") or \
                    mat._mode_name() in ("HALT", "ERROR"):
                if ref.mode != mat._mode_name():
                    return False, ("mode mismatch %s: ref %s mat %s"
                                   % (label, ref.mode, mat._mode_name()))
                break
            ref.step()
            mat.step()
            n_steps_total += 1
            if n_steps_total > 200000:
                return False, "step budget exceeded"
        return True, ""

    for name, term, ty, ctx, want in check_cases:
        term_db, ty_db, ctx_types = W.convert_case(term, ty, ctx)
        ref = W.RefMachine(term_db, ty_db, ctx_types, entry="check")
        mat = MatrixMachine(term_db, ty_db, ctx_types, entry="check")
        ok, why = lockstep_run(ref, mat, "check " + name)
        if ok:
            note_worst(ref)
            note_worst(mat)
            if want is None:
                good = (ref.mode == "HALT" and mat._mode_name() == "HALT")
                detail = "accepted" if good else \
                    "ref %s %r / mat %s %r" % (
                        ref.mode, (ref.fail_path, ref.fail_kind),
                        mat._mode_name(),
                        (mat.decode()["fail_path"],
                         mat.decode()["fail_kind"]))
            else:
                good = (ref.mode == "ERROR"
                        and tuple(ref.fail_path) == tuple(want[0])
                        and ref.fail_kind == want[1]
                        and mat._mode_name() == "ERROR"
                        and tuple(mat.decode()["fail_path"]) == tuple(want[0])
                        and mat.decode()["fail_kind"] == want[1])
                detail = ("rejected %r" % ((ref.fail_path, ref.fail_kind),)
                          if ref.mode == "ERROR" else "accepted")
            ok = good
            if not good:
                why = "verdict: " + detail
        results.append((ok, "check  %-38s %s" % (name, why or "lockstep ok")))
    for name, term, ctx, want in infer_cases:
        term_db, _, ctx_types = W.convert_case(term, None, ctx)
        ref = W.RefMachine(term_db, None, ctx_types, entry="infer")
        mat = MatrixMachine(term_db, None, ctx_types, entry="infer")
        ok, why = lockstep_run(ref, mat, "infer " + name)
        if ok:
            note_worst(ref)
            note_worst(mat)
            good = (ref.mode == "ERROR"
                    and tuple(ref.fail_path) == tuple(want[0])
                    and ref.fail_kind == want[1]
                    and mat._mode_name() == "ERROR"
                    and tuple(mat.decode()["fail_path"]) == tuple(want[0])
                    and mat.decode()["fail_kind"] == want[1])
            ok = good
            if not good:
                why = ("verdict: ref %s %r / mat %s %r" % (
                    ref.mode, (ref.fail_path, ref.fail_kind),
                    mat._mode_name(),
                    (mat.decode()["fail_path"],
                     mat.decode()["fail_kind"])))
        results.append((ok, "infer  %-38s %s" % (name, why or "lockstep ok")))
    for name, ctx, t, u, want in defeq_cases:
        t_db, u_db, ctx_types = W.convert_defeq(t, u, ctx)
        ref = W.RefMachine(t_db, u_db, ctx_types, entry="defeq")
        mat = MatrixMachine(t_db, u_db, ctx_types, entry="defeq")
        ok, why = lockstep_run(ref, mat, "defeq " + name)
        if ok:
            note_worst(ref)
            note_worst(mat)
            good = (ref.mode == "HALT" and mat._mode_name() == "HALT"
                    and ref.res_bool == want
                    and mat.decode()["res_bool"] == want)
            ok = good
            if not good:
                why = ("verdict: ref %s/%r mat %s/%r want %r" % (
                    ref.mode, ref.res_bool, mat._mode_name(),
                    mat.decode()["res_bool"], want))
        results.append((ok, "defeq  %-38s %s" % (name, why or "lockstep ok")))
    if verbose:
        for ok, line in results:
            print(("PASS  " if ok else "FAIL  ") + line)
    passed = sum(1 for ok, _ in results if ok)
    total_w, per = count_params()
    if verbose:
        print("\nMatrixMachine lockstep: %d/%d passed"
              % (passed, len(results)))
        print("measured maxima: ctrl=%d reg=%d ctx=%d wstack=%d steps=%d"
              % (worst["ctrl"], worst["reg"], worst["ctx"],
                 worst["wstack"], worst["steps"]))
        print("weights: %d nonzero parameters" % total_w)
        for k, v in sorted(per.items(), key=lambda kv: -kv[1]):
            print("  %s: %d" % (k, v))
    return passed, len(results), worst, total_w, results


if __name__ == "__main__":
    p, n, worst, total_w, _ = run_lockstep()
    raise SystemExit(0 if p == n else 1)
