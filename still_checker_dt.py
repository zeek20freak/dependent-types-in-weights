"""The Still on the DT checker — measure the floor of the MatrixMachine.

The Still's question, pointed at the DT weights kernel: what is the
smallest weight configuration that still checks dependent types?
The net: the MatrixMachine registry in matrix_machine_dt.py
(1,338,752 nonzero parameters, 26 entries), sealed by lockstep 41/41
(RefMachine vs MatrixMachine, step-for-step) with verdicts matching
kernel_dt.py (DT 20/20, Nat 21/21).

Method (adapted from still_checker.py / STILL-CHECKER.md, the STLC
template — same three phases, rescaled to a registry 276x larger):

  Phase 0 — baseline: re-run the full lockstep unpatched (own run).
    The bar for every promote below: full lockstep 41/41 AND
    verdicts matching kernel_dt.py. A prune promotes only if both
    hold. Screening uses a fast verdict-only runner (MatrixMachine
    alone, no RefMachine, no per-step decode: 41/41 in ~10s);
    anything that passes the screen still gets the full lockstep
    before it promotes.

  Phase 1 — coverage: one instrumented run of the 41 cases records
    which registry weights are READ. A weight is never-read iff its
    input component is always zero at every step where its
    construction is evaluated (STLC coverage.py's rule — a proof
    from the ledger, not a prediction). The never-read set gives
    the pruning-only floor. Two kinds of entries:
      * stored matrices (12): ptr permutations, shift matrices,
        W_CTRL + bias — real sparse matrices, patchable, per-weight
        coverage via the input columns/rows actually exercised.
      * analytic constructions (14): counted by nnz but implemented
        inline (node read/write, register-file gating, frame/walk/
        ctx write routing, free-stack routing, liveness OR,
        fail-path, eta error-catch, comparison bits, derived-bit
        layers). Their read sets are measured at the call sites
        from the actual arguments/values (the matrix form's read
        semantics, applied to the recorded values). The floor for
        these is a modeled floor: the construction is one piece of
        code, so the never-read part is a cost-model statement,
        not a code change — disclosed, not hidden.

  Phase 2 — entry prunes: zero/disable each of the 26 entries as a
    whole, run the screen (fast-fail staged), then the full lockstep
    for anything that passes the screen. A break names the failing
    case and step (first differing field).

  Phase 3 — batch confirm + transpose twins: zero all never-read
    weights in the stored matrices as one batch and re-run the bar
    (the STLC seal's coverage batch prune); then check which
    stored-matrix pairs survive as exact transposes.

Caveats (carried, not buried):
  * The floor is against the 41-case bar, not a universal minimum.
  * Two registry entries are estimates: 'comparison bits' (512) is
    a bare literal, and 'control derived-bit threshold layers'
    (337) has a '~130' component (w_gt). The floor inherits a
    +/-500 softness from these. The two entries also appear to
    overlap (both cover the VAR-idx / SUBST-idx / level-max
    threshold layers — the 512's comparisons are the same ones the
    337's derived bits implement); if so the baseline itself is
    overstated by up to ~512. Flagged as a registry question.
  * The liveness OR count (1,664) assumes ~103 sel fields; the
    field enumeration used here is the same 103 (see _live_bits).
  * The eta error-catch truncation gating (2,278) is a coarse
    analytic count (ctrl stack // 64); it is treated as read iff
    a catch happens in the bar.

This file never modifies matrix_machine_dt.py, weights_kernel_dt.py,
subst_circuit.py, or any sealed file. It monkeypatches module
attributes and methods at runtime and restores them (try/finally).
"""

import json
import time

import numpy as np
import scipy.sparse as sp

import matrix_machine_dt as mm
import weights_kernel_dt as W

# ---------------------------------------------------------------------------
# Fast verdict runner (the screen): MatrixMachine alone, no RefMachine,
# no per-step decode. Mirrors run_lockstep's final verdict logic.
# ---------------------------------------------------------------------------

# Stage-1 cases: the heaviest machinery first, so most prunes fail fast.
STAGE1 = ["induction principle", "eta through check", "S",
          "add 2 3 : Nat"]


def _case_verdict(kind, name, mat, want):
    """Check one finished MatrixMachine against the expected verdict.
    Returns (ok, detail)."""
    mode = mat._mode_name()
    d = mat.decode()
    if kind == "check":
        if want is None:
            ok = (mode == "HALT")
            detail = "accepted" if ok else "mode %s" % mode
        else:
            ok = (mode == "ERROR"
                  and tuple(d["fail_path"]) == tuple(want[0])
                  and d["fail_kind"] == want[1])
            detail = ("rejected %r" % ((d["fail_path"], d["fail_kind"]),)
                      if mode == "ERROR" else "mode %s" % mode)
    elif kind == "infer":
        ok = (mode == "ERROR"
              and tuple(d["fail_path"]) == tuple(want[0])
              and d["fail_kind"] == want[1])
        detail = ("rejected %r" % ((d["fail_path"], d["fail_kind"]),)
                  if mode == "ERROR" else "mode %s" % mode)
    else:  # defeq
        ok = (mode == "HALT" and d["res_bool"] == want)
        detail = "defeq=%r want=%r" % (d["res_bool"], want)
    return ok, detail


def _run_one(kind, case, max_steps):
    """Run one case on the MatrixMachine alone. Returns
    (ok, label, detail). Exceptions are failures, not crashes."""
    try:
        if kind == "check":
            name, term, ty, ctx, want = case
            term_db, ty_db, ctx_types = W.convert_case(term, ty, ctx)
            mat = mm.MatrixMachine(term_db, ty_db, ctx_types,
                                   entry="check", max_steps=max_steps)
        elif kind == "infer":
            name, term, ctx, want = case
            term_db, _, ctx_types = W.convert_case(term, None, ctx)
            mat = mm.MatrixMachine(term_db, None, ctx_types,
                                   entry="infer", max_steps=max_steps)
        else:
            name, ctx, t, u, want = case
            t_db, u_db, ctx_types = W.convert_defeq(t, u, ctx)
            mat = mm.MatrixMachine(t_db, u_db, ctx_types,
                                   entry="defeq", max_steps=max_steps)
        mat.run()
        ok, detail = _case_verdict(kind, name, mat, want)
        return ok, "%s %s" % (kind, name), detail
    except Exception as e:  # noqa: BLE001 - a prune breaking the machine
        # is a measurement, and the exception text is the evidence.
        return False, "%s %s" % (kind, case[0]), "EXC %s: %s" % (
            type(e).__name__, e)


def fast_verdicts(max_steps=2000, verbose=False):
    """Staged verdict screen over all 41 cases. Returns
    (passed, total, failures) where failures is a list of
    (label, detail) for the first failing stage only (fail-fast)."""
    check_cases, infer_cases, defeq_cases = W.build_cases()
    by_name = {}
    for kind, cases in (("check", check_cases), ("infer", infer_cases),
                        ("defeq", defeq_cases)):
        for case in cases:
            by_name[case[0]] = (kind, case)
    failures = []
    passed = 0
    total = len(by_name)
    # Stage 1: heavy cases first.
    for name in STAGE1:
        kind, case = by_name[name]
        ok, label, detail = _run_one(kind, case, max_steps)
        if ok:
            passed += 1
        else:
            failures.append((label, detail))
            if verbose:
                print("  screen stage 1 FAIL %s: %s" % (label, detail))
            return passed, total, failures
    # Stage 2: the rest.
    for name, (kind, case) in by_name.items():
        if name in STAGE1:
            continue
        ok, label, detail = _run_one(kind, case, max_steps)
        if ok:
            passed += 1
        else:
            failures.append((label, detail))
            if verbose:
                print("  screen stage 2 FAIL %s: %s" % (label, detail))
            return passed, total, failures
    return passed, total, failures


def baseline_lockstep():
    """Phase 0: the full bar, unpatched, by my own run."""
    t0 = time.time()
    p, n, worst, total_w, results = mm.run_lockstep(verbose=False)
    dt = time.time() - t0
    fails = [line for ok, line in results if not ok]
    print("phase 0 baseline lockstep: %d/%d passed in %.0fs "
          "(maxima ctrl=%d reg=%d ctx=%d wstack=%d steps=%d)"
          % (p, n, dt, worst["ctrl"], worst["reg"], worst["ctx"],
             worst["wstack"], worst["steps"]), flush=True)
    for f in fails:
        print("  FAIL " + f, flush=True)
    return p == n, p, n, worst


# ---------------------------------------------------------------------------
# Coverage instrumentation
# ---------------------------------------------------------------------------

class CountingMat:
    """Wrap a scipy sparse matrix; record which input columns (for
    M @ x) or rows (for x @ M) are nonzero on each use. Returns
    identical results. Unknown attributes delegate to the matrix."""

    def __init__(self, mat, rec):
        self._m = mat
        self._rec = rec

    def __matmul__(self, other):
        x = np.asarray(other)
        self._rec.update(int(i) for i in np.nonzero(x)[0])
        return self._m @ other

    def __rmatmul__(self, other):
        x = np.asarray(other)
        self._rec.update(int(i) for i in np.nonzero(x)[0])
        return other @ self._m

    def __getattr__(self, attr):
        return getattr(self._m, attr)


# The coverage record: sets of exercised weights/positions per entry.
COV = {}


def _fresh_cov():
    return {
        "mat_cols": {},       # matrix attr -> set of input indices used
        "feat_rows": set(),   # W_CTRL feature positions ever nonzero
        "fidx_depth": set(),  # (fidx, depth) pairs seen
        "widx_wp1": set(),    # (widx, wp1, is_var) triples seen
        "r1k_rk": set(),      # (r1kind, rkind) pairs seen
        "fkind_seen": set(), "wkind_seen": set(),
        "derived_ever1": set(),  # derived-bit group names ever 1
        "read_nodes": set(),     # (sel, ptr) read via read_node
        "read_node_fields": set(),  # (ptr, field) E_p positions read
        "read_sel_fields": set(),   # (sel, field) mux positions read
        "write_nodes": set(),       # (ptr, field) node-write positions
        "erase_read": set(),        # (slot, field) erase positions read
        "gating_read": {},          # reg r -> set of (slot, field)
        "frame_write_pos": set(),   # frame-vector positions written
        "walk_write_pos": set(),    # walk-item positions written
        "ctx_write_pos": set(),     # ctx-slot positions written
        "regs_alloc": set(), "regs_freed": set(),
        "live_pairs": set(),        # (reg, field-name) liveness OR
        "err_slots_kinds": set(),   # (slot, kind) present at errors
        "err_path_routing": set(),  # (j, i, seg) path-slot routing
        "err_any": False,
        "catch_any": False,
        "eta_scan_slots": set(),    # slots scanned by eta encoders
        "level_pairs": set(),       # (i, j) level-max pairs
        "actions": {},              # action name -> count
        "steps": 0,
    }


# Stored-matrix attributes, and the registry entry each belongs to.
# LEFT_CHILD and REC_MOTIVE_PTR are the same object (aliased, counted
# once); they share one recording set.
MAT_ATTRS = {
    "ptr left-child permutation": ["LEFT_CHILD", "REC_MOTIVE_PTR"],
    "ptr right-child permutation": ["RIGHT_CHILD"],
    "ptr parent permutation": ["PARENT_PTR"],
    "ptr rec-base permutation": ["REC_BASE_PTR"],
    "ptr rec-step permutation": ["REC_STEP_PTR"],
    "ptr rec-target permutation": ["REC_TARGET_PTR"],
    "ctx shift-down": ["SHIFT_DOWN_CTX"],
    "ctx shift-up": ["SHIFT_UP_CTX"],
    "ctrl shift-down": ["SHIFT_DOWN_CTRL"],
    "ctrl shift-up": ["SHIFT_UP_CTRL"],
    "wstack shift-down": ["SHIFT_DOWN_WALK"],
    "wstack shift-up": ["SHIFT_UP_WALK"],
}


def _install_coverage():
    """Install CountingMat wrappers + method wrappers. Returns a
    restore callable."""
    global COV
    COV = _fresh_cov()
    saved_mats = {}
    saved_methods = {}
    saved_action_fns = []

    # Stored matrices: wrap with CountingMat (shared set per entry).
    for entry, attrs in MAT_ATTRS.items():
        rec = set()
        COV["mat_cols"][entry] = rec
        for attr in attrs:
            orig = getattr(mm, attr)
            saved_mats[attr] = orig
            setattr(mm, attr, CountingMat(orig, rec))

    def wrap_method(name, fn):
        orig = getattr(mm.MatrixMachine, name)
        saved_methods[name] = orig

        def wrapped(self, *args, _orig=orig, _fn=fn, **kwargs):
            return _fn(self, _orig, *args, **kwargs)
        setattr(mm.MatrixMachine, name, wrapped)

    # _feature_vector: W_CTRL rows (feature positions ever nonzero)
    # + the action this step will fire.
    def fv(self, orig):
        vec, pos = orig(self)
        COV["feat_rows"].update(int(i) for i in np.nonzero(vec)[0])
        COV["steps"] += 1
        logits = vec @ mm.W_CTRL + mm.W_BIAS
        action = np.maximum(0.0, logits)
        ai = int(np.argmax(action))
        aname = mm.ACTION_NAMES[ai]
        COV["actions"][aname] = COV["actions"].get(aname, 0) + 1
        return vec, pos
    wrap_method("_feature_vector", fv)

    # _features: derived-bit inputs (raw one-hots + derived bits).
    def feats(self, orig):
        f = orig(self)
        fidx = int(np.argmax(f["fidx"]))
        depth = int(np.argmax(f["ctxdepth"]))
        COV["fidx_depth"].add((fidx, depth))
        widx = int(np.argmax(f["widx"]))
        wp1 = int(np.argmax(f["wp1"]))
        is_var = f["wkind"][mm.KIND["VAR"]] > 0.5
        COV["widx_wp1"].add((widx, wp1, bool(is_var)))
        COV["r1k_rk"].add((int(np.argmax(f["r1kind"])),
                           int(np.argmax(f["rkind"]))))
        COV["fkind_seen"].add(int(np.argmax(f["fkind"])))
        COV["wkind_seen"].add(int(np.argmax(f["wkind"])))
        for grp in ("var_ok", "var_bad", "fkind_bad",
                    "fkind_not_app_rec", "rkind_is_lam", "rkind_not_lam",
                    "rkind_is_pi", "rkind_not_pi", "rkind_is_type",
                    "rkind_not_type", "rkind_is_zero", "rkind_is_succ",
                    "rkind_not_zero_succ", "r1kind_is_lam",
                    "r1kind_not_lam", "wkind_is_var", "wkind_is_leaf",
                    "wkind_is_lampi", "w_graft", "w_gt", "w_var_lt",
                    "chk_w_other", "cmp_w_r_else"):
            if f[grp][0] > 0.5:
                COV["derived_ever1"].add(grp)
        return f
    wrap_method("_features", feats)

    # read_node: E_p + sel mux.
    def read_node(self, orig, sel, ptr):
        node = orig(self, sel, ptr)
        COV["read_nodes"].add((int(sel), int(ptr)))
        for f in np.nonzero(node)[0]:
            COV["read_node_fields"].add((int(ptr), int(f)))
            COV["read_sel_fields"].add((int(sel), int(f)))
        return node
    wrap_method("read_node", read_node)

    # read_level: level values read (level-max inputs).
    def read_level(self, orig, sel, ptr):
        lv = orig(self, sel, ptr)
        return lv
    wrap_method("read_level", read_level)

    # write_node_reg: node-write write part + erase part (old tree).
    def write_node_reg(self, orig, r, ptr, kind, level=0, idx=0):
        old = self.regs[r]
        for (q, f) in zip(*np.nonzero(old)):
            if int(q) != int(ptr):
                COV["erase_read"].add((int(q), int(f)))
        node = np.zeros(mm.SLOT_DIM)
        node[mm.KIND_OFF + mm.KIND[kind]] = 1.0
        node[mm.LEVEL_OFF + level] = 1.0
        node[mm.IDX_OFF + idx] = 1.0
        for f in np.nonzero(node)[0]:
            COV["write_nodes"].add((int(ptr), int(f)))
        return orig(self, r, ptr, kind, level, idx)
    wrap_method("write_node_reg", write_node_reg)

    # write_tree_reg: register-file gating (per register).
    def write_tree_reg(self, orig, r, tree):
        arr = np.asarray(tree)
        s = COV["gating_read"].setdefault(int(r), set())
        for (q, f) in zip(*np.nonzero(arr)):
            s.add((int(q), int(f)))
        return orig(self, r, tree)
    wrap_method("write_tree_reg", write_tree_reg)

    # _alloc: free-stack pop (register allocated).
    def alloc(self, orig):
        r = orig(self)
        COV["regs_alloc"].add(int(r))
        return r
    wrap_method("_alloc", alloc)

    # _sweep: free-stack push (registers freed) + liveness OR pairs.
    def sweep(self, orig):
        # Liveness OR: enumerate the sel fields exactly as
        # _live_bits does, and record (reg, field) pairs where the
        # field references that register.
        d = self._ctrl_depth()
        fields = []
        for i in range(d):
            fr = self._frame_read_slot(i)
            for j, ref in enumerate(
                    (fr[1], fr[2], fr[3], fr[4])):
                fields.append((("ctrl", i, "r%d" % (j + 1)), ref))
            if fr[0] in W.RefMachine.SMALLS_REF_FRAMES:
                fields.append((("ctrl", i, "smalls"),
                               (fr[5], 0) if isinstance(fr[5], int)
                               else None))
        for i in range(self._ctx_depth()):
            fields.append((("ctx", i), self._ctx_ref(i)))
        fields.append((("dest",), (self._dest_idx(), 0)
                       if self._dest_idx() >= 0 else None))
        fields.append((("res_sel",),
                       (int(np.argmax(self.res_sel)), 0)))
        fields.append((("fsel",), (int(np.argmax(self.fsel)), 0)))
        fields.append((("asel",), (int(np.argmax(self.asel)), 0)))
        for i in range(self._wstack_depth()):
            it = self._walk_read_slot(i)
            fields.append((("wstack", i, "ssel"), (it[1], 0)))
            fields.append((("wstack", i, "dsel"), (it[3], 0)))
            if isinstance(it[7], int) and it[7] >= 0:
                fields.append((("wstack", i, "rsel"), (it[7], 0)))
        for fname, ref in fields:
            if ref is None:
                continue
            sel = ref[0]
            # A sel is a register ref only for 2 <= sel < 2 + POOL_SIZE;
            # sel == N_SRC (18) means "none" (e.g. res_sel) and is not
            # a register.
            if isinstance(sel, int) and 2 <= sel < 2 + mm.POOL_SIZE:
                COV["live_pairs"].add((sel - 2, fname))
        before = self._free_set()
        out = orig(self)
        after = self._free_set()
        for r in after - before:
            COV["regs_freed"].add(int(r))
        return out
    wrap_method("_sweep", sweep)

    # _frame_write_slot0 / _walk_write_slot0 / _ctx_write_slot:
    # record the written vector's nonzero positions.
    def frame_write(self, orig, kind, refs, smalls, saved_dest, wdst):
        out = orig(self, kind, refs, smalls, saved_dest, wdst)
        vec = self.ctrl[0:mm.FRAME_DIM]
        COV["frame_write_pos"].update(
            int(i) for i in np.nonzero(vec)[0])
        return out
    wrap_method("_frame_write_slot0", frame_write)

    def walk_write(self, orig, task, ssel, sptr, dsel, dptr, p1, p2,
                   rsel, rptr):
        out = orig(self, task, ssel, sptr, dsel, dptr, p1, p2, rsel, rptr)
        vec = self.wstack[0:mm.WALK_DIM]
        COV["walk_write_pos"].update(
            int(i) for i in np.nonzero(vec)[0])
        return out
    wrap_method("_walk_write_slot0", walk_write)

    def ctx_write(self, orig, i, sel, ptr):
        out = orig(self, i, sel, ptr)
        base = i * mm.CTX_SLOT_DIM
        vec = self.ctx[base:base + mm.CTX_SLOT_DIM]
        COV["ctx_write_pos"].update(
            int(j) for j in np.nonzero(vec)[0])
        return out
    wrap_method("_ctx_write_slot", ctx_write)

    # _error: fail-path construction.
    def error(self, orig, kind, suffix=()):
        d = self._ctrl_depth()
        kinds = []
        for i in range(d):
            fr = self._frame_read_slot(i)
            kinds.append(fr[0])
            COV["err_slots_kinds"].add((i, fr[0]))
        COV["err_any"] = True
        out = orig(self, kind, suffix)
        # Path-slot routing: path slot j (j < d) takes the segment
        # of frame slot (d-1-j).
        for j in range(min(d, mm.MAX_PATH_LEN)):
            seg = self.fail_path[j]
            s = int(np.argmax(seg))
            COV["err_path_routing"].add(
                (j, d - 1 - j, mm.PATH_SEGS[s]))
        return out
    wrap_method("_error", error)

    # _try_catch_error: eta error-catch (encoder scans + catch?).
    def try_catch(self, orig):
        if self._mode_name() != "ERROR":
            return orig(self)
        if int(np.argmax(self.fail_kind)) == mm.FAIL_KIND["fuel"]:
            return orig(self)
        d = self._ctrl_depth()
        kinds = [self._frame_read_slot(i)[0] for i in range(d)]
        i_eta = None
        for i in range(d):
            if kinds[i] == "CMP_ETA":
                i_eta = i
                break
        if i_eta is None:
            for i in range(d):
                COV["eta_scan_slots"].add(i)
            return orig(self)
        for i in range(i_eta + 1):
            COV["eta_scan_slots"].add(i)
        j = None
        for i in range(i_eta + 1, d):
            COV["eta_scan_slots"].add(i)
            if kinds[i] == "CMP_W_R":
                j = i
                break
        out = orig(self)
        if out:
            COV["catch_any"] = True
        return out
    wrap_method("_try_catch_error", try_catch)

    # _act_ret_pi_body_w_ok: level-max pairs (domain level, codomain).
    # This action is dispatched through the module-level ACTION_FNS
    # list (function objects captured at import), so patching the
    # class attribute does NOT intercept it. Replace the ACTION_FNS
    # entry directly.
    _pi_idx = [i for i, a in enumerate(mm.ACTION_SPECS)
               if a[1] == "_act_ret_pi_body_w_ok"][0]
    _pi_orig = mm.ACTION_FNS[_pi_idx]

    def _pi_wrapped(self):
        fr = self._frame_read_slot(0)
        i = fr[5]
        j = self.read_level(*self._res_ref())
        COV["level_pairs"].add((int(i), int(j)))
        return _pi_orig(self)

    mm.ACTION_FNS[_pi_idx] = _pi_wrapped
    saved_action_fns.append((_pi_idx, _pi_orig))

    # __init__: ctx types are materialised into registers by direct
    # assignment (self.regs[r] = ...), bypassing write_tree_reg.
    # Those are tree writes into registers, so in matrix form they
    # read the register-file gating at the tree's nonzero positions.
    # Record them here (write_tree_reg itself is dead code).
    def mm_init(self, orig, *args, **kwargs):
        orig(self, *args, **kwargs)
        for i in range(self.max_ctx):
            sel, ptr = self._ctx_ref(i)
            if sel < 2:
                continue
            r = int(sel) - 2
            s = COV["gating_read"].setdefault(r, set())
            for (q, f) in zip(*np.nonzero(self.regs[r])):
                s.add((int(q), int(f)))
    wrap_method("__init__", mm_init)

    def restore():
        for attr, orig in saved_mats.items():
            setattr(mm, attr, orig)
        for name, orig in saved_methods.items():
            setattr(mm.MatrixMachine, name, orig)
        for idx, orig in saved_action_fns:
            mm.ACTION_FNS[idx] = orig
    return restore


def instrumented_run():
    """One run of all 41 cases, MatrixMachine alone, fully
    instrumented. Verifies the instrumented verdicts are 41/41
    (instrumentation must not change behavior). Returns the COV."""
    restore = _install_coverage()
    try:
        p, n, failures = fast_verdicts(max_steps=5000)
        print("instrumented run: verdicts %d/%d, %d steps recorded"
              % (p, n, COV["steps"]), flush=True)
        if p != n:
            for label, detail in failures:
                print("  FAIL " + label + ": " + detail, flush=True)
            raise RuntimeError("instrumented run broke the bar")
        return COV
    finally:
        restore()

# ---------------------------------------------------------------------------
# Never-read sets + pruning-only floor (from the coverage record)
# ---------------------------------------------------------------------------

def _mat_never_read(mat, used_cols):
    """For a stored sparse matrix used as M @ x: the weights (row, col)
    with col never nonzero at any use. Returns (read, never)."""
    coo = mat.tocoo()
    read = set()
    never = set()
    for r, c in zip(coo.row, coo.col):
        if int(c) in used_cols:
            read.add((int(r), int(c)))
        else:
            never.add((int(r), int(c)))
    return read, never


def compute_never_read(cov):
    """Per-entry (read, never) counts from the coverage record.
    Returns (per_entry, floor) where per_entry maps registry name to
    {'total': n, 'read': n, 'never': n, 'note': str} and floor is the
    pruning-only floor (baseline - total never-read)."""
    total, per = mm.count_params()
    out = {}

    def put(name, read_n, note=""):
        t = per[name]
        out[name] = {"total": t, "read": read_n, "never": t - read_n,
                     "note": note}

    # Stored matrices: per-weight, from the recorded input columns.
    for entry, attrs in MAT_ATTRS.items():
        mat = getattr(mm, attrs[0])
        # REC_MOTIVE_PTR aliases LEFT_CHILD; the first attr is the
        # canonical one (its wrapper shared the recording set).
        used = cov["mat_cols"][entry]
        read, never = _mat_never_read(mat, used)
        put(entry, len(read),
            "input positions used: %d" % len(used))

    # W_CTRL + bias: rows are feature positions; bias is added every
    # step, so all 116 bias weights are read.
    wc = mm.W_CTRL
    read_rows = cov["feat_rows"]
    read_w = sum(1 for i in range(wc.shape[0])
                 for j in range(wc.shape[1])
                 if wc[i, j] != 0 and i in read_rows)
    bias_n = int(np.count_nonzero(mm.W_BIAS))
    put("control matrix W_CTRL + bias", read_w + bias_n,
        "feature positions used: %d/156; bias all read" % len(read_rows))

    # Node-read E_p + sel mux.
    put("node-read slot extraction (E_p)", len(cov["read_node_fields"]),
        "slots read: %d" % len({p for p, _ in cov["read_node_fields"]}))
    put("node-read sel mux", len(cov["read_sel_fields"]),
        "sources read: %s" % sorted({s for s, _ in cov["read_sel_fields"]}))

    # Node-write: write part + erase part (separate 49,128 each).
    put("node-write (write + erase)",
        len(cov["write_nodes"]) + len(cov["erase_read"]),
        "write positions: %d, erase positions: %d"
        % (len(cov["write_nodes"]), len(cov["erase_read"])))

    # Register-file gating: per register.
    gating_read = sum(len(s) for s in cov["gating_read"].values())
    put("register-file per-register gating", gating_read,
        "registers written (write_tree_reg is dead; __init__ ctx writes): %s"
        % sorted(cov["gating_read"].keys()))

    # Write routings.
    put("frame write routing", len(cov["frame_write_pos"]))
    put("walk-item write routing", len(cov["walk_write_pos"]))
    put("ctx slot write routing", len(cov["ctx_write_pos"]))

    # Free-stack routing: push (freed) + pop (allocated), 16 each.
    put("free-stack push/pop routing",
        len(cov["regs_freed"]) + len(cov["regs_alloc"]),
        "registers freed: %d, allocated: %d"
        % (len(cov["regs_freed"]), len(cov["regs_alloc"])))

    # Liveness OR: (reg, field) pairs.
    put("liveness OR (live bits)", len(cov["live_pairs"]),
        "distinct (reg, field) pairs: %d" % len(cov["live_pairs"]))

    # Fail-path: kind->segment + prefix sums (read iff any error) +
    # path-slot routing.
    fp_read = (len(cov["err_slots_kinds"])
               + (98 if cov["err_any"] else 0)
               + len(cov["err_path_routing"]))
    put("fail-path construction", fp_read,
        "errors: %s, (slot,kind) pairs: %d, routing triples: %d"
        % (cov["err_any"], len(cov["err_slots_kinds"]),
           len(cov["err_path_routing"])))

    # Eta error-catch: encoders 4 x slots scanned + truncation gating
    # (read iff a catch happened).
    eta_read = (4 * len(cov["eta_scan_slots"])
                + (2278 if cov["catch_any"] else 0))
    put("eta error-catch construction", eta_read,
        "slots scanned: %d, catch happened: %s"
        % (len(cov["eta_scan_slots"]), cov["catch_any"]))

    # Comparison bits (512, bare literal): exercised as a whole.
    # Evidence that each comparison kind is evaluated in the bar.
    ev = []
    if "var_ok" in cov["derived_ever1"] or "var_bad" in cov["derived_ever1"]:
        ev.append("idx<depth (var_ok/var_bad) evaluated")
    if ("w_graft" in cov["derived_ever1"]
            or "w_gt" in cov["derived_ever1"]
            or "w_var_lt" in cov["derived_ever1"]):
        ev.append("SUBST idx==k/>k (w_graft/w_gt/w_var_lt) evaluated")
    if cov["level_pairs"]:
        ev.append("level max evaluated (%d pairs)"
                  % len(cov["level_pairs"]))
    put("comparison bits (idx/depth/level)", 512,
        "bare literal, kept whole; " + "; ".join(ev))

    # Derived-bit threshold layers (337): only the two outer-product
    # parts admit a per-weight never-read (var_ok/var_bad 72 from
    # (fidx, depth) pairs; level max 16 from (i, j) pairs). The rest
    # is evaluated every step and kept whole.
    deriv_read = (len(cov["fidx_depth"]) + 37
                  + len(cov["level_pairs"]) + 4 + 208)
    put("control derived-bit threshold layers", deriv_read,
        "(fidx,depth) pairs: %d/72; level pairs: %d/16; "
        "rest evaluated every step" % (len(cov["fidx_depth"]),
                                       len(cov["level_pairs"])))

    never_total = sum(v["never"] for v in out.values())
    floor = total - never_total
    return out, floor, total


def print_coverage(per_entry, floor, total):
    print("\ncoverage (one instrumented run, %d steps):" % COV["steps"],
          flush=True)
    for name, v in sorted(per_entry.items(), key=lambda kv: -kv[1]["never"]):
        print("  %-38s total %7d read %7d never %7d  %s"
              % (name, v["total"], v["read"], v["never"], v["note"]),
              flush=True)
    print("pruning-only floor (coverage): %d / %d (%.1f%%)"
          % (floor, total, 100.0 * floor / total), flush=True)
    print("actions fired: %s"
          % dict(sorted(COV["actions"].items(),
                        key=lambda kv: -kv[1])), flush=True)
    return floor

# ---------------------------------------------------------------------------
# Entry prunes (phase 2)
# ---------------------------------------------------------------------------

DERIVED_BITS = ["var_ok", "var_bad", "fkind_bad", "fkind_not_app_rec",
                "rkind_is_lam", "rkind_not_lam", "rkind_is_pi",
                "rkind_not_pi", "rkind_is_type", "rkind_not_type",
                "rkind_is_zero", "rkind_is_succ", "rkind_not_zero_succ",
                "r1kind_is_lam", "r1kind_not_lam", "wkind_is_var",
                "wkind_is_leaf", "wkind_is_lampi", "w_graft", "w_gt",
                "w_var_lt", "chk_w_other", "cmp_w_r_else"]
CMP_BITS = ["var_ok", "var_bad", "w_graft", "w_gt", "w_var_lt"]


def _zero_features(groups):
    """Wrap _features to zero the given derived-bit groups."""
    orig = mm.MatrixMachine._features

    def wrapped(self):
        f = orig(self)
        for g in groups:
            f[g] = np.zeros_like(f[g])
        return f
    return orig, wrapped


def install_prune(name):
    """Disable one registry entry; return a restore callable."""
    saved_mats = {}
    saved_methods = {}

    def save_mat(attr):
        saved_mats[attr] = getattr(mm, attr)

    def save_method(mname):
        saved_methods[mname] = getattr(mm.MatrixMachine, mname)

    if name in MAT_ATTRS:
        for attr in MAT_ATTRS[name]:
            save_mat(attr)
            orig = saved_mats[attr]
            setattr(mm, attr, sp.csr_matrix(orig.shape))
    elif name == "control matrix W_CTRL + bias":
        save_mat("W_CTRL")
        save_mat("W_BIAS")
        mm.W_CTRL = np.zeros_like(mm.W_CTRL)
        mm.W_BIAS = np.zeros_like(mm.W_BIAS)
    elif name == "node-read slot extraction (E_p)":
        save_method("read_node")

        def read_node_zero(self, sel, ptr):
            return np.zeros(mm.SLOT_DIM)
        mm.MatrixMachine.read_node = read_node_zero
    elif name == "node-read sel mux":
        save_method("_tree_of")

        def tree_of_input(self, sel):
            return self.input_tree
        mm.MatrixMachine._tree_of = tree_of_input
    elif name == "node-write (write + erase)":
        save_method("write_node_reg")

        def write_node_noop(self, r, ptr, kind, level=0, idx=0):
            return None
        mm.MatrixMachine.write_node_reg = write_node_noop
    elif name == "register-file per-register gating":
        save_method("write_tree_reg")

        def write_tree_noop(self, r, tree):
            return None
        mm.MatrixMachine.write_tree_reg = write_tree_noop
    elif name == "frame write routing":
        save_method("_frame_write_slot0")

        def frame_write_noop(self, kind, refs, smalls, saved_dest, wdst):
            return None
        mm.MatrixMachine._frame_write_slot0 = frame_write_noop
    elif name == "walk-item write routing":
        save_method("_walk_write_slot0")

        def walk_write_noop(self, task, ssel, sptr, dsel, dptr, p1, p2,
                            rsel, rptr):
            return None
        mm.MatrixMachine._walk_write_slot0 = walk_write_noop
    elif name == "ctx slot write routing":
        save_method("_ctx_write_slot")

        def ctx_write_noop(self, i, sel, ptr):
            return None
        mm.MatrixMachine._ctx_write_slot = ctx_write_noop
    elif name == "free-stack push/pop routing":
        save_method("_alloc")

        def alloc_pruned(self):
            raise RuntimeError("pruned free-stack routing")
        mm.MatrixMachine._alloc = alloc_pruned
    elif name == "liveness OR (live bits)":
        save_method("_live_bits")

        def live_all(self):
            return np.ones(mm.POOL_SIZE)
        mm.MatrixMachine._live_bits = live_all
    elif name == "fail-path construction":
        save_method("_error")
        orig_err = saved_methods["_error"]

        def error_empty(self, kind, suffix=()):
            out = orig_err(self, kind, suffix)
            self.fail_path = np.zeros((mm.MAX_PATH_LEN, mm.N_PATH_SEGS))
            for j in range(mm.MAX_PATH_LEN):
                self.fail_path[j] = mm._onehot(mm.N_PATH_SEGS,
                                               mm.PATH_SEG[""])
            self.fail_kind = mm._onehot(mm.N_FAIL_KINDS,
                                        mm.FAIL_KIND["none"])
            return out
        mm.MatrixMachine._error = error_empty
    elif name == "eta error-catch construction":
        save_method("_try_catch_error")

        def catch_never(self):
            return False
        mm.MatrixMachine._try_catch_error = catch_never
    elif name == "comparison bits (idx/depth/level)":
        save_method("_features")
        orig_f, wrapped_f = _zero_features(CMP_BITS)
        mm.MatrixMachine._features = wrapped_f
    elif name == "control derived-bit threshold layers":
        save_method("_features")
        orig_f, wrapped_f = _zero_features(DERIVED_BITS)
        mm.MatrixMachine._features = wrapped_f
    else:
        raise ValueError("unknown entry %r" % name)

    def restore():
        for attr, orig in saved_mats.items():
            setattr(mm, attr, orig)
        for mname, orig in saved_methods.items():
            setattr(mm.MatrixMachine, mname, orig)
    return restore


def prune_entry(name):
    """Disable one entry; screen; if the screen passes, the full
    lockstep is the bar. Returns a result dict."""
    restore = install_prune(name)
    try:
        t0 = time.time()
        p, n, failures = fast_verdicts(max_steps=2000)
        dt = time.time() - t0
        if p != n:
            return {"entry": name, "result": "break", "stage": "screen",
                    "failures": failures, "seconds": round(dt, 1)}
        t0 = time.time()
        lp, ln, worst, total_w, results = mm.run_lockstep(verbose=False)
        dt2 = time.time() - t0
        fails = [line for ok, line in results if not ok]
        if lp != ln:
            return {"entry": name, "result": "break", "stage": "lockstep",
                    "failures": fails, "seconds": round(dt + dt2, 1)}
        return {"entry": name, "result": "hold", "stage": "lockstep",
                "failures": [], "seconds": round(dt + dt2, 1)}
    finally:
        restore()


# ---------------------------------------------------------------------------
# Batch confirm + transpose twins (phase 3)
# ---------------------------------------------------------------------------

def _pruned_matrix(mat, never):
    """Copy of a sparse matrix with the given (row, col) entries zeroed."""
    coo = mat.tocoo()
    keep = np.array([(int(r), int(c)) not in never
                     for r, c in zip(coo.row, coo.col)])
    return sp.csr_matrix((coo.data[keep],
                          (coo.row[keep], coo.col[keep])),
                         shape=coo.shape)


def stored_never_sets(cov):
    """(row, col) never-read sets for the stored matrices + W_CTRL."""
    out = {}
    for entry, attrs in MAT_ATTRS.items():
        mat = getattr(mm, attrs[0])
        used = cov["mat_cols"][entry]
        _, never = _mat_never_read(mat, used)
        out[entry] = never
    wc = mm.W_CTRL
    rows = cov["feat_rows"]
    never = set()
    for i in range(wc.shape[0]):
        if i in rows:
            continue
        for j in range(wc.shape[1]):
            if wc[i, j] != 0:
                never.add((i, j))
    out["control matrix W_CTRL + bias"] = never
    return out


def batch_confirm(never_sets):
    """Zero all never-read weights in the stored matrices (+ W_CTRL)
    as one batch; screen, then the full lockstep."""
    saved = {}
    for entry, never in never_sets.items():
        if entry == "control matrix W_CTRL + bias":
            saved["W_CTRL"] = mm.W_CTRL
            wc = mm.W_CTRL.copy()
            for (i, j) in never:
                wc[i, j] = 0.0
            mm.W_CTRL = wc
            continue
        for attr in MAT_ATTRS[entry]:
            if attr not in saved:
                saved[attr] = getattr(mm, attr)
        mat = saved[MAT_ATTRS[entry][0]]
        pruned = _pruned_matrix(mat, never)
        for attr in MAT_ATTRS[entry]:
            setattr(mm, attr, pruned)
    try:
        p, n, failures = fast_verdicts(max_steps=2000)
        if p != n:
            return {"result": "break", "stage": "screen",
                    "failures": failures}
        lp, ln, worst, total_w, results = mm.run_lockstep(verbose=False)
        fails = [line for ok, line in results if not ok]
        return {"result": "hold" if lp == ln else "break",
                "stage": "lockstep", "failures": fails}
    finally:
        for attr, orig in saved.items():
            setattr(mm, attr, orig)


def transpose_twins(never_sets):
    """After the batch prune, which stored-matrix pairs survive as
    exact transposes? Returns (twins, saving)."""
    pruned = {}
    for entry, attrs in MAT_ATTRS.items():
        mat = getattr(mm, attrs[0])
        pruned[entry] = _pruned_matrix(mat, never_sets[entry])
    twins = []
    names = list(pruned.keys())
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            a, b = pruned[n1], pruned[n2]
            if a.shape == b.T.shape and (a != b.T).nnz == 0:
                twins.append((n1, n2))
    saving = 0
    for _, n2 in twins:
        saving += int(pruned[n2].nnz)
    return twins, saving


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ENTRY_ORDER = [
    "ptr left-child permutation", "ptr right-child permutation",
    "ptr parent permutation", "ptr rec-base permutation",
    "ptr rec-step permutation", "ptr rec-target permutation",
    "ctx shift-down", "ctx shift-up",
    "ctrl shift-down", "ctrl shift-up",
    "wstack shift-down", "wstack shift-up",
    "control matrix W_CTRL + bias",
    "node-read slot extraction (E_p)", "node-read sel mux",
    "node-write (write + erase)", "register-file per-register gating",
    "frame write routing", "walk-item write routing",
    "ctx slot write routing", "free-stack push/pop routing",
    "liveness OR (live bits)", "fail-path construction",
    "eta error-catch construction",
    "comparison bits (idx/depth/level)",
    "control derived-bit threshold layers",
]


def run():
    print("Still on the DT checker: baseline 1,338,752 weights, "
          "bar = lockstep 41/41 + verdicts 20/20 + 21/21", flush=True)
    ok, p, n, worst = baseline_lockstep()
    assert ok, "baseline lockstep failed — the bar does not hold unpatched"
    cov = instrumented_run()
    per_entry, floor, total = compute_never_read(cov)
    print_coverage(per_entry, floor, total)

    entry_results = {}
    try:
        with open("still-checker-dt-map.json") as f:
            saved = json.load(f)
        entry_results = saved.get("entry_results", {})
        print("resuming entry prunes: %d done" % len(entry_results),
              flush=True)
    except FileNotFoundError:
        pass
    for name in ENTRY_ORDER:
        if name in entry_results:
            print("  skip %s: already pruned (%s)"
                  % (name, entry_results[name]["result"]), flush=True)
            continue
        res = prune_entry(name)
        entry_results[name] = res
        print("  prune %-38s %s (%s, %.0fs)%s"
              % (name, res["result"], res["stage"], res["seconds"],
                 (" — " + "; ".join("%s: %s" % f for f in res["failures"][:1])
                  if res["failures"] else "")),
              flush=True)
        with open("still-checker-dt-map.json", "w") as f:
            json.dump({"entry_results": entry_results,
                       "coverage": {k: v for k, v in per_entry.items()}},
                      f, indent=1, default=str)

    never_sets = stored_never_sets(cov)
    batch = batch_confirm(never_sets)
    print("batch confirm (all never-read stored weights zeroed): %s (%s)"
          % (batch["result"], batch["stage"]), flush=True)
    for f in batch["failures"][:3]:
        print("  FAIL " + str(f), flush=True)
    twins, saving = transpose_twins(never_sets)
    floor_shared = floor - saving
    print("transpose twins after batch prune: %s, sharing saves %d; "
          "floor (pruning + sharing) = %d"
          % (twins, saving, floor_shared), flush=True)

    out = {"baseline": total,
           "floor_pruning_only": floor,
           "floor_shared": floor_shared,
           "twins": twins, "sharing_saving": saving,
           "batch_confirm": batch,
           "entry_results": entry_results,
           "coverage": per_entry,
           "caveats": [
               "floor is against the 41-case bar, not a universal minimum",
               "comparison bits (512) is a bare literal; derived-bit "
               "layers (337) have a ~130 estimated component (w_gt); "
               "floor inherits +/-500 softness",
               "comparison bits and derived-bit layers appear to overlap "
               "(same threshold layers counted twice); baseline may be "
               "overstated by up to ~512",
               "analytic entries are cost models of inline code, not "
               "stored matrices; their floor share is coverage-measured, "
               "not code-pruned",
               "eta truncation gating (2,278) is a coarse analytic count "
               "(ctrl stack // 64)"]}
    with open("still-checker-dt-result.json", "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps({k: out[k] for k in
                      ("baseline", "floor_pruning_only", "floor_shared",
                       "twins", "sharing_saving")}, indent=1))
    return out


if __name__ == "__main__":
    run()
