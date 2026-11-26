"""Debug validation for the bounded register file: after EVERY step
of all 41 suite cases, assert no freed register is referenced
anywhere in the machine state (check_refs). A failure would mean
_live_regs missed a source, or the sweep freed a live register.
Also records how many sweeps fired and the peak simultaneous
allocations, per case."""
import weights_kernel_dt as W


def run_all():
    check_cases, infer_cases, defeq_cases = W.build_cases()
    total_steps = 0
    total_sweeps = 0
    worst_live = 0
    worst_alloc = 0
    orig_sweep = W.RefMachine._sweep

    def counting_sweep(self):
        nonlocal total_sweeps
        total_sweeps += 1
        return orig_sweep(self)

    W.RefMachine._sweep = counting_sweep
    try:
        for name, term, ty, ctx, want in check_cases:
            term_db, ty_db, ctx_types = W.convert_case(term, ty, ctx)
            m = W.RefMachine(term_db, ty_db, ctx_types, entry="check")
            while m.mode not in ("HALT", "ERROR"):
                m.step()
                total_steps += 1
                live = m.check_refs()
                worst_live = max(worst_live, len(live))
                worst_alloc = max(worst_alloc, m.max_reg)
            m.check_refs()
        for name, term, ctx, want in infer_cases:
            term_db, _, ctx_types = W.convert_case(term, None, ctx)
            m = W.RefMachine(term_db, None, ctx_types, entry="infer")
            while m.mode not in ("HALT", "ERROR"):
                m.step()
                total_steps += 1
                live = m.check_refs()
                worst_live = max(worst_live, len(live))
                worst_alloc = max(worst_alloc, m.max_reg)
            m.check_refs()
        for name, ctx, a, b, want in defeq_cases:
            a_db, b_db, ctx_types = W.convert_defeq(a, b, ctx)
            m = W.RefMachine(a_db, b_db, ctx_types, entry="defeq")
            while m.mode not in ("HALT", "ERROR"):
                m.step()
                total_steps += 1
                live = m.check_refs()
                worst_live = max(worst_live, len(live))
                worst_alloc = max(worst_alloc, m.max_reg)
            m.check_refs()
    finally:
        W.RefMachine._sweep = orig_sweep
    print("INVARIANT OK: %d steps checked, %d sweeps, "
          "max_live=%d max_simultaneous_alloc=%d pool=%d"
          % (total_steps, total_sweeps, worst_live, worst_alloc,
             W.POOL_SIZE))


if __name__ == "__main__":
    run_all()
