"""Measure TRUE max live registers in the DT RefMachine (complete
liveness: frames incl. saved dest / walk dest / smalls refs, ctx,
dest, res, focus fsel/asel, walk items) at every step boundary,
across all 41 suite cases. Replaces measure_live_regs.py, whose
scan missed fsel/asel and the smalls refs."""
import weights_kernel_dt as W


def run_all():
    check_cases, infer_cases, defeq_cases = W.build_cases()
    max_live = 0
    worst = None
    max_alloc = 0
    for name, term, ty, ctx, want in check_cases:
        term_db, ty_db, ctx_types = W.convert_case(term, ty, ctx)
        m = W.RefMachine(term_db, ty_db, ctx_types, entry="check")
        while m.mode not in ("HALT", "ERROR"):
            lv = len(m._live_regs())
            if lv > max_live:
                max_live, worst = lv, ("check", name, m.steps)
            m.step()
        lv = len(m._live_regs())
        if lv > max_live:
            max_live, worst = lv, ("check", name, m.steps)
        max_alloc = max(max_alloc, m.max_reg)
    for name, term, ctx, want in infer_cases:
        term_db, _, ctx_types = W.convert_case(term, None, ctx)
        m = W.RefMachine(term_db, None, ctx_types, entry="infer")
        while m.mode not in ("HALT", "ERROR"):
            lv = len(m._live_regs())
            if lv > max_live:
                max_live, worst = lv, ("infer", name, m.steps)
            m.step()
        lv = len(m._live_regs())
        if lv > max_live:
            max_live, worst = lv, ("infer", name, m.steps)
        max_alloc = max(max_alloc, m.max_reg)
    for name, ctx, a, b, want in defeq_cases:
        a_db, b_db, ctx_types = W.convert_defeq(a, b, ctx)
        m = W.RefMachine(a_db, b_db, ctx_types, entry="defeq")
        while m.mode not in ("HALT", "ERROR"):
            lv = len(m._live_regs())
            if lv > max_live:
                max_live, worst = lv, ("defeq", name, m.steps)
            m.step()
        lv = len(m._live_regs())
        if lv > max_live:
            max_live, worst = lv, ("defeq", name, m.steps)
        max_alloc = max(max_alloc, m.max_reg)
    print("RESULT max_live_complete=%d max_alloc=%d worst=%s"
          % (max_live, max_alloc, worst))
    return max_live, max_alloc, worst


if __name__ == "__main__":
    run_all()
