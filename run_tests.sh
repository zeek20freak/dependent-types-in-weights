#!/bin/sh
# Rung 1 + rung 2 stranger test: every suite, in dependency order.
# compare/battery/hostile rewrite the *.json receipts in place;
# the committed copies are the sealed 2026-11-25 receipts.
# The final step asserts the merged receipts are actually sealed —
# a run that prints PASSED while sealed is False is a failure.
set -e
cd "$(dirname "$0")"
python3 test_kernel_dt.py
python3 test_kernel_dt_nat.py
python3 test_kernel_dt_independent.py
python3 compare_dt.py
python3 fuzz_nat.py
python3 hostile_dt.py
python3 battery_dt.py
python3 merge_seal.py
python3 examples_dt.py
python3 test_repl.py
# Weights kernel (rung 3): RefMachine, substitution circuit, pool
# invariant, MatrixMachine lockstep. The Still floor is a measured
# receipt, not re-run here — asserted from still-checker-dt-result.json
# below. The independent-check harness script was lost to /tmp
# cleanup; its numbers and hashes are recorded in
# MATRIX_MACHINE_STATUS.md, not re-runnable from this package.
python3 weights_kernel_dt.py
python3 test_subst_circuit.py
python3 check_pool_invariant.py
python3 matrix_machine_dt.py
python3 - <<'EOF'
import json, sys
seal = json.load(open("sealing-pass-result.json"))
checks = [
    ("sealed", seal["seal_verdict"]["sealed"] is True),
    ("no live disagreements", seal["disagreements"] == []),
    ("comparison 43/43", seal["comparison"]["verdict_agreement"] == "43/43"),
    ("failure paths 43/43", seal["comparison"]["failure_path_agreement"] == "43/43"),
    ("battery capture 5/5", seal["adversarial_battery"]["capture"]["verdict_and_path_agreement"] == "5/5"),
    ("hostile 10/10", seal["hostile_read"]["agreements"] == 10),
    ("fuzz 500/500", seal["adversarial_battery"]["fuzz"]["verdict_agreement"] == 500),
]
bad = [name for name, ok in checks if not ok]
if bad:
    print("SEAL CHECK FAILED (rung 1):", bad)
    sys.exit(1)
print("seal check passed (rung 1):", ", ".join(name for name, _ in checks))
seal2 = json.load(open("sealing-pass-result-rung2.json"))
checks2 = [
    ("rung-2 sealed", seal2["seal_verdict"]["sealed"] is True),
    ("rung-2 no disagreements",
     seal2["comparison"]["disagreements"] == []),
    ("rung-2 comparison 67/67",
     seal2["comparison"]["verdict_agreement"] == "67/67"),
    ("rung-2 failure paths 67/67",
     seal2["comparison"]["failure_path_agreement"] == "67/67"),
    ("rung-2 fuzz verdict 300/300",
     seal2["seal_verdict"]["fuzz_nat"]["verdict_agreement"] == "300/300"),
]
bad2 = [name for name, ok in checks2 if not ok]
if bad2:
    print("SEAL CHECK FAILED (rung 2):", bad2)
    sys.exit(1)
print("seal check passed (rung 2):", ", ".join(name for name, _ in checks2))
still = json.load(open("still-checker-dt-result.json"))
checks3 = [
    ("still baseline 1338752", still["baseline"] == 1338752),
    ("still floor pruning-only 49196",
     still["floor_pruning_only"] == 49196),
    ("still floor shared 47687", still["floor_shared"] == 47687),
    ("still batch confirm held",
     still["batch_confirm"]["result"] == "hold"),
    ("still sharing saving 1509", still["sharing_saving"] == 1509),
    ("still 26 entry results", len(still["entry_results"]) == 26),
]
bad3 = [name for name, ok in checks3 if not ok]
if bad3:
    print("STILL CHECK FAILED (weights kernel):", bad3)
    sys.exit(1)
print("still check passed (weights kernel):",
      ", ".join(name for name, _ in checks3))
EOF
echo "ALL RUNG-1 + RUNG-2 + WEIGHTS-KERNEL CHECKS PASSED"
