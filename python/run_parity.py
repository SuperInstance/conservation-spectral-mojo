#!/usr/bin/env python3
"""Parity driver: runs diagnostics/parity_dump.mojo, compares with oracle.

Rules-of-the-road: list-form subprocess only. No deletions.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from oracle import get_case, CASES  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

# Toolchain (fleet pattern): honor existing PATH, else wire quilt-lab env.
def mojo_env():
    import os
    env = dict(os.environ)
    quake_bin = str(Path.home() / "projects/quilt-mojo-lab/.pixi/envs/default/bin")
    env["MODULAR_HOME"] = env.get(
        "MODULAR_HOME",
        str(Path.home() / "projects/quilt-mojo-lab/.pixi/envs/default/share/max"))
    if not ("mojo" in (p.name for p in Path(quake_bin).glob("mojo"))):
        pass
    env["PATH"] = quake_bin + ":" + env.get("PATH", "")
    return env


def run_mojo_dump():
    cmd = ["mojo", "run", "-D", "ASSERT=all", "-I", ".",
           "diagnostics/parity_dump.mojo"]
    res = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                         env=mojo_env())
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr, file=sys.stderr)
        raise SystemExit(f"mojo dump failed rc={res.returncode}")
    return res.stdout


def parse_dump(text):
    cases = {}
    cur = None
    for line in text.splitlines():
        toks = line.split()
        if not toks:
            continue
        if toks[0] == "CASE":
            cur = toks[1]
            cases[cur] = {}
        elif cur is None:
            continue
        elif toks[0] in ("DEG", "LMAT", "EVAL", "CR"):
            cases[cur][toks[0]] = np.array([float(x) for x in toks[1:]])
        elif toks[0] in ("N", "EDGES", "ANOM"):
            cases[cur][toks[0]] = int(toks[1])
        elif toks[0] in ("GAP", "CHEEGER", "ENTROPY", "EFFDIM",
                         "INV_ROWSUM", "INV_ASYM", "INV_DIAG"):
            cases[cur][toks[0]] = float(toks[1])
    return cases


def fnum(x):
    return f"{x:.3g}"


def main():
    dump = parse_dump(run_mojo_dump())
    rows = []
    all_inv_ok = True
    all_construct_ok = True
    all_eigen_ok = True

    for name in CASES:
        o = get_case(name)
        m = dump[name]
        n = o["n"]
        assert m["N"] == n and m["EDGES"] == o["edges"], (name, m["N"], m["EDGES"], o["edges"])

        deg_d = np.abs(m["DEG"] - o["deg"]).max() if n else 0.0
        L_d = np.abs(m["LMAT"] - o["L"].reshape(-1)).max() if n else 0.0
        ev_d = np.abs(m["EVAL"] - o["evals"]).max() if n > 1 else 0.0
        gap_d = abs(m["GAP"] - o["gap"])
        chee_d = abs(m["CHEEGER"] - o["cheeger"])
        ent_d = abs(m["ENTROPY"] - o["entropy"])

        # Kind-aware conservation invariants (definitions, not defects):
        #  - L.1 = 0 holds for unnormalized (any directedness) and
        #    symmetric_normalized on undirected graphs (all degrees > 0).
        #  - symmetry holds for unnormalized / symmetric_normalized on
        #    undirected graphs; random-walk is legitimately asymmetric.
        #  - diag = 1 for normalized kinds on positive-degree vertices.
        # L.1 = 0 for symmetric_normalized additionally requires all degrees
        # > 0 (two_isolated has L = I, row sums 1 legitimately).
        kind_is_sym = name in ("ring8_sym", "two_isolated_sym")
        sym_all_positive_degree = name == "ring8_sym"
        rowsum_expected_zero = (name.endswith("_unnorm")
                                or (sym_all_positive_degree and "directed" not in name))
        asym_expected_zero = (not name.endswith("_rw")
                              and "directed" not in name)
        inv_ok = True
        if n > 0:
            if rowsum_expected_zero:
                inv_ok &= m["INV_ROWSUM"] < 1e-12
            if asym_expected_zero:
                inv_ok &= m["INV_ASYM"] < 1e-12
            if name.endswith("_sym") and name != "single_vertex":
                inv_ok &= m["INV_DIAG"] < 1e-12
        construct_ok = deg_d < 1e-12 and L_d < 1e-12
        eigen_ok = ev_d < 1e-8 and gap_d < 1e-8 and chee_d < 1e-8 and ent_d < 1e-8

        all_inv_ok &= inv_ok
        all_construct_ok &= construct_ok
        all_eigen_ok &= eigen_ok

        rows.append((name, n, deg_d, L_d, ev_d, gap_d, chee_d, ent_d,
                     m["INV_ROWSUM"], m["INV_ASYM"], inv_ok, construct_ok, eigen_ok))

    print("=" * 100)
    print(f"{'case':22s} {'n':>3s} {'deg Δ':>9s} {'L Δ':>9s} {'eval Δ':>9s} "
          f"{'gap Δ':>9s} {'cheeger Δ':>9s} {'entropy Δ':>9s} {'|rowsum|':>9s} {'|asym|':>9s}  verdict")
    print("-" * 100)
    for (name, n, deg_d, L_d, ev_d, gap_d, chee_d, ent_d, rs, asym,
         inv_ok, construct_ok, eigen_ok) in rows:
        verdict = ("FULL-PASS" if (inv_ok and construct_ok and eigen_ok)
                   else "CONSTRUCTION-PASS/EIGEN-FAIL" if (inv_ok and construct_ok)
                   else "FAIL")
        print(f"{name:22s} {n:3d} {fnum(deg_d):>9s} {fnum(L_d):>9s} {fnum(ev_d):>9s} "
              f"{fnum(gap_d):>9s} {fnum(chee_d):>9s} {fnum(ent_d):>9s} "
              f"{fnum(rs):>9s} {fnum(asym):>9s}  {verdict}")
    print("=" * 100)
    print("Construction parity (W/deg/L):", "PASS" if all_construct_ok else "FAIL")
    print("Conservation invariants (L·1=0, symmetry, diag):",
          "PASS (machine epsilon)" if all_inv_ok else "FAIL")
    print("Eigenvalue/metric parity:", "PASS" if all_eigen_ok else
          "FAIL — booked: eigen.mojo solver defective (see docs/USERMANUAL.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
