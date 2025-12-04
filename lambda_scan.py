import os
import numpy as np
from scanners import ParameterScanner
from topologies import get_topology


def main():
    # Parameters
    N = 4
    alpha = 0.8
    lambda_vals = np.array([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])

    # Graph / topology
    topology = get_topology("path", N)
    edges, probes = topology.edges, topology.probes

    scanner = ParameterScanner()
    output_dir = "outputs/vacuum_lambda_run"
    run_tag = "vacuum_lambda_run"

    df = scanner.scan_2d_phase_space(
        N=N,
        lambda_vals=lambda_vals,
        alpha_vals=np.array([alpha]),
        output_dir=output_dir,
        run_tag=run_tag,
        edges=edges,
        probes=probes,
    )

    # Extract the 1D slice at this alpha and drop the redundant column
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(
        output_dir,
        f"scan_N{N}_alpha{alpha:.2f}_vacuum_lambda.csv",
    )
    df[df["alpha"] == alpha].drop(columns=["alpha"]).to_csv(csv_path, index=False)

    print(df)
    print(f"Saved λ-scan slice for α={alpha} to {csv_path}")


if __name__ == "__main__":
    main()
