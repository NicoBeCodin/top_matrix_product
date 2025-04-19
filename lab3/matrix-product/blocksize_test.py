import subprocess
import matplotlib.pyplot as plt
import os
import re
import numpy as np
import time

# Configurations
matrix_sizes = [(500, 500, 500), (1000, 1000, 1000), (2000, 2000, 2000)]
block_sizes = [
    (8, 8, 8), (8, 8, 16), (8, 8, 32),
    (16, 16, 16), (16, 16, 32),
    (32, 32, 32), (64, 64, 64)
]
fixed_threads = 8
num_runs = 3
perf_events = "L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses"

# Global output dir
base_plot_dir = "plots_blocksize_analysis"
os.makedirs(base_plot_dir, exist_ok=True)

# Set threading environment
env = os.environ.copy()
env["OMP_NUM_THREADS"] = str(fixed_threads)
env["OMP_PROC_BIND"] = "true"
env["OMP_PLACES"] = "cores"

def run_block_tests(matrix_size):
    m, n, k = matrix_size
    label = f"{m}x{n}x{k}"
    print(f"\n🔬 Testing Matrix Size: {label}\n")

    block_results = []

    for BM, BN, BK in block_sizes:
        runtimes = []
        gflops_values = []
        l1_miss_rates = []
        llc_miss_rates = []

        time.sleep(2.5)  # Let the system breathe

        for _ in range(num_runs):
            cmd = [
                "perf", "stat",
                "-e", perf_events,
                "./build/src/top.matrix_product",
                str(m), str(n), str(k),
                str(BM), str(BN), str(BK)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)

            # Parse output
            time_match = re.search(r"Runtime:\s+([0-9.]+)", result.stdout)
            perf_match = re.search(r"Performance:\s+([0-9.]+)", result.stdout)
            l1_loads = re.search(r"([\d,]+)\s+L1-dcache-loads", result.stderr)
            l1_misses = re.search(r"([\d,]+)\s+L1-dcache-load-misses", result.stderr)
            llc_loads = re.search(r"([\d,]+)\s+LLC-loads", result.stderr)
            llc_misses = re.search(r"([\d,]+)\s+LLC-load-misses", result.stderr)

            if time_match and perf_match and all([l1_loads, l1_misses, llc_loads, llc_misses]):
                runtime = float(time_match.group(1))
                gflops = float(perf_match.group(1))
                runtimes.append(runtime)
                gflops_values.append(gflops)

                l1_total = int(l1_loads.group(1).replace(",", ""))
                l1_miss = int(l1_misses.group(1).replace(",", ""))
                llc_total = int(llc_loads.group(1).replace(",", ""))
                llc_miss = int(llc_misses.group(1).replace(",", ""))

                l1_rate = l1_miss / l1_total if l1_total else 0
                llc_rate = llc_miss / llc_total if llc_total else 0

                l1_miss_rates.append(l1_rate)
                llc_miss_rates.append(llc_rate)

        if runtimes:
            block_label = f"{BM}x{BN}x{BK}"
            result = {
                "block": block_label,
                "time": np.mean(runtimes),
                "gflops": np.mean(gflops_values),
                "l1_miss": np.mean(l1_miss_rates),
                "llc_miss": np.mean(llc_miss_rates)
            }
            block_results.append(result)
            print(f"🧱 Block {block_label}: "
                  f"Time={result['time']:.4f}s | GFLOP/s={result['gflops']:.2f} | "
                  f"L1 Miss={result['l1_miss']*100:.2f}% | LLC Miss={result['llc_miss']*100:.2f}%")

    return label, block_results


def plot_block_results(label, block_results):
    # Unpack
    blocks = [r["block"] for r in block_results]
    times = [r["time"] for r in block_results]
    gflops = [r["gflops"] for r in block_results]
    l1 = [r["l1_miss"] for r in block_results]
    llc = [r["llc_miss"] for r in block_results]

    # Save location
    plot_dir = os.path.join(base_plot_dir, label)
    os.makedirs(plot_dir, exist_ok=True)

    # Runtime
    plt.figure()
    plt.bar(blocks, times, color='skyblue')
    plt.xlabel("Block Size")
    plt.ylabel("Runtime (s)")
    plt.title(f"Runtime vs Block Size ({label})")
    plt.ylim(0, max(times) * 1.2)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "runtime_vs_blocksize.png"))

    # GFLOP/s
    plt.figure()
    plt.bar(blocks, gflops, color='lightgreen')
    plt.xlabel("Block Size")
    plt.ylabel("Performance (GFLOP/s)")
    plt.title(f"GFLOP/s vs Block Size ({label})")
    plt.ylim(0, max(gflops) * 1.2)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "gflops_vs_blocksize.png"))

    # Cache Miss Rates
    plt.figure()
    x = np.arange(len(blocks))
    width = 0.35
    plt.bar(x - width/2, l1, width, label="L1 Miss Rate", color='coral')
    plt.bar(x + width/2, llc, width, label="LLC Miss Rate", color='salmon')
    plt.xticks(x, blocks)
    plt.xlabel("Block Size")
    plt.ylabel("Miss Rate")
    plt.title(f"Cache Miss Rate vs Block Size ({label})")
    plt.ylim(0, max(l1 + llc) * 1.2)
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "cache_miss_vs_blocksize.png"))


# === Run All Matrix Sizes ===
for matrix_size in matrix_sizes:
    label, block_results = run_block_tests(matrix_size)
    plot_block_results(label, block_results)

print(f"\n✅ All matrix size and block size plots saved in '{base_plot_dir}'")
