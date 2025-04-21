#!/usr/bin/env python3
import subprocess
import matplotlib.pyplot as plt
import os
import re
import numpy as np
import time

# ------------------------------
# Parameters (Final Study)
# ------------------------------
matrix_sizes = [(500, 500, 500), (1000, 1000, 1000), (2000, 2000, 2000)]
threads_list = [1, 2, 4, 8]
num_runs = 5
block_config = (32, 32, 32)  # Best performing block configuration from earlier tests
executable = "./build/src/top.matrix_product"  # Use your chosen executable

# Output directory for final study plots
plot_dir = "final_plots"
os.makedirs(plot_dir, exist_ok=True)

print("Block config : ", block_config)

# ------------------------------
# Run the strong scaling study for each matrix size
# ------------------------------
results = {}  # Dictionary keyed by matrix size label (e.g., "500x500x500")

for m, n, k in matrix_sizes:
    size_label = f"{m}x{n}x{k}"
    print(f"\n🔬 Testing matrix size: {size_label}")
    results[size_label] = {}  # Will hold results for each thread count

    for threads in threads_list:
        # Set the environment for OpenMP
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(threads)
        env["OMP_PROC_BIND"] = "true"
        env["OMP_PLACES"] = "cores"

        runtimes = []
        gflops_list = []
        
        # Sleep briefly between configurations to avoid resource contention.
        time.sleep(2.5)
        
        for _ in range(num_runs):
            cmd = [
                executable,
                str(m), str(n), str(k),
                str(block_config[0]), str(block_config[1]), str(block_config[2])
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)

            # Parse runtime and performance (GFLOP/s) from stdout.
            runtime_match = re.search(r"Runtime:\s+([0-9.]+)", result.stdout)
            perf_match = re.search(r"Performance:\s+([0-9.]+)", result.stdout)

            if runtime_match and perf_match:
                runtime_val = float(runtime_match.group(1))
                gflops_val = float(perf_match.group(1))
                runtimes.append(runtime_val)
                gflops_list.append(gflops_val)
            else:
                print(f"⚠️ Failed to parse output for matrix size {size_label} with {threads} threads.")
                print(result.stdout)
        
        if runtimes:
            avg_runtime = np.mean(runtimes)
            std_runtime = np.std(runtimes)
            avg_gflops = np.mean(gflops_list)
            std_gflops = np.std(gflops_list)
            results[size_label][threads] = {
                "runtime": avg_runtime,
                "std_runtime": std_runtime,
                "gflops": avg_gflops,
                "std_gflops": std_gflops
            }
            print(f"{threads} threads: Runtime = {avg_runtime:.4f}s ± {std_runtime:.4f}, "
                  f"GFLOP/s = {avg_gflops:.2f} ± {std_gflops:.2f}")

# ------------------------------
# Prepare data for plotting
# ------------------------------
# For each matrix size, build lists of thread counts, runtime, GFLOP/s, and compute speedup (relative to 1 thread)
runtime_data = {}
gflops_data = {}
speedup_data = {}

for size_label, data in results.items():
    thr = sorted(data.keys())
    runtime_vals = [data[t]["runtime"] for t in thr]
    gflops_vals = [data[t]["gflops"] for t in thr]
    # Compute speedup relative to single-thread runtime for this matrix size
    base_time = runtime_vals[0]
    speedup_vals = [base_time / rt for rt in runtime_vals]
    runtime_data[size_label] = (thr, runtime_vals)
    gflops_data[size_label] = (thr, gflops_vals)
    speedup_data[size_label] = (thr, speedup_vals)

# ------------------------------
# Combined Plots: One plot per metric with all matrix sizes
# ------------------------------

# (a) Runtime vs Threads
plt.figure()
for size_label, (thr, runtimes) in runtime_data.items():
    plt.errorbar(thr, runtimes, fmt='-o', label=size_label)
plt.xlabel("Number of Threads")
ideal_runtime = [runtimes[0] / t for t in threads_list]
plt.plot(threads_list, ideal_runtime, '--', label="Ideal Scaling (Runtime)")
plt.ylabel("Runtime (s)")
plt.title("Final Strong Scaling: Runtime vs Threads")
plt.xticks(threads_list)
plt.ylim(0, None)
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(plot_dir, "final_runtime_vs_threads.png"))

# (b) GFLOP/s vs Threads
plt.figure()
for size_label, (thr, gflops_vals) in gflops_data.items():
    plt.errorbar(thr, gflops_vals, fmt='-o', label=size_label)
plt.xlabel("Number of Threads")
ideal_gflops = [gflops_vals[0] * t for t in threads_list]
plt.plot(threads_list, ideal_gflops, '--', label="Ideal Scaling (GFLOP/s)")
plt.ylabel("Performance (GFLOP/s)")
plt.title("Final Strong Scaling: GFLOP/s vs Threads")
plt.xticks(threads_list)
plt.ylim(0, None)
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(plot_dir, "final_gflops_vs_threads.png"))

# (c) Speedup vs Threads (log-log scale)
plt.figure()
for size_label, (thr, speedup_vals) in speedup_data.items():
    plt.plot(thr, speedup_vals, '-o', label=size_label)
# Ideal speedup: for each thread count, ideal speedup equals the number of threads.
ideal_speedup = threads_list[:]  # e.g., [1, 2, 4, 8]
plt.plot(threads_list, ideal_speedup, 'k--', label="Ideal Scaling")

plt.xlabel("Number of Threads")
plt.ylabel("Speedup")
# Ideal Speedup Line
plt.plot(threads_list, threads_list, '--', label="Ideal Speedup")
plt.title("Final Strong Scaling: Speedup vs Threads")
plt.xscale('log', base=2)
plt.yscale('log', base=2)
plt.xticks(threads_list, threads_list)
plt.grid(True, which="both", ls="--")
plt.legend()
plt.savefig(os.path.join(plot_dir, "final_speedup_vs_threads.png"))

print(f"\n✅ Final performance and strong scaling plots saved in '{plot_dir}'")
