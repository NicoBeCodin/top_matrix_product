import subprocess
import matplotlib.pyplot as plt
import os
import re
import numpy as np
import time

# Parameters
matrix_size = (1000, 1000, 1000)
threads_list = [1, 2, 4, 8]
num_runs = 5

executables = {
    "left": "./build/src/top.matrix_product",
    "right": "./build/src/top.matrix_product_right"
}

# Output directory
plot_dir = "plots_strong_scaling"
os.makedirs(plot_dir, exist_ok=True)

# Store results by layout
results = {"left": {}, "right": {}}

print("🔁 Running strong scaling tests for both layouts...\n")

for layout, exec_path in executables.items():
    print(f"▶️ Layout: {layout.upper()}")

    for threads in threads_list:
        time.sleep(2.5)
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(threads)
        env["OMP_PROC_BIND"] = "true"
        env["OMP_PLACES"] = "cores"
    

        runtimes = []
        gflops_values = []

        for _ in range(num_runs):
            cmd = [exec_path, *map(str, matrix_size)]
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)

            time_match = re.search(r"Runtime:\s+([0-9.]+)", result.stdout)
            perf_match = re.search(r"Performance:\s+([0-9.]+)", result.stdout)

            if time_match and perf_match:
                runtime_val = float(time_match.group(1))
                gflops = float(perf_match.group(1))
                runtimes.append(runtime_val)
                gflops_values.append(gflops)
            else:
                print(f"⚠️ Failed to parse output for {threads} threads ({layout}):")
                print(result.stdout)

        if runtimes:
            results[layout][threads] = {
                "time_mean": np.mean(runtimes),
                "time_std": np.std(runtimes),
                "gflops_mean": np.mean(gflops_values),
                "gflops_std": np.std(gflops_values),
            }
            print(f"{threads} threads: {np.mean(runtimes):.4f}s ± {np.std(runtimes):.4f}, "
                  f"{np.mean(gflops_values):.2f} GFLOP/s ± {np.std(gflops_values):.2f}")

print("\n📊 Generating comparison plots...")

# Extract data for both layouts
def get_data(key):
    return (
        [results["left"][t][f"{key}_mean"] for t in threads_list],
        [results["left"][t][f"{key}_std"] for t in threads_list],
        [results["right"][t][f"{key}_mean"] for t in threads_list],
        [results["right"][t][f"{key}_std"] for t in threads_list],
    )

# === Runtime Plot ===
left_times, left_stds, right_times, right_stds = get_data("time")

plt.figure()
plt.errorbar(threads_list, left_times, yerr=left_stds, label="LayoutLeft", fmt='-o', capsize=5)
plt.errorbar(threads_list, right_times, yerr=right_stds, label="LayoutRight", fmt='-s', capsize=5)
# Ideal Runtime Line
ideal_runtime = [left_times[0] / t for t in threads_list]
plt.plot(threads_list, ideal_runtime, '--', label="Ideal Scaling (Runtime)")
plt.xlabel("Number of Threads")
plt.ylabel("Runtime (s)")
plt.title("Strong Scaling: Runtime vs Threads")
plt.ylim(0, max(max(left_times), max(right_times)) * 1.2)
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(plot_dir, "runtime_comparison.png"))

# === GFLOP/s Plot ===
left_gflops, left_gflops_std, right_gflops, right_gflops_std = get_data("gflops")

plt.figure()
plt.errorbar(threads_list, left_gflops, yerr=left_gflops_std, label="LayoutLeft", fmt='-o', capsize=5)
plt.errorbar(threads_list, right_gflops, yerr=right_gflops_std, label="LayoutRight", fmt='-s', capsize=5)
# Ideal GFLOP/s Line
ideal_gflops = [left_gflops[0] * t for t in threads_list]
plt.plot(threads_list, ideal_gflops, '--', label="Ideal Scaling (GFLOP/s)")

plt.xlabel("Number of Threads")
plt.ylabel("Performance (GFLOP/s)")
plt.title("Strong Scaling: GFLOP/s vs Threads")
plt.ylim(0, max(max(left_gflops), max(right_gflops)) * 1.2)
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(plot_dir, "gflops_comparison.png"))

# === Speedup Plot ===
left_speedup = [left_times[0] / t for t in left_times]
right_speedup = [right_times[0] / t for t in right_times]

plt.figure()
plt.plot(threads_list, left_speedup, '-o', label="LayoutLeft")
plt.plot(threads_list, right_speedup, '-s', label="LayoutRight")
# Ideal Speedup Line
plt.plot(threads_list, threads_list, '--', label="Ideal Speedup")
plt.xlabel("Number of Threads")
plt.ylabel("Speedup")
plt.xscale('log', base=2)
plt.yscale('log', base=2)
plt.title("Strong Scaling: Speedup Comparison")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.savefig(os.path.join(plot_dir, "speedup_comparison.png"))

print(f"\n✅ Plots saved in '{plot_dir}':")
print(" - runtime_comparison.png")
print(" - gflops_comparison.png")
print(" - speedup_comparison.png")
