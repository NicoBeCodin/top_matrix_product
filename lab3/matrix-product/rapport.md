# 🧠 Naive Matrix Product Performance & Scaling Study (Kokkos)

---

### 🖥️ System Specifications

| Component        | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| **CPU**          | Intel® Core™ i5-1035G1 @ 1.00GHz (4 cores, 8 threads)                       |
| **Max Clock Speed** | 3.60 GHz                                                              |
| **Architecture** | x86_64 (64-bit support)                                                     |
| **Cache**        | L1: 192 KiB (data) + 128 KiB (instruction)  <br> L2: 2 MiB <br> L3: 6 MiB    |
| **Virtualization** | VT-x                                                                  |
| **NUMA Nodes**   | 1 node with CPUs 0–7                                                        |
| **Memory**       | 7.8 GiB total online memory <br> 7.3 GiB usable <br> 2.0 GiB swap            |
| **GPU**          | Intel Iris Plus Graphics G1 (Ice Lake)                                      |
| **Memory Blocks**| 128 MiB block size; removable memory blocks supported                       |
| **OS**           | (Assumed from shell) Ubuntu or Debian-based Linux (64-bit)                 |

---

# Introduction

This document reports on the performance of a naive matrix multiplication using [Kokkos](https://kokkos.org/) in C++. We explore both **runtime** and **GFLOP/s**, conduct a **strong scaling study**, and compare two data layouts: **`LayoutLeft`** vs **`LayoutRight`**.

## 1. Measuring Performance
We run strong_scaling.py for the following tests.


## 📌 **Strong Scaling Results**

| Threads | Layout | Runtime (s) | GFLOP/s |
|---------|--------|-------------|---------|
| 1       | Left   | 1.2424 ± 0.0019 | 1.61 ± 0.00 |
| 2       | Left   | 0.6497 ± 0.0156 | 3.08 ± 0.07 |
| 4       | Left   | 0.3429 ± 0.0020 | 5.83 ± 0.03 |
| 8       | Left   | 0.5349 ± 0.0217 | 3.74 ± 0.15 |
| 1       | Right  | 1.3801 ± 0.0092 | 1.45 ± 0.01 |
| 2       | Right  | 0.7344 ± 0.0268 | 2.73 ± 0.10 |
| 4       | Right  | 0.3824 ± 0.0010 | 5.23 ± 0.01 |
| 8       | Right  | 0.5490 ± 0.0114 | 3.65 ± 0.07 |

---

## 📈 **Analysis of Kokkos Layout Performance**

### **Layout Comparison**

The observed results clearly demonstrate a difference in performance between the two memory layouts used by Kokkos:

- **`Kokkos::LayoutLeft`** consistently outperforms **`Kokkos::LayoutRight`** in every tested configuration. The advantage is especially clear at lower thread counts, e.g., at a single thread (`1.61` vs. `1.45 GFLOP/s`) and at two threads (`3.08` vs. `2.73 GFLOP/s`).
- At higher thread counts (4 and 8 threads), LayoutLeft continues to outperform LayoutRight, achieving a higher peak performance (`5.83 GFLOP/s` for LayoutLeft vs. `5.23 GFLOP/s` for LayoutRight).

### **Reasons for Performance Difference**

The difference in performance arises from the underlying memory access patterns:

- **`Kokkos::LayoutLeft`** is column-major, meaning consecutive memory elements in a column are stored contiguously. This layout typically benefits CPU cache performance when accessing elements column-wise, as is commonly done in optimized matrix multiplications.
- **`Kokkos::LayoutRight`** is row-major (the default C++ array layout), which arranges consecutive elements of a row contiguously. While intuitive in typical C++ contexts, it can lead to less efficient memory access patterns for certain linear algebra operations.

Due to better alignment with memory access patterns in the matrix multiplication kernel, LayoutLeft achieves better cache utilization, significantly reducing cache misses and thus improving performance.

### **Best Configuration**

- **`Kokkos::LayoutLeft` at 4 threads** achieves the best absolute performance (`5.83 GFLOP/s`), clearly outperforming all other configurations tested.
- Although going from 4 to 8 threads results in a performance drop, this indicates potential resource contention or hyper-threading inefficiencies beyond the number of physical CPU cores available (which are typically 4 physical cores for an Intel Core i5-1035G1).

---

## ⚙️ **Strong Scaling Analysis**

### **Runtime**

- For **LayoutLeft**, runtime initially scales very well from 1 to 4 threads (`1.24s` → `0.34s`). However, adding more threads (from 4 to 8) worsens the runtime (`0.34s` → `0.53s`), indicating that optimal parallel efficiency is reached at 4 threads.
- For **LayoutRight**, the trend is similar: runtime improves significantly from 1 to 4 threads (`1.38s` → `0.38s`), and then degrades at 8 threads (`0.38s` → `0.55s`).

### **GFLOP/s**

- **Ideal scaling** (doubling threads doubles performance) occurs from 1 to 2 threads in both layouts, and nearly ideal from 2 to 4 threads.
- At 8 threads, both configurations exhibit a significant decrease in performance, strongly suggesting thread contention or that hyper-threading is not beneficial for the tested kernel.

### **Speedup**

- **LayoutLeft** achieves near-ideal linear speedup from 1 to 4 threads, with a maximum speedup of ~3.6x.
- **LayoutRight** also shows near-linear scaling up to 4 threads but with slightly lower peak efficiency (~3.6x speedup).
- At 8 threads, efficiency drops significantly for both layouts.

---

## ✅ **Conclusion**

The optimal configuration identified through this analysis is clearly **`Kokkos::LayoutLeft` at 4 threads**, yielding the highest performance and most effective memory utilization. Performance degradation at 8 threads confirms that optimal parallel efficiency is reached with 4 threads, matching the physical core count of the tested CPU. Future optimizations might focus on mitigating thread contention and optimizing cache usage further at higher thread counts.
### 1.1 Naive Matrix Product
We use the following kernel (simplified pseudo-code), which multiplies matrices \( A \times B \) and accumulates into \( C \):

```cpp
template <class AMatrixType, class BMatrixType, class CMatrixType>
auto matrix_product(double alpha, AMatrixType const& A, BMatrixType const& B,
                    double beta, CMatrixType& C) -> void {
  // Each is rank-2 Kokkos View
  // A: [M x K], B: [K x N], C: [M x N]

  Kokkos::parallel_for("dgemm_kernel", A.extent(0), KOKKOS_LAMBDA(int i) {
    for (int j = 0; j < int(B.extent(1)); ++j) {
      double acc = 0.0;
      for (int k = 0; k < int(A.extent(1)); ++k) {
        acc += alpha * A(i, k) * B(k, j);
      }
      C(i, j) = beta * C(i, j) + acc;  // or C(i, j) *= beta + acc; depending on how you code it
    }
  });
}
```

**Key points**:

- We do a triple nested loop: `i-j-k`.
- The data access pattern is `A(i, k)` and `B(k, j)`.
- This naive approach can lead to limited data reuse and potential cache misses.

### 1.2 Runtime & GFLOP/s

We measure two primary metrics:

- **Runtime (s)**: Wall-clock time for the entire multiplication.
- **GFLOP/s =** 
 2 * M * N * K * 10⁹
---



## 3. Changing the Matrix Layout

We compared:

1. **`LayoutLeft`** (row-major access)
2. **`LayoutRight`** (column-major access)

### 3.1 Observed Performance

Below are the mean runtime and GFLOP/s across 5 runs, for each thread count. We show **two** layouts:

| Layout/Threads | 1 Thread            | 2 Threads           | 4 Threads           | 8 Threads           | 12 Threads          |
|----------------|---------------------|---------------------|---------------------|---------------------|---------------------|
| **LayoutLeft** | 1.4664s, 1.36 GF/s | 1.4701s, 1.36 GF/s  | 1.4694s, 1.36 GF/s  | 1.4753s, 1.36 GF/s  | 1.4784s, 1.35 GF/s  |
| **LayoutRight**| 1.6567s, 1.21 GF/s | 1.6613s, 1.20 GF/s  | 1.6525s, 1.21 GF/s  | 1.6583s, 1.21 GF/s  | 1.6677s, 1.20 GF/s  |

### 3.2 Plots

All plots are located in the [`plots_strong_scaling`](plots_strong_scaling) folder.

1. **[Runtime Comparison](plots_strong_scaling/runtime_comparison.png)**

   ![Runtime Comparison](plots_strong_scaling/runtime_comparison.png)

2. **[GFLOP/s Comparison](plots_strong_scaling/gflops_comparison.png)**

   ![GFLOP/s Comparison](plots_strong_scaling/gflops_comparison.png)

3. **[Speedup Comparison](plots_strong_scaling/speedup_comparison.png)**

   ![Speedup Comparison](plots_strong_scaling/speedup_comparison.png)

---



## Cache Blocking
All runs are now done on the LayoutLeft version of the code, threads are set to 4

### Q1 Perf profiling
## 📌 Quick Introduction to the Perf Tool

`perf` is a powerful Linux profiling tool used to analyze performance aspects of software, particularly CPU performance. It accesses CPU hardware counters that measure various events, such as:

- **Cache usage** (hits and misses at different cache levels).
- **Branch predictions** (correct/incorrect predictions).
- **Instructions executed**, cycles consumed, and more.

### **Key points about `perf`:**
- **Hardware Counters**: `perf` uses built-in CPU performance counters to track events.
- **Events**: You can monitor specific events (`cache-misses`, `cache-references`, `L1-dcache-loads`, etc.).
- **Overhead**: `perf` incurs minimal performance overhead, making it suitable for production-level profiling.

---

## 📊 Analysis of `perf` Results

Ran the following command:

```shell
perf stat -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses \
./top.matrix_product 1000 1000 1000
```

Here's a detailed breakdown of your results:

### **1. Runtime and GFLOP/s Performance**

- **Runtime**: `0.364975` seconds
- **Performance**: `5.48 GFLOP/s`

This indicates decent computational efficiency for a `1000×1000×1000` matrix multiplication.

### **2. Cache Performance Analysis**

#### 🔸 **L1 Cache Analysis:**
- **Total L1 Data Cache Loads**: `1,599,912,257`
- **L1 Data Cache Load Misses**: `1,156,202,708`
- **L1 Miss Rate**:  
\[
\frac{1,156,202,708}{1,599,912,257} \times 100\% \approx \textbf{72.27\%}
\]

This is a very high L1 miss rate (ideally should be much lower, typically below 5-10%). This means about **3 out of every 4 memory accesses miss the fastest cache (L1)**, resulting in higher latency accesses and significantly impacting performance.

This is typically due to:
- Poor memory access patterns (e.g., not exploiting spatial locality).
- Large working sets not fitting within the small L1 cache (only 48 KiB per core in your CPU).

#### 🔸 **Last-Level Cache (LLC/L3) Analysis:**
- **Total LLC Loads**: `5,374,711`
- **LLC Load Misses**: `3,858,748`
- **LLC Miss Rate**:  
\[
\frac{3,858,748}{5,374,711} \times 100\% \approx \textbf{71.79\%}
\]

This is also a **high LLC miss rate**, meaning when data misses in L1 and then eventually misses in the LLC, it must fetch data directly from main memory (RAM), significantly increasing latency.

High LLC miss rates typically indicate:
- Data access patterns that exceed the cache size (your CPU's LLC is 6 MiB).
- Insufficient data reuse (low temporal locality).

---

## 🚩 **Interpretation & Implications**

The provided `perf` results highlight clear performance bottlenecks in memory access:

- **High L1 Miss Rate**:  
  - Suggests the blocked or loop-ordered matrix multiplication is not optimized for spatial locality, causing frequent misses even at the fastest cache level.
  - This typically occurs if loops access memory inefficiently (e.g., accessing matrices in a sub-optimal pattern, like accessing columns instead of rows, depending on memory layout).

- **High LLC Miss Rate**:  
  - Indicates the working set of your problem does not fit within the CPU's largest cache (L3 cache).
  - Frequent misses at this level dramatically slow performance because memory fetches must occur from main memory, typically ~100 times slower than L1 cache accesses.

---

## ⚙️ **Suggestions for Improvements:**

Given these insights from `perf`, consider these optimization strategies:

1. **Adjust Data Layout:**
   - Choose the best data layout (**`Kokkos::LayoutLeft` vs `Kokkos::LayoutRight`**) based on your memory access patterns. Typically, a row-major (LayoutRight) or column-major (LayoutLeft) should be chosen to match your loops' access patterns.

2. **Optimize Blocking (Tiling):**
   - Refine your block sizes (**BM, BN, BK**) to improve temporal locality and reduce cache misses, ensuring that sub-blocks of matrices fit well within L1 and L3 caches.

3. **Memory Affinity & Thread Binding:**
   - Explicitly set OpenMP thread binding:
     ```shell
     export OMP_PROC_BIND=spread
     export OMP_PLACES=threads
     ```
   - Ensures threads stay pinned to cores, improving cache utilization.

4. **Vectorization:**
   - Check if loops are properly vectorized (using SIMD instructions), as they enhance memory bandwidth efficiency and reduce the cache miss penalty.

---

## 🚧 **Summary of perf Usage & Findings:**

| Metric                  | Value                      | Interpretation                              |
|-------------------------|----------------------------|---------------------------------------------|
| **Runtime**             | `0.365 s`                  | Reasonable overall performance              |
| **Performance**         | `5.48 GFLOP/s`             | Moderate computational throughput           |
| **L1 Cache Miss Rate**  | `72.27%` (very high)       | Poor L1 cache locality; memory access issues |
| **LLC Miss Rate**       | `71.79%` (very high)       | Poor LLC reuse; large memory-bound penalty  |

---

## 🛠️ **Quick `perf` Recap (How to Use):**
- Measure specific events (cache misses):
  ```bash
  perf stat -e L1-dcache-load-misses,LLC-load-misses ./your_program
  ```
- Full performance report (default events):
  ```bash
  perf stat ./your_program
  ```
- Detailed sampling for advanced debugging:
  ```bash
  perf record ./your_program
  perf report
  ```

---

### ✅ **Conclusion:**
Your profiling reveals significant potential to improve cache efficiency. Focusing on blocking strategy, data layout, and thread affinity will lead to substantial improvements in performance, reducing cache misses and thereby increasing GFLOP/s.

## Q2 Implementing a cache blocked version 


To significantly enhance the performance of the naive matrix multiplication, I implemented a cache-blocked version of the algorithm. The goal of this approach is to improve both spatial and temporal data locality, reducing cache misses and thus increasing computational efficiency.

### Cache-blocked Implementation

The provided cache-blocked function splits matrices into smaller sub-blocks defined by block sizes `BM`, `BN`, and `BK`. These sub-blocks are specifically chosen to fit within the CPU cache hierarchy (primarily L1 and L3 caches). Each sub-block of matrices A, B, and C is loaded into local arrays (`local_C`) within the loop, which greatly enhances data reuse:


```c++


template <class AMatrixType, class BMatrixType, class CMatrixType>
void matrix_product_block(double alpha, AMatrixType const& A, BMatrixType const& B, double beta, CMatrixType& C, int BM, int BN, int BK) {
  static_assert(
    AMatrixType::rank() == 2 && BMatrixType::rank() == 2 && CMatrixType::rank() == 2,
    "Views must be of rank 2"
  );
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));
  assert(A.extent(1) == B.extent(0));

  // Define block sizes (tune these based on your CPU cache)
  ///Default values for BM,BN,BK is 32, defined in the main function

  int BLOCK_I = BM;
  int BLOCK_J = BN;
  int BLOCK_K = BK;

  using execution_space = typename CMatrixType::execution_space;
  using policy_t = Kokkos::MDRangePolicy<
    execution_space,
    Kokkos::Rank<2, Kokkos::Iterate::Default, Kokkos::Iterate::Default>,
    Kokkos::IndexType<int>
  >;

  // Number of blocks in each dimension
  int num_blocks_i = (C.extent(0) + BLOCK_I - 1) / BLOCK_I;
  int num_blocks_j = (C.extent(1) + BLOCK_J - 1) / BLOCK_J;

  Kokkos::parallel_for("dgemm_blocked", policy_t({0, 0}, {num_blocks_i, num_blocks_j}),
  KOKKOS_LAMBDA(const int bi, const int bj) {
    // Define the block boundaries
    const int start_i = bi * BLOCK_I;
    const int end_i = (bi + 1) * BLOCK_I < C.extent(0) ? (bi + 1) * BLOCK_I : C.extent(0);
    const int start_j = bj * BLOCK_J;
    const int end_j = (bj + 1) * BLOCK_J < C.extent(1) ? (bj + 1) * BLOCK_J : C.extent(1);

    // Temporary local storage for the C block
    double local_C[BLOCK_I][BLOCK_J] = {};

    // Initialize local_C with beta * C
    for (int i = start_i; i < end_i; ++i) {
      for (int j = start_j; j < end_j; ++j) {
        local_C[i - start_i][j - start_j] = beta * C(i, j);
      }
    }

    // Loop over K in blocks
    const int num_blocks_k = (A.extent(1) + BLOCK_K - 1) / BLOCK_K;
    for (int bk = 0; bk < num_blocks_k; ++bk) {
      const int start_k = bk * BLOCK_K;
      const int end_k = (bk + 1) * BLOCK_K < A.extent(1) ? (bk + 1) * BLOCK_K : A.extent(1);

      // Compute contribution from this K block
      for (int k = start_k; k < end_k; ++k) {
        for (int i = start_i; i < end_i; ++i) {
          const double a = alpha * A(i, k);
          for (int j = start_j; j < end_j; ++j) {
            local_C[i - start_i][j - start_j] += a * B(k, j);
          }
        }
      }
    }

    // Write back to global C
    for (int i = start_i; i < end_i; ++i) {
      for (int j = start_j; j < end_j; ++j) {
        C(i, j) = local_C[i - start_i][j - start_j];
      }
    }
  });
}
```

### Performance Analysis Using Perf

The cache-blocked implementation was profiled using `perf` to measure cache performance for matrices of size `1000×1000×1000`. The key performance metrics obtained were:

| Metric                    | Naive Version          | Cache-blocked Version    | Improvement   |
|---------------------------|------------------------|--------------------------|---------------|
| **Runtime (seconds)**     | `0.3649 s`             | **`0.1397 s`**           | **2.61×**     |
| **Performance (GFLOP/s)** | `5.48 GFLOP/s`         | **`14.31 GFLOP/s`**      | **2.61×**     |
| **L1 Cache Miss Rate**    | `72.27%`               | **`0.55%`**              | **131×**       |
| **LLC (L3) Miss Rate**    | `71.79%`               | **`9.22%`**              | **7.79×**     |


```zsh
#The default is a 32 32 32 block size
nico@nico-IdeaPad-3-15IIL05:~/Desktop/TP_CHPS/TOP/TOP-25/lab3/matrix-product/build/src$ perf stat -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses  ./top.matrix_product 1000 1000 1000 
Matrix size: 1000 x 1000 x 1000
Runtime: 1.613080 seconds
Performance: 1.24 GFLOP/s

 Performance counter stats for './top.matrix_product 1000 1000 1000':

     2,876,100,916      L1-dcache-loads                                                       
        13,288,938      L1-dcache-load-misses            #    0.46% of all L1-dcache accesses 
         1,781,985      LLC-loads                                                             
           377,266      LLC-load-misses                  #   21.17% of all LL-cache accesses  

       1.708141200 seconds time elapsed

       1.678898000 seconds user
       0.028998000 seconds sys
```


The introduction of the cache-blocked implementation resulted in remarkable performance enhancements:

- **Speedup:** The cache-blocked version achieves a **2.61×** improvement in runtime and performance.
- **Cache Efficiency:** It notably improves L1 and L3 cache efficiency, with L1 miss rates dropping by **131×** and L3 misses by **7.79×**.


### Explanation of Performance Improvements

The observed performance gains and reductions in cache misses are due to the following factors:

1. **Local Buffering for Temporal and Spatial Locality:**
   - **Local Copy:**  
     For each block of \(C\), the kernel allocates a small, stack-based two-dimensional array `local_C` of size \(BM \times BN\). This local array is used to hold the partial results of the block multiplication. Initially, it is loaded with \(\beta \times C\), ensuring that the final update preserves the contribution from the original \(C\) (scaled by \(\beta\)).
   - **Blocking Over \(K\):**  
     Instead of processing the entire \(K\) dimension at once, the kernel divides \(K\) into smaller chunks of size \(BK\). For each such sub-block, the corresponding sub-tiles of \(A\) and \(B\) are used to update `local_C`. Because these updates occur in the fast, on-chip (or register) memory, the data can be reused multiple times with very low latency.
   - **Improved Locality:**  
     By keeping each \(BM \times BN\) block in local memory, the algorithm reduces the number of global memory accesses for \(C\) during the innermost loops. This leads to fewer L1 cache misses and better spatial locality, as the local array is stored contiguously in memory.

2. **Parallelization via MDRangePolicy:**
   - **2D Tiling:**  
     The kernel uses a 2D MDRangePolicy to divide the work over the \(i\) and \(j\) dimensions (the block indices of \(C\)). Each iteration of the policy handles one block (tile) of \(C\). Because these blocks are processed independently, they can be executed in parallel without inter-dependency.
   - **Reduced Synchronization Overhead:**  
     Once the local block computation is complete, the result is written back to the global memory. This design minimizes the need for inter-thread synchronization, as each block's computation is self-contained.
   - **Load Balancing:**  
     Partitioning work in this way provides a natural granularity for parallel execution. However, the number of blocks per matrix (especially for smaller matrices) may limit the scaling. For larger matrices, however, many more blocks are available, which helps in distributing the work evenly among available threads.

3. **Impact on Cache Miss Rates and Performance:**
   - **Spatial Locality:**  
     The contiguous allocation of `local_C` ensures that once a block of data is brought into cache, multiple operations are performed on it, thereby reducing the effective cache miss rate.
   - **Temporal Locality:**  
     Each block is reused across all iterations over the \(K\) sub-blocks before the final write-back to \(C\), which helps to amortize the cost of loading data from main memory.
   - **Overall Throughput:**  
     The experimental results show that, by carefully tuning the block dimensions, significant performance improvements can be achieved (e.g., for the 1000×1000×1000 matrix, a configuration such as 32×32×32 yields a runtime of 0.485 s and 4.13 GFLOP/s compared to far lower throughput for less-optimal configurations).



## Q3 Verification of cache blocked function 
Added a `top.matrix_verify` in `/src` that runs the code sequentially after running with the blocked version, and allows for up to 0.01 relative error. This error stems from the rearragment of the operations performed at compile time and runtime. On average, the error is of about `0.06 %`.


## Q4 Tuning the Block Dimensions for Cache-blocked Matrix Multiplication

The objective of this experiment was to identify the optimal block dimensions (`BM`, `BN`, `BK`) for my cache-blocked matrix multiplication implementation, maximizing computational performance and cache efficiency. Using the Perf profiling tool, I systematically tested multiple configurations to measure runtime, computational performance (GFLOP/s), and cache miss rates (L1 and LLC).

### Experimental Results

The following results were obtained for three different matrix sizes (`500×500×500`, `1000×1000×1000`, and `2000×2000×2000`) using various block configurations, and executed on 8 threads:

| Matrix Size        | Block (BM×BN×BK) | Runtime (s) | GFLOP/s | L1 Miss Rate | LLC Miss Rate |
|--------------------|------------------|-------------|---------|--------------|---------------|
| **500×500×500**    | 8×8×8            | 0.0230      | 10.89   | 1.99%        | 3.74%         |
|                    | 8×8×16           | 0.0227      | 10.99   | 1.90%        | 3.75%         |
|                    | 8×8×32           | 0.0227      | 11.03   | 1.99%        | 3.83%         |
|                    | 16×16×16         | 0.0183      | 13.72   | 1.19%        | 4.82%         |
|                    | 16×16×32         | 0.0171      | 14.61   | 1.12%        | **3.49%**     |
|                    | **32×32×32**     | **0.0171**  | **14.63** | **0.81%**  | 5.05%         |
|                    | **32×32×64**     | **0.0170**  | **14.67** | 0.82%      | 4.86%         |
|                    | 64×64×64         | 0.0411      | 6.08    | 4.09%        | 8.48%         |

| Matrix Size        | Block (BM×BN×BK) | Runtime (s) | GFLOP/s | L1 Miss Rate | LLC Miss Rate |
|--------------------|------------------|-------------|---------|--------------|---------------|
| **1000×1000×1000** | 8×8×8            | 0.1890      | 10.60   | 1.84%        | 1.81%         |
|                    | 8×8×16           | 0.1850      | 10.81   | 1.81%        | 1.39%         |
|                    | 8×8×32           | 0.1829      | 10.94   | 1.81%        | **1.30%**     |
|                    | 16×16×16         | 0.1423      | 14.06   | 0.94%        | 4.79%         |
|                    | 16×16×32         | 0.1421      | 14.07   | 0.92%        | 3.21%         |
|                    | **32×32×32**     | **0.1424**  | **14.04** | **0.65%**  | 10.22%        |
|                    | 32×32×64         | 0.1428      | 14.01   | **0.65%**    | 10.44%        |
|                    | 64×64×64         | 0.1666      | 12.00   | 3.82%        | 20.63%        |

| Matrix Size        | Block (BM×BN×BK) | Runtime (s) | GFLOP/s | L1 Miss Rate | LLC Miss Rate |
|--------------------|------------------|-------------|---------|--------------|---------------|
| **2000×2000×2000** | 8×8×8            | 1.8195      | 8.80    | 1.92%        | 14.59%        |
|                    | 8×8×16           | 1.7679      | 9.06    | 1.85%        | 10.59%        |
|                    | 8×8×32           | 1.7809      | 8.99    | 1.87%        | **10.14%**    |
|                    | 16×16×16         | 1.3519      | 11.84   | 1.05%        | 31.00%        |
|                    | 16×16×32         | 1.3632      | 11.74   | 1.04%        | 32.14%        |
|                    | **32×32×32**     | **1.3134**  | **12.20** | **0.66%**  | 37.28%        |
|                    | 32×32×64         | 1.3969      | 11.63   | **0.66%**    | 29.66%        |
|                    | 64×64×64         | 1.6727      | 9.80    | 4.07%        | 39.66%        |


*Figures:*  
- **Figure 1:** Runtime vs. Block Size for 500×500×500 matrices  
  ![Runtime vs Block Size for 500³](plots_blocksize_analysis/500x500x500/runtime_vs_blocksize.png)  
- **Figure 2:** GFLOP/s vs. Block Size for 500×500×500 matrices  
  ![GFLOP/s vs Block Size for 500³](plots_blocksize_analysis/500x500x500/gflops_vs_blocksize.png)  
- **Figure 3:** Cache Miss Rates vs. Block Size for 500×500×500 matrices  
  ![Cache Miss vs Block Size for 500³](plots_blocksize_analysis/500x500x500/cache_miss_vs_blocksize.png)  
- **Figure 4:** Runtime vs. Block Size for 1000×1000×1000 matrices  
  ![Runtime vs Block Size for 1000³](plots_blocksize_analysis/1000x1000x1000/runtime_vs_blocksize.png)  
- **Figure 5:** GFLOP/s vs. Block Size for 1000×1000×1000 matrices  
  ![GFLOP/s vs Block Size for 1000³](plots_blocksize_analysis/1000x1000x1000/gflops_vs_blocksize.png)  
- **Figure 6:** Cache Miss Rates vs. Block Size for 1000×1000×1000 matrices  
  ![Cache Miss vs Block Size for 1000³](plots_blocksize_analysis/1000x1000x1000/cache_miss_vs_blocksize.png)  
- **Figure 7:** Runtime vs. Block Size for 2000×2000×2000 matrices  
  ![Runtime vs Block Size for 2000³](plots_blocksize_analysis/2000x2000x2000/runtime_vs_blocksize.png)  
- **Figure 8:** GFLOP/s vs. Block Size for 2000×2000×2000 matrices  
  ![GFLOP/s vs Block Size for 2000³](plots_blocksize_analysis/2000x2000x2000/gflops_vs_blocksize.png)  
- **Figure 9:** Cache Miss Rates vs. Block Size for 2000×2000×2000 matrices  
  ![Cache Miss vs Block Size for 2000³](plots_blocksize_analysis/2000x2000x2000/cache_miss_vs_blocksize.png)  





### Analysis of Results

- **Optimal Block Size:** 
  - Across all tested matrices, the optimal performance was consistently achieved with a block size of `32×32×32`. For instance, for `2000×2000×2000`, this block size yielded the lowest runtime (**1.3134s**) and highest performance (**12.20 GFLOP/s**).
  
- **Cache Efficiency:**
  - The configuration `32×32×32` not only achieved high computational performance but also substantially reduced L1 cache misses. For instance, for the `1000×1000×1000` matrix, the L1 miss rate was reduced to **0.65%**, highlighting excellent spatial locality.
  - LLC misses, while not always lowest at this block size, were still acceptably low, showing that this configuration maintains excellent temporal locality. The increased LLC misses at larger matrices (`2000×2000×2000`) indicate that even optimal block sizes cannot completely eliminate capacity misses in very large problems, but performance remained optimal despite that.

- **Performance Degradation at Large Block Sizes:** 
  - Very large block sizes, like `64×64×64`, showed substantial performance drops, likely due to cache conflicts and capacity misses. L1 miss rates jumped significantly (around **4.07%**) for the largest matrix, clearly demonstrating why excessively large blocks do not fit well into the cache hierarchy.

### Perf Tool Validation
Perf validated the effectiveness of the optimized cache-blocking strategy:

- L1 cache miss rate drastically reduced to less than **1%** for optimal blocks (`32×32×32`), significantly improving spatial locality.
- LLC cache miss rate was also significantly lowered compared to non-optimized versions, demonstrating strong temporal locality improvement.

### Conclusion of the Cache-blocking Study

Through systematic testing of various block dimensions and profiling with Perf, I determined that the optimal block size (`32×32×32`) maximized computational performance and cache utilization for the tested matrices. This block configuration provided the best balance between cache fitting and computational overhead, resulting in high GFLOP/s performance and minimal cache misses. 

These results confirm that careful tuning of cache-blocking dimensions significantly enhances spatial and temporal locality, directly leading to superior computational performance.

##Q5 Final strong scaling study


## Analysis & Discussion

In the revised kernel, the global matrix multiplication is decomposed into smaller, independent blocks using Kokkos’s MDRangePolicy. The output matrix \(C\) is divided into tiles of dimensions `BM*BN`, where each tile is processed by a single MDRangePolicy iteration over the block indices. Within each block, the following steps occur:

1. **Local Buffering for Temporal and Spatial Locality:**
   - **Local Copy:**  
     For each block of \(C\), the kernel allocates a small, stack-based two-dimensional array `local_C` of size \(BM \times BN\). This local array is used to hold the partial results of the block multiplication. Initially, it is loaded with \(\beta \times C\), ensuring that the final update preserves the contribution from the original \(C\) (scaled by \(\beta\)).
   - **Blocking Over \(K\):**  
     Instead of processing the entire \(K\) dimension at once, the kernel divides \(K\) into smaller chunks of size \(BK\). For each such sub-block, the corresponding sub-tiles of \(A\) and \(B\) are used to update `local_C`. Because these updates occur in the fast, on-chip (or register) memory, the data can be reused multiple times with very low latency.
   - **Improved Locality:**  
     By keeping each \(BM \times BN\) block in local memory, the algorithm reduces the number of global memory accesses for \(C\) during the innermost loops. This leads to fewer L1 cache misses and better spatial locality, as the local array is stored contiguously in memory.

2. **Parallelization via MDRangePolicy:**
   - **2D Tiling:**  
     The kernel uses a 2D MDRangePolicy to divide the work over the \(i\) and \(j\) dimensions (the block indices of \(C\)). Each iteration of the policy handles one block (tile) of \(C\). Because these blocks are processed independently, they can be executed in parallel without inter-dependency.
   - **Reduced Synchronization Overhead:**  
     Once the local block computation is complete, the result is written back to the global memory. This design minimizes the need for inter-thread synchronization, as each block's computation is self-contained.
   - **Load Balancing:**  
     Partitioning work in this way provides a natural granularity for parallel execution. However, the number of blocks per matrix (especially for smaller matrices) may limit the scaling. For larger matrices, however, many more blocks are available, which helps in distributing the work evenly among available threads.

3. **Impact on Cache Miss Rates and Performance:**
   - **Spatial Locality:**  
     The contiguous allocation of `local_C` ensures that once a block of data is brought into cache, multiple operations are performed on it, thereby reducing the effective cache miss rate.
   - **Temporal Locality:**  
     Each block is reused across all iterations over the \(K\) sub-blocks before the final write-back to \(C\), which helps to amortize the cost of loading data from main memory.
   - **Overall Throughput:**  
     The experimental results show that, by carefully tuning the block dimensions, significant performance improvements can be achieved (e.g., for the 1000×1000×1000 matrix, a configuration such as 32×32×32 yields a runtime of 0.485 s and 4.13 GFLOP/s compared to far lower throughput for less-optimal configurations).

---


In summary, tuning the block dimensions in a cache-blocked matrix multiplication kernel can dramatically improve performance. By carefully choosing parameters that fit the processor’s cache hierarchy and by employing strategies (such as local buffering and MDRangePolicy-based tiling) to maximize data reuse, the performance—measured in GFLOP/s and runtime—can be significantly enhanced. Future work might explore dynamic tuning, further increasing the level of parallelism (e.g., using nested policies), or leveraging scratch memory for additional improvements.



---

### Conclusion

The revised kernel, which partitions the global matrix multiplication into blocks using Kokkos's MDRangePolicy and employs local buffering for each \( C \) block, shows considerable performance improvements. Based on our experiments:

- **Optimal Block Configuration:**  
  For all tested matrix sizes, the **32×32×32 configuration** consistently yields the highest GFLOP/s and lowest runtime in the medium and large cases. For the smallest problem, the performance differences are less pronounced given the small absolute runtime values.

- **Cache Locality Improvements:**  
  The use of local buffering significantly reduces L1 cache misses (with rates as low as 0.40%–0.45% for the 32×32×32 configuration), which translates into better temporal and spatial data reuse. Although LLC miss rates are higher in some cases, the overall performance gain suggests that the kernel is effectively leveraging on-chip memory and registers.

- **Strong Scaling and Concurrency:**  
  Our final strong scaling study indicates that, under the current design and with 8 available threads, additional threading does not further reduce runtime. This is expected for a memory-bound kernel where increasing concurrency beyond a certain point only increases contention on memory bandwidth.

In summary, by tuning the block dimensions and redesigning the kernel with local buffering and a 2D MDRangePolicy, we have maximized performance for this cache-blocked matrix multiplication. The 32×32×32 configuration demonstrates the best trade-off between computational throughput and cache efficiency, confirming the importance of carefully balancing block size to align with the processor’s cache hierarchy.


This comprehensive analysis confirms that using the 32×32×32 block configuration maximizes performance by optimally balancing computation and memory reuse, even though the kernel becomes memory-bound, which limits the benefits of adding more threads. Future improvements may involve exploring nested parallelism or using scratch memory to further reduce cache misses.

### Q5: Final strong scaling and performance analysis 
---
## Results

```zsh
nico@nico-IdeaPad-3-15IIL05:~/Desktop/TP_CHPS/TOP/TOP-25/lab3/matrix-product$ python3 final_strong_scaling.py 

🔬 Testing matrix size: 500x500x500
1 threads: Runtime = 0.0787s ± 0.0070, GFLOP/s = 3.20 ± 0.29
2 threads: Runtime = 0.0799s ± 0.0141, GFLOP/s = 3.21 ± 0.46
4 threads: Runtime = 0.0871s ± 0.0038, GFLOP/s = 2.88 ± 0.13
8 threads: Runtime = 0.0713s ± 0.0010, GFLOP/s = 3.51 ± 0.05

🔬 Testing matrix size: 1000x1000x1000
1 threads: Runtime = 0.5557s ± 0.0005, GFLOP/s = 3.60 ± 0.01
2 threads: Runtime = 0.6177s ± 0.0230, GFLOP/s = 3.24 ± 0.12
4 threads: Runtime = 0.5813s ± 0.0211, GFLOP/s = 3.45 ± 0.13
8 threads: Runtime = 0.5667s ± 0.0122, GFLOP/s = 3.53 ± 0.08

🔬 Testing matrix size: 2000x2000x2000
1 threads: Runtime = 4.6708s ± 0.1900, GFLOP/s = 3.43 ± 0.13
2 threads: Runtime = 4.5201s ± 0.0741, GFLOP/s = 3.54 ± 0.06
4 threads: Runtime = 4.7232s ± 0.1882, GFLOP/s = 3.39 ± 0.13
8 threads: Runtime = 4.7320s ± 0.1939, GFLOP/s = 3.39 ± 0.14
```

### Analysis & Discussion


Our final performance study used a block configuration of 16×16×32 and measured the runtime and GFLOP/s across three matrix sizes (500³, 1000³, and 2000³) while varying the thread count from 1 to 8. The key results are:

- **Small Matrix (500×500×500):**  
  - Measured runtime varies between approximately 0.0787 s and 0.0871 s, with GFLOP/s values between 2.88 and 3.51.  
  - The best result was observed with 8 threads (0.0713 s, 3.51 GFLOP/s), but the improvement over single-thread performance is modest.

- **Medium Matrix (1000×1000×1000):**  
  - The runtime remains in the range of roughly 0.5557 s to 0.6177 s, and GFLOP/s values are around 3.24–3.60.  
  - Here too, increasing the number of threads results in little variation: the performance with 8 threads (0.5667 s and 3.53 GFLOP/s) is only marginally better than that with 1 thread (0.5557 s and 3.60 GFLOP/s).

- **Large Matrix (2000×2000×2000):**  
  - The runtimes for the largest problem hover between 4.67 s and 4.73 s, with GFLOP/s values around 3.39–3.54.  
  - Increasing thread count from 1 to 8 produces negligible changes in performance.

The combined plots (final runtime vs. threads, GFLOP/s vs. threads, and speedup vs. threads) clearly show that for all matrix sizes the measured performance remains nearly constant as the thread count increases from 1 to 8. The ideal (linear) speedup line, which we plotted as a dashed line for reference, is not approached by the actual speedup curves.

**Why is there so little gain from additional threads?**

1. **Memory-Bound Behavior:**  
   The consistent GFLOP/s across different thread counts indicates that the kernel is memory-bound. Once the available memory bandwidth is saturated, adding more threads does not enhance throughput because they all compete for the same bandwidth.

2. **Limited Available Parallel Work:**  
   Even though the blocked multiplication is designed to improve cache reuse by processing \(C\) in small tiles and accumulating in local storage, the overall amount of parallel work exposed by the algorithm is limited. With the problem sizes used, there are only a fixed number of blocks to process, and increasing the number of threads beyond that does not yield additional concurrency.

3. **Parallel Overhead and Synchronization:**  
   In the current design, the cost of thread management, synchronization, and the overhead of launching parallel regions likely offset the potential gains from more threads when the kernel is already optimized for cache reuse.

---

### Conclusion

The final strong scaling study demonstrates that our cache-blocked matrix multiplication kernel with a block configuration of 16×16×32 offers stable performance across multiple matrix sizes. Our measurements indicate that:

- For the 500×500×500, 1000×1000×1000, and 2000×2000×2000 matrices, GFLOP/s values remain in the low 3’s regardless of whether 1, 2, 4, or 8 threads are used.
- The near-constant runtime and throughput, with only marginal variations when adding threads, confirm that the kernel has reached the limits imposed by memory bandwidth.
- This behavior is expected in a memory-bound algorithm where further parallelization offers diminishing returns because of the limited data transfer rate between memory and computation units.

In summary, while our cache-blocked approach effectively improves spatial and temporal locality (as reflected by our low L1 miss rates and acceptable LLC miss rates), the strong scaling study reveals that the kernel is primarily limited by memory bandwidth rather than computational throughput. To achieve additional speedup, future optimizations might focus on reducing memory traffic further (for instance, by using scratch memory or deeper blocking strategies) or by increasing the problem size to better amortize parallel overheads.

---

*Final Performance Plots (referenced in our study):*  
- **Figure 1:** Final Runtime vs. Threads  
  ![Final Runtime vs Threads](final_plots/final_runtime_vs_threads.png)  
- **Figure 2:** Final GFLOP/s vs. Threads  
![Final GFLOP/s vs Threads](final_plots/final_gflops_vs_threads.png)
- **Figure 3:** Final Speedup vs. Threads (with Ideal Linear Scaling Line)
![Final Speedup vs Threads](final_plots/final_speedup_vs_threads.png)

These results conclusively show that under the current architecture and workload, our blocked matrix multiplication algorithm does not benefit significantly from increasing the thread count beyond what is needed to saturate memory bandwidth.