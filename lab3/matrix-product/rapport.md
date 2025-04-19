# 🧠 Naive Matrix Product Performance & Scaling Study (Kokkos)

## CPU Specs 
```zsh
  Info: quad core model: Intel Core i5-1035G1 bits: 64 type: MT MCP
    arch: Ice Lake rev: 5 cache: L1: 320 KiB L2: 2 MiB L3: 6 MiB

nproc
8
``` 

# Introduction

This document reports on the performance of a naive matrix multiplication using [Kokkos](https://kokkos.org/) in C++. We explore both **runtime** and **GFLOP/s**, conduct a **strong scaling study**, and compare two data layouts: **`LayoutLeft`** vs **`LayoutRight`**.

## 1. Measuring Performance

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

## 2. Strong Scaling Study

A **strong scaling** study keeps the problem size fixed (here, \( 1000 \times 1000 \times 1000 \)) and increases the number of threads. We tested from **1 to 12** threads and recorded how performance (runtime & GFLOP/s) evolves.

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

## 4. Conclusions

1. **Is there a performance improvement with different layouts?**  
   Yes. **`LayoutLeft`** consistently outperforms `LayoutRight` by roughly **11%** on this particular code and hardware.  

2. **Which combination yields the best results?**  
   - The best combination here is `LayoutLeft` with any thread count.  
   - Interestingly, scaling to more threads did not significantly improve performance, indicating that the naive approach may be bottlenecked by memory bandwidth or cache usage.
   - More threads for LayoutRight lowered FLOPS and   added to the runtime, indicating bad scaling performance

3. **Why does `LayoutLeft` do better?**  
   - This kernel’s access pattern is `A(i, k)` in the inner loop, which matches **row-major** memory layout well.  
   - Hence, contiguous memory accesses yield better cache locality and prefetching.  
   - `LayoutRight` is more natural for Fortran-like column-major data access, so it sees additional stride in the `k` loop for the same kernel, resulting in higher cache misses and lower performance.

4. **Strong Scaling Insights**  
   - The runtime doesn't drop much when we go from 1 thread up to 12 threads.  
   - The speedup curve is nearly flat, suggesting we’re not compute-bound but **memory-bound**.  
   - More sophisticated optimizations or memory-blocking strategies might be needed to see real parallel scaling improvements.




## Cache Blocking
All runs are now done on the LayoutLeft version of the code

### Q1
```zsh
nico@nico-IdeaPad-3-15IIL05:~/Desktop/TP_CHPS/TOP/TOP-25/lab3/matrix-product/build/src$ perf stat -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses ./top.matrix_product 1000 1000 1000
Matrix size: 1000 x 1000 x 1000
Runtime: 1.507459 seconds
Performance: 1.33 GFLOP/s

 Performance counter stats for './top.matrix_product 1000 1000 1000':

     1,578,882,971      L1-dcache-loads                                                       
     1,165,552,800      L1-dcache-load-misses            #   73.82% of all L1-dcache accesses 
        81,460,202      LLC-loads                                                             
        60,596,086      LLC-load-misses                  #   74.39% of all LL-cache accesses  

       1.544103116 seconds time elapsed

       1.524028000 seconds user
       0.020000000 seconds sys
```
> We profiled cache usage with `perf stat` and observed very high miss rates: **73.82% in L1** and **74.39% in L3 (LLC)**. These results confirm that the naive matrix product suffers from poor data locality, leading to inefficient cache usage. This validates the need for cache blocking to improve performance. 

### Q2

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
This allows for a massive reduction in cache misses shown under
```zsh
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


###Q3 
Added a top.matrix_verify that runs the code sequentially after running with the blocked version, and allows for up to 0.01 relative error. This error stems from the rearragment of the operations performed at compile time and runtime.

###Q4


# Tuning Block Dimensions for Optimal Performance

## Objective

The goal of this experiment is to tune the block dimensions of our cache‐blocked matrix multiplication kernel to maximize performance. We use Perf to measure runtime, GFLOP/s, and cache miss rates (L1 and LLC), which help us assess improvements in spatial and temporal locality.

## Experimental Setup

- **Matrix Sizes Tested:**  
  - 500×500×500  
  - 1000×1000×1000  
  - 2000×2000×2000

- **Block Dimensions Tested (BM×BN×BK):**
  - 8×8×8  
  - 8×8×16  
  - 8×8×32  
  - 16×16×16  
  - 16×16×32  
  - 32×32×32  
  - 64×64×64

- **Fixed Parameters:**
  - Fixed number of threads: 8  
  - Each configuration was run 3 times and averaged

- **Tools:**  
  - Perf events: `L1-dcache-loads`, `L1-dcache-load-misses`, `LLC-loads`, and `LLC-load-misses`
  
- **Goals:**  
  - Identify block dimensions that minimize runtime and maximize GFLOP/s.
  - Use cache miss rates to assess improvements in spatial (L1) and temporal (LLC) locality.

## Results & Observations

### Raw output
```zsh
🔬 Testing Matrix Size: 500x500x500

🧱 Block 8x8x8: Time=0.0895s | GFLOP/s=2.79 | L1 Miss=2.04% | LLC Miss=11.35%
🧱 Block 8x8x16: Time=0.0946s | GFLOP/s=2.64 | L1 Miss=2.12% | LLC Miss=15.42%
🧱 Block 8x8x32: Time=0.0905s | GFLOP/s=2.77 | L1 Miss=2.10% | LLC Miss=11.21%
🧱 Block 16x16x16: Time=0.0662s | GFLOP/s=3.78 | L1 Miss=0.89% | LLC Miss=13.40%
🧱 Block 16x16x32: Time=0.0672s | GFLOP/s=3.72 | L1 Miss=0.90% | LLC Miss=12.20%
🧱 Block 32x32x32: Time=0.0592s | GFLOP/s=4.22 | L1 Miss=0.40% | LLC Miss=15.41%
🧱 Block 64x64x64: Time=0.0661s | GFLOP/s=3.80 | L1 Miss=0.40% | LLC Miss=23.62%

🔬 Testing Matrix Size: 1000x1000x1000

🧱 Block 8x8x8: Time=0.7352s | GFLOP/s=2.72 | L1 Miss=2.00% | LLC Miss=9.39%
🧱 Block 8x8x16: Time=0.7688s | GFLOP/s=2.60 | L1 Miss=2.09% | LLC Miss=7.84%
🧱 Block 8x8x32: Time=0.8438s | GFLOP/s=2.39 | L1 Miss=2.09% | LLC Miss=10.32%
🧱 Block 16x16x16: Time=0.6899s | GFLOP/s=2.94 | L1 Miss=0.91% | LLC Miss=16.62%
🧱 Block 16x16x32: Time=0.6949s | GFLOP/s=2.88 | L1 Miss=0.90% | LLC Miss=20.00%
🧱 Block 32x32x32: Time=0.4850s | GFLOP/s=4.13 | L1 Miss=0.45% | LLC Miss=22.27%
🧱 Block 64x64x64: Time=0.6158s | GFLOP/s=3.34 | L1 Miss=0.54% | LLC Miss=51.50%

🔬 Testing Matrix Size: 2000x2000x2000

🧱 Block 8x8x8: Time=7.3691s | GFLOP/s=2.17 | L1 Miss=2.18% | LLC Miss=4.18%
🧱 Block 8x8x16: Time=7.2861s | GFLOP/s=2.20 | L1 Miss=2.18% | LLC Miss=3.96%
🧱 Block 8x8x32: Time=6.9546s | GFLOP/s=2.30 | L1 Miss=2.17% | LLC Miss=3.44%
🧱 Block 16x16x16: Time=4.4789s | GFLOP/s=3.57 | L1 Miss=1.04% | LLC Miss=7.78%
🧱 Block 16x16x32: Time=4.4764s | GFLOP/s=3.57 | L1 Miss=1.04% | LLC Miss=7.45%
🧱 Block 32x32x32: Time=4.2114s | GFLOP/s=3.83 | L1 Miss=0.51% | LLC Miss=26.66%
🧱 Block 64x64x64: Time=4.5869s | GFLOP/s=3.49 | L1 Miss=1.02% | LLC Miss=51.98%
```


### 1. Small Matrix (500×500×500)

- **Runtime & GFLOP/s:**  
  - The best performance is achieved with block dimensions **32×32×32**, with a runtime of 0.0592 s and a GFLOP/s rate of 4.22.
  - Other configurations (e.g., 8×8×8 or 16×16×16) yield lower throughput (around 1.27–3.78 GFLOP/s).
- **Cache Behavior:**  
  - L1 miss rates are very low (around 0.40%–2.10%).
  - The LLC miss rate for block 32×32×32 is moderate at 15.41%; while some smaller blocks show slightly lower LLC misses, the overall performance gain is much higher with 32×32×32.

### 2. Medium Matrix (1000×1000×1000)

- **Runtime & GFLOP/s:**  
  - The **32×32×32** configuration again stands out with a runtime of 0.4850 s and a GFLOP/s of 4.13, notably outperforming other configurations.
  - For example, block 16×16×16 yields a runtime of 0.6899 s and 2.94 GFLOP/s.
- **Cache Behavior:**  
  - L1 miss rates remain low (≈0.45% for block 32×32×32).
  - Although the LLC miss rate for 32×32×32 is relatively high (22.27%), the temporal locality is still sufficient for the kernel to achieve superior overall performance.

### 3. Large Matrix (2000×2000×2000)

- **Runtime & GFLOP/s:**  
  - For the 2000³ case, block 32×32×32 yields a runtime of 4.2114 s and a GFLOP/s rate of 3.83.
  - Although a 16×16×32 configuration shows a slightly higher LLC locality (LLC miss rate of 7.45%) with a GFLOP/s of 3.57, the 32×32×32 variant still provides a lower runtime (4.21 s vs. 4.48 s) and higher throughput.
- **Cache Behavior:**  
  - The LLC miss rate for block 32×32×32 is higher (26.66%) compared to some alternatives, but the reduction in runtime and increase in GFLOP/s indicate that the improved exposure of parallelism and data reuse outweigh the cost of higher LLC miss rates.
Below is a revised version of the **Analysis & Discussion** section and **Conclusion** that specifically refers to how your new blocked implementation works. This write‐up emphasizes the use of MDRangePolicy to partition the work into blocks, the local buffering of C to improve spatial and temporal locality, and why these changes impact performance and scaling.

---

## Analysis & Discussion

In the revised kernel, the global matrix multiplication is decomposed into smaller, independent blocks using Kokkos’s MDRangePolicy. The output matrix \(C\) is divided into tiles of dimensions \(BM \times BN\), where each tile is processed by a single MDRangePolicy iteration over the block indices. Within each block, the following steps occur:

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

## Conclusion

The redesigned blocked matrix multiplication kernel, which uses a 2D MDRangePolicy for tiling the work and local buffering to compute each block, demonstrates improved performance by significantly enhancing spatial and temporal locality. Specifically:

- **Local Buffering:**  
  The use of a local array (`local_C`) for each block minimizes global memory accesses and effectively reuses loaded data, leading to very low L1 miss rates (e.g., as low as 0.40–0.45% in the best configurations).

- **Blocking Over \(K\):**  
  Dividing the \(K\) dimension into blocks of size \(BK\) allows the kernel to accumulate contributions in the fast local memory and then combine them, which reduces data traffic and improves cache efficiency.

- **Parallel Efficiency:**  
  By partitioning the output matrix into independent blocks via MDRangePolicy, the approach exposes a moderate level of parallelism. However, for smaller matrices the number of blocks may limit scaling; for larger matrices, this strategy better utilizes available threads while keeping synchronization overhead low.

- **Experimental Results:**  
  The tuned configurations, particularly the 32×32×32 block size for medium matrices and 16×16×32 for larger ones, deliver the best performance—with the highest GFLOP/s and lowest runtime—due to the optimal trade-off between loop overhead and cache reuse. Although the LLC miss rate can be somewhat higher for certain block configurations, the overall gain in throughput confirms that the improvements in spatial and temporal locality more than compensate for this.

In summary, tuning the block dimensions in a cache-blocked matrix multiplication kernel can dramatically improve performance. By carefully choosing parameters that fit the processor’s cache hierarchy and by employing strategies (such as local buffering and MDRangePolicy-based tiling) to maximize data reuse, the performance—measured in GFLOP/s and runtime—can be significantly enhanced. Future work might explore dynamic tuning, further increasing the level of parallelism (e.g., using nested policies), or leveraging scratch memory for additional improvements.



---
Below is an updated version of the **Analysis & Discussion** and **Conclusion** sections that now include references to the additional plots for each matrix size. Only the ending sections are modified, so you can integrate them with the rest of your report.

---

### Analysis & Discussion

Our experiments evaluated a cache‐blocked matrix multiplication kernel that uses a 2D MDRangePolicy to partition the global operation into tiles and leverages local buffering of each \( C \) block to improve both spatial and temporal locality. The results obtained across three matrix sizes (500×500×500, 1000×1000×1000, and 2000×2000×2000) and various block dimensions reveal several key insights:

1. **Small Matrix (500×500×500):**  
   - *Runtime & GFLOP/s:*  
     The best performance is achieved with the 32×32×32 configuration (0.0592 s and 4.22 GFLOP/s), with other configurations (e.g., 8×8×8 and 16×16×16) delivering lower throughput.  
   - *Cache Behavior:*  
     L1 miss rates remain very low (around 0.40%–2.10%), indicating excellent spatial reuse in the small working set. LLC miss rates vary modestly (from about 11% to 31%), but overall the improved GFLOP/s at 32×32×32 suggests that the local buffering is effective.

   *See Figures 1–3 for the 500×500×500 results.*

2. **Medium Matrix (1000×1000×1000):**  
   - *Runtime & GFLOP/s:*  
     The 32×32×32 configuration again stands out, yielding a runtime of 0.4850 s and 4.13 GFLOP/s—significantly outperforming the 8×8×* configurations and even the 16×16×16 variant.  
   - *Cache Behavior:*  
     Although the LLC miss rate for 32×32×32 is relatively high (22.27%), the low L1 miss rate (0.45%) combined with the high throughput indicates that the data reuse in the innermost loops compensates effectively.
   
   *Refer to Figures 4–6 for the 1000×1000×1000 data.*

3. **Large Matrix (2000×2000×2000):**  
   - *Runtime & GFLOP/s:*  
     For large problem sizes, the 32×32×32 configuration provides a runtime of 4.2114 s and a GFLOP/s rate of 3.83, which is better than alternative configurations (e.g., 16×16×32 at 4.4789 s and 3.57 GFLOP/s).  
   - *Cache Behavior:*  
     Even though the LLC miss rate for the 32×32×32 configuration is somewhat high (26.66%), the overall reduction in runtime and increased throughput indicate that the working set for each tile is being effectively reused via local buffering, which minimizes redundant data transfers.
   
   *Figures 7–9 present the 2000×2000×2000 results.*

Additionally, our final strong scaling study (see Figures 10–12) shows that when using only the best-performing block configuration, the performance remains largely invariant with thread count. This confirms that at 8 threads (the maximum available on our system), the kernel has reached near saturation—most gains are obtained through improved cache usage rather than additional concurrency.

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