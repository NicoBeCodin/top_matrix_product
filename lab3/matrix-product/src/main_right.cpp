#include <cassert>
#include <cstdlib>

#include <Kokkos_Core.hpp>

#include <fmt/core.h>

using Matrix = Kokkos::View<double**, Kokkos::LayoutRight>;

template <class MatrixType>
auto matrix_init(MatrixType& M) -> void {
  static_assert(2 == MatrixType::rank(), "View must be of rank 2");

  Kokkos::parallel_for(
    "init",
    M.extent(0),
    KOKKOS_LAMBDA(int i) {
      for (int j = 0; j < int(M.extent(1)); ++j) {
        M(i, j) = drand48();
      }
    }
  );
}

template <class AMatrixType, class BMatrixType, class CMatrixType>
auto matrix_product(double alpha, AMatrixType const& A, BMatrixType const& B, double beta, CMatrixType& C) -> void {
  static_assert(
    AMatrixType::rank() == 2 && BMatrixType::rank() == 2 && CMatrixType::rank() == 2, "Views must be of rank 2"
  );
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));
  assert(A.extent(1) == B.extent(0));

  Kokkos::parallel_for(
    "dgemm_kernel",
    A.extent(0),
    KOKKOS_LAMBDA(int i) {
      for (int j = 0; j < int(B.extent(1)); ++j) {
        double acc = 0.0;
        for (int k = 0; k < int(A.extent(1)); ++k) {
          acc += alpha * A(i, k) * B(k, j);
        }
        C(i, j) *= beta + acc;
      }
    }
  );
}

template <class AMatrixType, class BMatrixType, class CMatrixType>
auto matrix_product_block(double alpha, AMatrixType const& A, BMatrixType const& B, double beta, CMatrixType& C, int BM, int BN, int BK)
  -> void {
  static_assert(
    AMatrixType::rank() == 2 && BMatrixType::rank() == 2 && CMatrixType::rank() == 2, "Views must be of rank 2"
  );
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));
  assert(A.extent(1) == B.extent(0));

  int M = A.extent(0);
  int N = B.extent(1);
  int K = A.extent(1);

  Kokkos::parallel_for(
    "blocked_matmul",
    Kokkos::RangePolicy<>(0, (M + BM - 1) / BM),
    KOKKOS_LAMBDA(int block_i) {
      int ii = block_i * BM;
      for (int jj = 0; jj < N; jj += BN) {
        for (int i = ii; i < ii + BM && i < M; ++i) {
          for (int j = jj; j < jj + BN && j < N; ++j) {
            // Apply beta once before accumulating
            double acc = beta * C(i, j);
            for (int kk = 0; kk < K; kk += BK) {
              for (int k = kk; k < kk + BK && k < K; ++k) {
                acc += alpha * A(i, k) * B(k, j);
              }
            }
            C(i, j) = acc;
          }
        }
      }
    }
  );
}

auto main(int argc, char* argv[]) -> int {
  if (argc < 4) {
    fmt::print("Usage: {} <M> <N> <K>\n", argv[0]);
    return -1;
  }
  int m = std::atoi(argv[1]);
  int n = std::atoi(argv[2]);
  int k = std::atoi(argv[3]);

  int BM,BN,BK;

  if (argc>4){
      BM = std::atoi(argv[4]);
      BN = std::atoi(argv[5]);
      BK = std::atoi(argv[6]);
  } else {
    BM = 32;
    BN = 32;
    BK = 32;
  }

  // Known seed for deterministic RNG
  srand48(42);

  Kokkos::initialize(argc, argv);
  {
    auto A = Matrix("A", m, k);
    auto B = Matrix("B", k, n);
    auto C = Matrix("C", m, n);

    double alpha = drand48();
    matrix_init(A);
    matrix_init(B);
    double beta = drand48();
    matrix_init(C);

    Kokkos::fence();
    // Ensure all matrices are initialized
    Kokkos::Timer timer;
    matrix_product(alpha, A, B, beta, C);
    // matrix_product_block(alpha, A, B, beta, C, BM,BN,BK);
    Kokkos::fence();
    double time   = timer.seconds();
    double flops  = 2.0 * m * n * k;
    double gflops = flops / (time * 1e9);

    fmt::print("Matrix size: {} x {} x {}\n", m, n, k);
    fmt::print("Runtime: {:.6f} seconds\n", time);
    fmt::print("Performance: {:.2f} GFLOP/s\n", gflops);

  }
  Kokkos::finalize();
  // Create a reference C matrix for correctness check
  return 0;
}
