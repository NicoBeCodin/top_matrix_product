#include <cassert>
#include <cstdlib>
#include <cmath>
#include <Kokkos_Core.hpp>
#include <fmt/core.h>

using Matrix = Kokkos::View<double**, Kokkos::LayoutRight>;

template <class MatrixType>
void matrix_init(MatrixType& M) {
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


// template <class AMatrixType, class BMatrixType, class CMatrixType>
// auto matrix_product_block(double alpha, 
//                           AMatrixType const& A, 
//                           BMatrixType const& B, 
//                           double beta, 
//                           CMatrixType& C, 
//                           int BM, int BN, int BK)
//   -> void {
//   static_assert( AMatrixType::rank() == 2 &&
//                  BMatrixType::rank() == 2 &&
//                  CMatrixType::rank() == 2,
//                  "Views must be of rank 2");
//   assert(A.extent(0) == C.extent(0));
//   assert(B.extent(1) == C.extent(1));
//   assert(A.extent(1) == B.extent(0));

//   int M = A.extent(0);
//   int N = B.extent(1);
//   int K = A.extent(1);

//   // Determine number of blocks along rows and columns.
//   int teams_i = (M + BM - 1) / BM;
//   int teams_j = (N + BN - 1) / BN;
//   // Total number of teams: one per block of C.
//   int numTeams = teams_i * teams_j;

//   // Use Kokkos::TeamPolicy to distribute blocks among teams.
//   using team_policy = Kokkos::TeamPolicy<>;
//   Kokkos::parallel_for(
//     "blocked_matmul",
//     team_policy(numTeams, Kokkos::AUTO), // Kokkos::AUTO lets Kokkos choose team size
//     KOKKOS_LAMBDA(const typename team_policy::member_type & teamMember) {
//       // Map team index to block indices
//       int team_id = teamMember.league_rank();
//       int bi = team_id / teams_j;   // Block row index
//       int bj = team_id % teams_j;   // Block column index
      
//       int i_start = bi * BM;
//       int i_end   = (i_start + BM > M ? M : i_start + BM);
//       int j_start = bj * BN;
//       int j_end   = (j_start + BN > N ? N : j_start + BN);
      
//       // Parallelize over the rows in the block.
//       Kokkos::parallel_for(
//         Kokkos::TeamThreadRange(teamMember, i_start, i_end),
//         [=](int i) {
//           // Within each row, parallelize over the columns using vector-range.
//           Kokkos::parallel_for(
//             Kokkos::ThreadVectorRange(teamMember, j_start, j_end),
//             [=](int j) {
//               double acc = beta * C(i,j);
//               // Process the K-dimension in blocks of size BK.
//               for (int kk = 0; kk < K; kk += BK) {
//                 int k_end = (kk + BK > K ? K : kk + BK);
//                 for (int k = kk; k < k_end; ++k) {
//                   acc += alpha * A(i,k) * B(k,j);
//                 }
//               }
//               C(i,j) = acc;
//             }
//           );
//         }
//       );
//       teamMember.team_barrier();
//     }
//   );
// }


int main(int argc, char* argv[]) {
  if (argc < 7) {
    fmt::print("Usage: {} <M> <N> <K> <BM> <BN> <BK>\n", argv[0]);
    return 1;
  }

  int M = std::atoi(argv[1]);
  int N = std::atoi(argv[2]);
  int K = std::atoi(argv[3]);
  int BM = std::atoi(argv[4]);
  int BN = std::atoi(argv[5]);
  int BK = std::atoi(argv[6]);

  srand48(42);
  Kokkos::initialize(argc, argv);
  {
    Matrix A("A", M, K);
    Matrix B("B", K, N);
    Matrix C("C", M, N);
    Matrix C_ref("C_ref", M, N);

    double alpha = drand48();
    double beta = drand48();

    matrix_init(A);
    matrix_init(B);
    matrix_init(C);
    matrix_init(C_ref);

    Kokkos::fence();
    matrix_product_block(alpha, A, B, beta, C, BM, BN, BK);
    Kokkos::fence();

    // Sequential baseline (on host)
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        double acc = 0.0;
        for (int k = 0; k < K; ++k) {
          acc += alpha * A(i, k) * B(k, j);
        }
        C_ref(i, j) = beta * C_ref(i, j) + acc;
      }
    }

    auto C_host = Kokkos::create_mirror_view(C);
    auto C_ref_host = Kokkos::create_mirror_view(C_ref);
    Kokkos::deep_copy(C_host, C);
    Kokkos::deep_copy(C_ref_host, C_ref);

    bool correct = true;
    for (int i = 0; i < M && correct; ++i) {
      for (int j = 0; j < N && correct; ++j) {
        double rel_err = std::abs((C_host(i,j) - C_ref_host(i,j)) / C_ref_host(i,j));
        if (rel_err > 0.01) {
          fmt::print("❌ Mismatch at ({}, {}) — got {}, expected {}, rel error {}\n",
                     i, j, C_host(i,j), C_ref_host(i,j), rel_err);
          correct = false;
        }
      }
    }

    if (correct) {
      fmt::print("✅ Blocked matrix product is correct.\n");
    } else {
      fmt::print("❌ Blocked matrix product is incorrect.\n");
    }
  }
  Kokkos::finalize();
  return 0;
}
