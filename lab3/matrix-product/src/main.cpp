#include <cassert>
#include <cstdlib>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <fmt/core.h>

using Matrix = Kokkos::View<double**, Kokkos::LayoutLeft>;

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


// template <class AMatrixType, class BMatrixType, class CMatrixType>
// auto matrix_product_block(double alpha, 
//                           AMatrixType const& A, 
//                           BMatrixType const& B, 
//                           double beta, 
//                           CMatrixType& C, 
//                           int BM, int BN, int BK)
//   -> void {
//   static_assert(AMatrixType::rank() == 2 &&
//                 BMatrixType::rank() == 2 &&
//                 CMatrixType::rank() == 2,
//                 "Views must be of rank 2");
//   assert(A.extent(0) == C.extent(0));
//   assert(B.extent(1) == C.extent(1));
//   assert(A.extent(1) == B.extent(0));

//   int M = A.extent(0);
//   int N = B.extent(1);
//   int K = A.extent(1);

//   // Determine block grid dimensions.
//   int teams_i = (M + BM - 1) / BM;
//   int teams_j = (N + BN - 1) / BN;
//   int numTeams = teams_i * teams_j;

//   // You can allocate scratch memory if you have enough on-chip memory.
//   const int bytes_per_team = (BM * BK + BK * BN) * sizeof(double);

//   using team_policy = Kokkos::TeamPolicy<>;
//   Kokkos::parallel_for(
//     "blocked_matmul",
//     team_policy(numTeams, Kokkos::AUTO).set_scratch_size(0, Kokkos::PerTeam(bytes_per_team)),
//     KOKKOS_LAMBDA(const typename team_policy::member_type & teamMember) {
//       int team_id = teamMember.league_rank();
//       int bi = team_id / teams_j;
//       int bj = team_id % teams_j;
      
//       int i_start = bi * BM;
//       int i_end   = (i_start + BM > M ? M : i_start + BM);
//       int j_start = bj * BN;
//       int j_end   = (j_start + BN > N ? N : j_start + BN);
      
//       // Allocate scratch memory for tiles of A and B.
//       auto scratch_A = Kokkos::View<double**, Kokkos::MemoryTraits<Kokkos::Unmanaged> >(
//                              teamMember.team_scratch(0), BM, BK);
//       auto scratch_B = Kokkos::View<double**, Kokkos::MemoryTraits<Kokkos::Unmanaged> >(
//                              teamMember.team_scratch(0), BK, BN);
      
//       Kokkos::parallel_for(
//         Kokkos::TeamThreadRange(teamMember, i_start, i_end),
//         [=](int i) {
//           Kokkos::parallel_for(
//             Kokkos::ThreadVectorRange(teamMember, j_start, j_end),
//             [=](int j) {
//               double acc = beta * C(i,j);
//               for (int kk = 0; kk < K; kk += BK) {
//                 int k_bound = (kk + BK > K ? K : kk + BK);
//                 // Optionally: load a tile of A and B into scratch memory here.
//                 // For illustration, we proceed directly.
//                 for (int k = kk; k < k_bound; ++k) {
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
// void matrix_product_block(double alpha, AMatrixType const& A, BMatrixType const& B, double beta, CMatrixType& C, int BM, int BN, int BK) {
//   static_assert(
//     AMatrixType::rank() == 2 && BMatrixType::rank() == 2 && CMatrixType::rank() == 2,
//     "Views must be of rank 2"
//   );
//   assert(A.extent(0) == C.extent(0));
//   assert(B.extent(1) == C.extent(1));
//   assert(A.extent(1) == B.extent(0));

//   // Define block sizes (tune these based on your CPU cache)
//   int BLOCK_I = BM;
//   int BLOCK_J = BN;
//   int BLOCK_K = BK;

//   using execution_space = typename CMatrixType::execution_space;
//   using team_policy = Kokkos::TeamPolicy<execution_space>;
//   using team_member = typename team_policy::member_type;

//   constexpr int BLOCK_SIZE = 64; // Tune based on cache
//   const int league_size = (C.extent(0) + BLOCK_SIZE - 1) / BLOCK_SIZE;

//   Kokkos::parallel_for("dgemm_team", team_policy(league_size, Kokkos::AUTO),
//     KOKKOS_LAMBDA(const team_member& team) {
//       const int i_team = team.league_rank() * BLOCK_SIZE;
//       const int i_end = (i_team + BLOCK_SIZE < C.extent(0)) ? i_team + BLOCK_SIZE : C.extent(0);

//       // Distribute columns among threads in the team
//       Kokkos::parallel_for(Kokkos::TeamThreadRange(team, C.extent(1)), [&](const int j) {
//         for (int i = i_team; i < i_end; ++i) {
//           double acc = 0.0;
//           for (int k = 0; k < A.extent(1); ++k) {
//             acc += alpha * A(i, k) * B(k, j);
//           }
//           C(i, j) = beta * C(i, j) + acc;
//         }
//       });
//     }
//   );
// }

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
    #ifdef KOKKOS_ENABLE_OPENMP
    std::cout << "OpenMP is enabled!" << std::endl;
    std::cout << "Max threads: " << Kokkos::OpenMP::concurrency() << std::endl;
  #else
    std::cout << "OpenMP is NOT enabled!" << std::endl;
  #endif

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
    // matrix_product(alpha, A, B, beta, C);
    matrix_product_block(alpha, A, B, beta, C, BM,BN,BK);
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
