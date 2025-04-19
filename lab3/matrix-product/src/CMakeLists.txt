add_executable(top.matrix_product)
target_sources(top.matrix_product PRIVATE main.cpp)
target_compile_options(top.matrix_product PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(top.matrix_product PRIVATE Kokkos::kokkos fmt::fmt)


# ✅ Add verification tool
add_executable(top.matrix_verify)
target_sources(top.matrix_verify PRIVATE verify.cpp)
target_compile_options(top.matrix_verify PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(top.matrix_verify PRIVATE Kokkos::kokkos fmt::fmt)

# ✅ Add verification tool
add_executable(top.matrix_product_right)
target_sources(top.matrix_product_right PRIVATE main_right.cpp)
target_compile_options(top.matrix_product_right PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(top.matrix_product_right PRIVATE Kokkos::kokkos fmt::fmt)