cmake_minimum_required(VERSION 3.25)
project(top.matrix_product LANGUAGES CXX)

find_package(Kokkos REQUIRED)
find_package(fmt)
find_package(OpenMP REQUIRED)

add_subdirectory(src)
