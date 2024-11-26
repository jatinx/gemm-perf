cmake_minimum_required(VERSION 3.16)
project(arm-gemm-bench CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "-O2 -framework Accelerate")

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/opt/homebrew/opt/openblas/include)
link_directories(/opt/homebrew/opt/openblas/lib)
add_subdirectory(source)
