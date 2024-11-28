cmake_minimum_required(VERSION 3.16)
project(arm-gemm-bench CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

if(APPLE)
    set(CMAKE_CXX_FLAGS "-O2")
    set(CMAKE_EXE_LINKER_FLAGS "-framework Accelerate")
else()
    set(CMAKE_CXX_FLAGS "-O2 -mavx -msse4")
endif()

if(APPLE)
    include_directories(/opt/homebrew/opt/openblas/include)
    link_directories(/opt/homebrew/opt/openblas/lib)
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)
add_subdirectory(source)
