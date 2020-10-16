cmake_minimum_required(VERSION 3.17)
project(fnmax CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(fnmax src/main.cu src/kernels.cu src/functions.cu)
target_include_directories(fnmax PUBLIC include)

set_target_properties(
        fnmax
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)