cmake_minimum_required(VERSION 3.10)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(cudatest LANGUAGES CXX CUDA)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ./build)
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")

find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    message("CUDA found")
else()
    message("CUDA not found, doing something alternatively")
endif()

set(CMAKE_CXX_STANDARD 14)

add_executable(cudatest main.cpp)
target_include_directories(cudatest PUBLIC ${CUDA_INCLUDE_DIRS})
target_link_libraries(cudatest PUBLIC ${CUDA_LIBRARIES} /usr/local/cuda/lib64/libcublas.so /usr/local/cuda/lib64/libcusparse.so)
