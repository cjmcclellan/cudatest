cmake_minimum_required(VERSION 3.10)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(cudatest LANGUAGES C CXX CUDA)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ./build)
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")

find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    message("CUDA found")
else()
    message("CUDA not found, doing something alternatively")
endif()

set(CMAKE_CXX_STANDARD 14)

set(TENSORFLOW_LIB_DIR "/opt/tensorflow/lib")
#set(TENSORFLOW_LIB_DIR "~/Documents/DeepSim/TF/tensorflow/bazel-bin/tensorflow")

#
set(TFLIBS ${TENSORFLOW_LIB_DIR}/libtensorflow_cc.so ${TENSORFLOW_LIB_DIR}/libtensorflow_framework.so)

add_executable(cudatest main.cpp loadmodels_incuda.cpp)
target_include_directories(cudatest PUBLIC ${CUDA_INCLUDE_DIRS} ${TENSORFLOW_LIB_DIR}/include)
target_link_libraries(cudatest PUBLIC ${CUDA_LIBRARIES} ${TFLIBS} /usr/local/cuda/lib64/libcublas.so /usr/local/cuda/lib64/libcusparse.so)
