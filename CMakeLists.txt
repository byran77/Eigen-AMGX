cmake_minimum_required(VERSION 3.5)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 52)
endif()

project(cudaSolver LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS_RELEASE "/MT /O2 /Ob2 /DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-MT")

add_library(cudaSolver STATIC
    src/cudaSolver.cu
    src/cudaSolver.cpp
    src/transMatrix.cpp
)

target_include_directories(cudaSolver
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ../../AMGX-2.2.0/base/include
        ../../eigen-3.3.9
)

target_compile_definitions(cudaSolver
    PUBLIC
        AMGX_API_NO_IMPORTS
)

target_link_directories(cudaSolver
    PUBLIC
        ../../AMGX-2.2.0/build/Release
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1/lib/x64"
)

target_link_libraries(cudaSolver
    PUBLIC
        cublas
        cusolver
        cusparse
        amgx
)

add_executable(test 
    example/test.cpp
)

target_link_libraries(test
    PRIVATE
        cudaSolver
)