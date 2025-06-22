cmake_minimum_required(VERSION 3.22)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_CLANG_TIDY clang-tidy)

project(nnlcpp)

# Add include directory for compilation
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inc)

# Add all source files
set(SOURCES
    src/main.cpp
    src/layer.cpp
    src/matrix.cpp
    src/neuralnetwork.cpp
)

# Add executable target
add_executable(nnl ${SOURCES})