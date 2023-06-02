cmake_minimum_required(VERSION 3.25)
project(Proyecto-Paralela-3 CUDA)
enable_language(CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(Proyecto hough.cu common/pgm.cpp)