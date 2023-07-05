cmake_minimum_required(VERSION 3.22)
project(Magma)

set(CMAKE_CXX_STANDARD 23)

find_package(glm)
find_package(glfw3)

add_subdirectory(lib/volk)

add_executable(Magma main.cpp)

target_link_libraries(Magma glfw volk_headers shaderc_shared)
