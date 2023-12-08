cmake_minimum_required(VERSION 3.14)
project(illithid VERSION 0.0.1)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(nn)
add_subdirectory(plot)

add_executable(illithid main.cpp)
target_link_libraries(illithid PRIVATE nn plot)