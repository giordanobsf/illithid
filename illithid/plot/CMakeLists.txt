cmake_minimum_required(VERSION 3.0)

find_package(Matplot++)

if (Matplot++_FOUND)
    file(GLOB SOURCES "*.cpp")
    add_library(plot ${SOURCES})
    target_link_libraries(plot PRIVATE Matplot++::matplot)
endif ()