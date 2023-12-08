cmake_minimum_required(VERSION 3.0)

file(GLOB SOURCES "*.cpp")
add_library(nn ${SOURCES})