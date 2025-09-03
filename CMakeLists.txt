cmake_minimum_required(VERSION 3.12)
project(fa2s_interpreter_project LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(fa2s_interpreter src/interpreter.cpp)
target_include_directories(fa2s_interpreter PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(fa2s_interpreter PRIVATE -Wall -Wextra -Wpedantic)

add_executable(fa2s tools/fa2s_cli.cpp)
target_link_libraries(fa2s PRIVATE fa2s_interpreter)
target_compile_options(fa2s PRIVATE -Wall -Wextra -Wpedantic)