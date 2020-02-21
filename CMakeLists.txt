
cmake_minimum_required(VERSION 3.16.4)
project(pong-ml-workspace)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

# Chipmunk: 2D Physics Engine for Games (C)
add_subdirectory(external/Chipmunk-7.0.3)

# Pong: The Machine Learning Game (C++)
add_subdirectory(pong-ml)
