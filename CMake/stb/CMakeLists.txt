cmake_minimum_required(VERSION 3.9)
project(CheckSTB)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(alib SHARED alib.hpp alib.cpp)
add_library(blib SHARED blib.hpp blib.cpp)

#set(alib "alib.so")
#set(blib "libblib.so")

add_executable(CheckSTB
  main.cpp
  )

target_link_libraries(CheckSTB
  alib
  blib
  )
