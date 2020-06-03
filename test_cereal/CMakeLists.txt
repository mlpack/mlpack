cmake_minimum_required(VERSION 3.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## Specify your project's name
project(po)

set( SOURCES  
  knn_low_resource_main.cpp
  )

# Specify include directories
include_directories(
 
  )

#set(boost_lib_program_options "/usr/lib/libboost_program_options.a")

add_executable(po
  
 ${SOURCES}  
  # ... any other source files
  )

target_link_libraries(po

 # ${boost_lib_program_options}

  )



