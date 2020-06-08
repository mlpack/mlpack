cmake_minimum_required(VERSION 3.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


project(range_search)

set( SOURCES  
  range_search_low_resource_main.cpp
  )


include_directories(
 
  )

add_executable(range_search
  
 ${SOURCES}  

  )

target_link_libraries(range_search

  )



