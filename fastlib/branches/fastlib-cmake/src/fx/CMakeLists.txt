cmake_minimum_required(VERSION 2.8)

# Define the files that we need to compile
# Anything not in this list will not be compiled into the output library
# Do not add test executables here
set(SOURCES
   datanode.h
   datanode.c
   fx.h
   fx.c
   stopwatch.h
   stopwatch.c
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
    set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all FASTLIB sources (used at parent scope)
set(FASTLIB_SRCS ${FASTLIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_library(fx ${SOURCES})
# link dependencies of libfx
target_link_libraries(fx
   base
)
