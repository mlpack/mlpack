cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   ansi_colors.h
   base.h
   cc.h
   cc.cc
   ccmem.h
   ccmem.cc
   common.h
   common.c
   compiler.h
   debug.h
   debug.c
   deprecated.h
   fortran.h
   otrav.h
   otrav.cc
   otrav_impl.h
   test.h
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
    set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all FASTLIB sources (used at the parent cmake scope)
set(FASTLIB_SRCS ${FASTLIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_library(base ${SOURCES})
