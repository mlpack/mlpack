cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   adler32.c
   compressed_matrix.h
   compress.c
   crc32.h
   crc32.c
   deflate.h
   deflate.c
   example.c
   gzio.c
   infback.c
   inffast.h
   inffast.c
   inflate.h
   inflate.c
   inftrees.h
   inftrees.c
   minigzip.c
   trees.h
   trees.c
   uncompr.c
   zutil.h
   zutil.c
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# test executable
add_executable(compression_test
   EXCLUDE_FROM_ALL
   compressed_matrix.h
   compression_test.cc
)
# link dependencies of test executable
target_link_libraries(compression_test
   fastlib
   mlpack
   mlpack_contrib
)
