cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   kmeans_nonempty.h
   la_augment.h
   do_mcf.h
   mcf/readmin.c
   mcf/mcfutil.c
   mcf/pbeampp2.c
   mcf/pstart.c
   mcf/pbla.c
   mcf/treeup.c
   mcf/pflowup.c
   mcf/psimplex.c
   mcf/output.c
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# executable
add_executable(kmeans_nonempty_test
   kmeans_nonempty.cc
)
# link dependencies of executable
target_link_libraries(kmeans_nonempty_test
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(do_mcf
   do_mcf.cc
)
target_link_libraries(do_mcf
   fastlib
   mlpack
   mlpack_contrib
)
