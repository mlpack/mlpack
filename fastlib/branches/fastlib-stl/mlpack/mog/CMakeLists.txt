cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
# This directory has multiple output libraries, so sources are specified twice, once here, and once for
# the individual library, so make sure you specify your sources in the correct places
set(SOURCES
   # for mog_em
   mog_em.h
   mog_em.cc
   phi.h
   math_functions.h
   # for mog_l2e
   mog_l2e.h
   mog_l2e.cc
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# main executable, em
add_executable(mog_em
   mog_em_main.cc
)
# link dependencies of mog_em
target_link_libraries(mog_em
   mlpack
   fastlib
)

# main executable, l2e
add_executable(mog_l2e
   mog_l2e_main.cc
)
target_link_libraries(mog_l2e
   mlpack
   fastlib
)
