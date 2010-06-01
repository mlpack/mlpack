cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   gop_tight_nmf_engine.h
   gop_tight_nmf_engine_impl.h
   relaxed_nmf_bound_tightener.h
   relaxed_nmf_bound_tightener_impl.h
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# executable
#add_executable(gop_tight_nmf_engine_test
#   gop_tight_nmf_engine_test.cc
#)
# link dependencies of executable
#target_link_libraries(gop_tight_nmf_engine_test
#   fastlib
#   mlpack
#   mlpack_contrib
#)
