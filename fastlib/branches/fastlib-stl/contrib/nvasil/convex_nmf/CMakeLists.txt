cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   # nmflib
   sdp_objectives.h
   sdp_objectives_impl.h
   sdp_nmf_engine.h
   geometric_nmf.h
   geometric_nmf_impl.h
   geometric_nmf_engine.h
   geometric_nmf_seq_engine.h
   # gopnmflib
   gop_nmf.h
   gop_nmf_impl.h
   splitter.h
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# executable
add_executable(nmf
   EXCLUDE_FROM_ALL
   main.cc
)
# link dependencies of executable
target_link_libraries(nmf
   fastlib
   mlpack
   mlpack_contrib
)
