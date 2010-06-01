cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   # cover_tree
   ctree.h
   ctree_impl.h
   cover_tree.h
   # emst_cover
   dtb_cover.h
   emst_cover.h
   # geomst2
   geomst2.h
     # emst_cover.h  # already included
   # single_fragment
   friedman_bentley.h
     # emst_cover.h  # already included
   friedman_bentley.cc
   # multi_fragment
   multi_fragment.h
     # emst_cover.h  # already included
   multi_fragment.cc
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# executable
add_executable(emst_cover
   EXCLUDE_FROM_ALL
   emst_cover_main.cc
)
# link dependencies of executable
target_link_libraries(emst_cover
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(mst_comparison
   EXCLUDE_FROM_ALL
   mst_comparison.h
   comparison_main.cc
)
target_link_libraries(mst_comparison
   fastlib
   mlpack
   mlpack_contrib
)
