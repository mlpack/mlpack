cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   general_spacetree.h
   gen_kdtree.h
   gen_metric_tree.h
   gen_range.h
   opt_fw.h
   opt_lasvm.h
   opt_par.h
   opt_rivanov.h
   opt_sgd.h
   opt_smo.h
   svm.h
   general_type_bounds.h
   gen_kdtree_impl.h
   gen_metric_tree_impl.h
   opt_cd.h
   opt_hcy.h
   opt_mfw.h
   opt_pegasos.h
   opt_sfw.h
   opt_sgdwb.h
   opt_sparsereg.h
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# executable
add_executable(svm_houyang
   EXCLUDE_FROM_ALL
   svm_main.cc
   opt_smo.h
   opt_sgd.h
   svm.h
)
# link dependencies of executable
target_link_libraries(svm_houyang
   fastlib
   mlpack
   mlpack_contrib
)
