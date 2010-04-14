cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   dtw.h
   dtw_impl.h
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# executable
add_executable(dtw
   main.cc
   dtw.h
)
# link dependencies of executable
target_link_libraries(dtw
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(multidtw
   dtw.h
   dtw_impl.h
   multivariate_main.cc
)
target_link_libraries(multidtw
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(nearest_centroid
   mean_ts_classifier.cc
   dtw.h
   dtw_impl.h
)
target_link_libraries(nearest_centroid
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(arraylist_test
   test_arraylist.cc
)
target_link_libraries(arraylist_test
   fastlib
   mlpack
   mlpack_contrib
)
