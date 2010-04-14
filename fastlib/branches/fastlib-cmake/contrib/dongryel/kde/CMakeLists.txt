cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   kde_cv.h
   kde_delta.h
   kde_error.h
   kde_global.h
   kde_problem.h
   kde_query_postponed.h
   kde_query_result.h
   kde_query_summary.h
   kde_stat.h
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# executable, renamed because MLPACK has a 'kde' program
add_executable(kde_dongryel
   kde_main.cc
)
# link dependencies of test executable
target_link_libraries(kde_dongryel
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(kde_cv_dongryel
   kde_cv_main.cc
)
target_link_libraries(kde_cv_dongryel
   fastlib
   mlpack
   mlpack_contrib
)
