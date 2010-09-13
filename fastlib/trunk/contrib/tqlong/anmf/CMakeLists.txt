cmake_minimum_required(VERSION 2.8)

set(SOURCES
  anmf
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_executable(anmf_test
   EXCLUDE_FROM_ALL
   anmf_test.cpp
)
add_dependencies(anmf_test
   mlpack_contrib
)
# link dependencies of executable
target_link_libraries(anmf_test
   mlpack_contrib
   fastlib
)
