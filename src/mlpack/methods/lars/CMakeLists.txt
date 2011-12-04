cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
# In this library, these are specified twice, once here, and once for the individual library it belongs to, so make sure
# that you have files in both sections
set(SOURCES
   lars.hpp
   lars_impl.hpp
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# The executable for LARS
add_executable(lars
  lars_main.cpp
)

target_link_libraries(lars
  mlpack
)

add_executable(test_chol
  one_test_chol.cpp
)

target_link_libraries(test_chol
  mlpack
)

add_executable(test_gram
  one_test_gram.cpp
)

target_link_libraries(test_gram
  mlpack
)
