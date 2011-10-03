cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
# In this library, these are specified twice, once here, and once for the individual library it belongs to, so make sure
# that you have files in both sections
set(SOURCES
  # for discriminative_sparse_coding
   discr_sparse_coding.h
   discr_sparse_coding_impl.h
   lars.h
   lars_impl.h
   sparse_coding.h
   sparse_coding_impl.h
   lcc.h
   lcc_impl.h
   pegasos.h
   pegasos_impl.h
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
#set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# The executable for Discriminative Sparse Coding

add_executable(mnist_dlcc
    EXCLUDE_FROM_ALL
    mnist_dlcc_main.cc
)

target_link_libraries(mnist_dlcc
    mlpack
    fastlib
)

add_executable(mnist_dlcc_test
    EXCLUDE_FROM_ALL
    mnist_dlcc_test.cc
)

target_link_libraries(mnist_dlcc_test
    mlpack
    fastlib
)

add_executable(mnist_dlcc_test_1000
    EXCLUDE_FROM_ALL
    mnist_dlcc_test_1000.cc
)

target_link_libraries(mnist_dlcc_test_1000
    mlpack
    fastlib
)

add_executable(mnist_dsc
    EXCLUDE_FROM_ALL
    mnist_dsc.cc
)

target_link_libraries(mnist_dsc
    mlpack
    fastlib
)

add_executable(mnist_small_dsc
    EXCLUDE_FROM_ALL
    mnist_small_dsc.cc
)

target_link_libraries(mnist_small_dsc
    mlpack
    fastlib
)
