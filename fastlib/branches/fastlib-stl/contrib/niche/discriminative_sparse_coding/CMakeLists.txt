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
   lcc.h
   lcc_impl.h
   svm.h
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
#set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# The executable for Discriminative Sparse Coding
add_executable(discr_sparse_coding
    EXCLUDE_FROM_ALL
    main.cc
)

target_link_libraries(discr_sparse_coding
   fastlib
)

add_executable(dsc_mnist
    EXCLUDE_FROM_ALL
    mnist_main.cc
)

target_link_libraries(dsc_mnist
   fastlib
)

add_executable(mnist_1vsrest
    EXCLUDE_FROM_ALL
    mnist_one_vs_rest_main.cc
)

target_link_libraries(mnist_1vsrest
   fastlib
)

add_executable(mnist_pairwise
    EXCLUDE_FROM_ALL
    mnist_pairwise.cc
)

target_link_libraries(mnist_pairwise
   fastlib
)

add_executable(mnist_pairwise_test
    EXCLUDE_FROM_ALL
    mnist_pairwise_test.cc
)

target_link_libraries(mnist_pairwise_test
   fastlib
)

add_executable(mnist_dlcc
    EXCLUDE_FROM_ALL
    mnist_dlcc_main.cc
)

target_link_libraries(mnist_dlcc
    mlpack
    fastlib
)

