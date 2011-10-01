# All we have to do is recurse into the subdirectories.
add_subdirectory(kernels)

set(MLPACK_SRCS ${MLPACK_SRCS} PARENT_SCOPE)
