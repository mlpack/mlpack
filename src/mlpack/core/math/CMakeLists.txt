# Define the files we need to compile
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  clamp.hpp
  columns_to_blocks.hpp
  columns_to_blocks.cpp
  lin_alg.hpp
  lin_alg_impl.hpp
  lin_alg.cpp
  log_add.hpp
  log_add_impl.hpp
  make_alias.hpp
  multiply_slices_impl.hpp
  multiply_slices.hpp
  random.hpp
  random.cpp
  random_basis.hpp
  random_basis.cpp
  range.hpp
  range_impl.hpp
  round.hpp
  shuffle_data.hpp
  ccov.hpp
  ccov_impl.hpp
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all mlpack sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
