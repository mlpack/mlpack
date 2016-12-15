# Define the files that we need to compile.
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  bernoulli.hpp
  boost_backport_math.hpp
  boost_backport_serialization.hpp
  detail/bernoulli_details.hpp
  detail/polygamma.hpp
  detail/unchecked_bernoulli.hpp
  math_fwd.hpp
  policy.hpp
  polygamma.hpp
  trigamma.hpp
  unordered_collections_load_imp.hpp
  unordered_collections_save_imp.hpp
  unordered_map.hpp
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
    set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

# Append sources (with directory name) to list of all mlpack sources (used at
# parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
