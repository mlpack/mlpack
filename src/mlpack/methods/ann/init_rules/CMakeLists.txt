# Define the files we need to compile
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  const_init.hpp
  gaussian_init.hpp
  he_init.hpp
  init_rules_traits.hpp
  kathirvalavakumar_subavathi_init.hpp
  lecun_normal_init.hpp
  network_init.hpp
  nguyen_widrow_init.hpp
  oivs_init.hpp
  orthogonal_init.hpp
  random_init.hpp
  glorot_init.hpp
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all mlpack sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
