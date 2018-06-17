# Define the files we need to compile
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  no_normalization.hpp
  overall_mean_normalization.hpp
  user_mean_normalization.hpp
  item_mean_normalization.hpp
  z_score_normalization.hpp
  combined_normalization.hpp
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all mlpack sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
