# Define the files that we need to compile.
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
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
