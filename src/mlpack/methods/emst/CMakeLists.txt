# Define the files we need to compile
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  # union_find
  union_find.hpp
  # dtb
  dtb.hpp
  dtb_impl.hpp
  dtb_rules.hpp
  dtb_rules_impl.hpp
  dtb_stat.hpp
  edge_pair.hpp
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all mlpack sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_cli_executable(emst)
