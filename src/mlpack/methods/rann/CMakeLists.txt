# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
  # rank-approximate search files
  ra_search.hpp
  ra_search_impl.hpp

  # rank-approximate search rules
  ra_search_rules.hpp
  ra_search_rules_impl.hpp

  # query statistic
  ra_query_stat.hpp

  # typedefs
  ra_typedef.hpp

  # utilities
  ra_util.hpp
  ra_util.cpp

  # model
  ra_model.hpp
  ra_model_impl.hpp
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all mlpack sources (used at the parent scope)
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)


# The code to compute the rank-approximate neighbor
# for the given query and reference sets
add_cli_executable(allkrann)
