# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES

  # rank-approximate search files
  ra_search.hpp
  ra_search_impl.hpp

  # The rank-approximate search rules
  ra_search_rules.hpp
  ra_search_rules_impl.hpp

  # The typedefs
  ra_typedef.hpp
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)


# The code to compute the rank-approximate neighbor
# for the given query and reference sets
add_executable(allkrann
  allkrann_main.cpp
)
target_link_libraries(allkrann
  mlpack
)

install(TARGETS allkrann RUNTIME DESTINATION bin)