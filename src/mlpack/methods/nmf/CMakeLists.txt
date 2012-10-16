# Define the files we need to compile
# Anything not in this list will not be compiled into MLPACK.
set(SOURCES
  mult_dist_update_rules.hpp
  mult_div_update_rules.hpp
  als_update_rules.hpp
  random_init.hpp
  random_acol_init.hpp
  nmf.hpp
  nmf_impl.hpp
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all MLPACK sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_executable(nmf
  nmf_main.cpp
)
target_link_libraries(nmf
  mlpack
)
install(TARGETS nmf RUNTIME DESTINATION bin)
