cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile.
# Anything not in this list will not be compiled into MLPACK.
set(SOURCES
  ip_metric.hpp
  ip_metric_impl.hpp
  max_ip.hpp
  max_ip_impl.hpp
  max_ip_rules.hpp
  max_ip_rules_impl.hpp
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all MLPACK sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_executable(maxip
  max_ip_main.cpp
)
target_link_libraries(maxip
  mlpack
)

install(TARGETS maxip RUNTIME DESTINATION bin)
