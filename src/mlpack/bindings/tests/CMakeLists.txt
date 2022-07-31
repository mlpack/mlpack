# Define the files we need to compile into the test executable.
# Anything not in this list will not be compiled into mlpack_test.
set(BINDING_SOURCES
  clean_memory.hpp
  clean_memory.cpp
  test_option.hpp
  ignore_check.hpp
  delete_allocated_memory.hpp
  get_allocated_memory.hpp
  get_param.hpp
  get_printable_param.hpp
  get_printable_param_impl.hpp
  mlpack_main.hpp
  test_function_map.hpp
  test_function_map.cpp
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${BINDING_SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append source (with directory name) to list of all mlpack sources (used at the
# parent scope).
set(MLPACK_TEST_SRCS ${MLPACK_TEST_SRCS} ${DIR_SRCS} PARENT_SCOPE)
