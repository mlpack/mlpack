cmake_minimum_required(VERSION 2.8)

# Define the files that we need to compile.
# Anything not in this list will not be compiled into MLPACK.
set(SOURCES
  cli.hpp
  cli_impl.hpp
  cli.cpp
  cli_deleter.hpp
  cli_deleter.cpp
  log.hpp
  log.cpp
  option.hpp
  option_impl.hpp
  option.cpp
  optionshierarchy.h
  optionshierarchy.cc
  nulloutstream.h
  nulloutstream.cc
  prefixedoutstream.h
  prefixedoutstream.cc
  prefixedoutstream_impl.h
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
    set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

# Append sources (with directory name) to list of all MLPACK sources (used at
# parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# test executable
add_executable(cli_test
   cli_test.cpp
)
# link dependencies of test executable
target_link_libraries(cli_test
   mlpack
   boost_program_options
   boost_unit_test_framework
)


