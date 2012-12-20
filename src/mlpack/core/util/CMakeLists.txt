# Define the files we need to compile.
# Anything not in this list will not be compiled into MLPACK.
set(SOURCES
  cli.hpp
  cli.cpp
  cli_deleter.hpp
  cli_deleter.cpp
  cli_impl.hpp
  log.hpp
  log.cpp
  nulloutstream.hpp
  option.hpp
  option.cpp
  option_impl.hpp
  prefixedoutstream.hpp
  prefixedoutstream.cpp
  prefixedoutstream_impl.hpp
  save_restore_utility.hpp
  save_restore_utility.cpp
  save_restore_utility_impl.hpp
  sfinae_utility.hpp
  string_util.hpp
  string_util.cpp
  timers.hpp
  timers.cpp
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all MLPACK sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
