# Define the files we need to compile.
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  arma_traits.hpp
  backtrace.hpp
  backtrace_impl.hpp
  binding_details.hpp
  forward.hpp
  io.hpp
  io_impl.hpp
  deprecated.hpp
  hyphenate_string.hpp
  is_std_vector.hpp
  log.hpp
  log_impl.hpp
  mlpack_main.hpp
  nulloutstream.hpp
  param.hpp
  param_checks.hpp
  param_checks_impl.hpp
  param_data.hpp
  params.hpp
  params_impl.hpp
  prefixedoutstream.hpp
  prefixedoutstream_impl.hpp
  program_doc.hpp
  program_doc_impl.hpp
  size_checks.hpp
  sfinae_utility.hpp
  timers.hpp
  timers_impl.hpp
  to_lower.hpp
  version.hpp
  version_impl.hpp
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all mlpack sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
