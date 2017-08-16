# Define the files we need to compile
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  accuracy.hpp
  accuracy_impl.hpp
  average_strategy.hpp
  f1.hpp
  f1_impl.hpp
  facilities.hpp
  mse.hpp
  mse_impl.hpp
  precision.hpp
  precision_impl.hpp
  recall.hpp
  recall_impl.hpp
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all mlpack sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
