# Define the files we need to compile
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  async_learning.hpp
  async_learning_impl.hpp
  q_learning.hpp
  q_learning_impl.hpp
  sac.hpp
  sac_impl.hpp
  training_config.hpp
)

add_subdirectory(environment)
add_subdirectory(policy)
add_subdirectory(replay)
add_subdirectory(worker)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all mlpack sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
