# Define the files we need to compile
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  cnn.hpp
  cnn_impl.hpp
  ffn.hpp
  ffn_impl.hpp
  network_util.hpp
  network_util_impl.hpp
  rnn.hpp
  rnn_impl.hpp
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all mlpack sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_subdirectory(activation_functions)
add_subdirectory(init_rules)
add_subdirectory(layer)
add_subdirectory(performance_functions)
add_subdirectory(pooling_rules)
add_subdirectory(convolution_rules)
