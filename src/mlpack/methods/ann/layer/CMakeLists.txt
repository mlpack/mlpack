# Define the files we need to compile
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  layer_traits.hpp
  binary_classification_layer.hpp
  base_layer.hpp
  empty_layer.hpp
  bias_layer.hpp
  dropout_layer.hpp
  dropconnect_layer.hpp
  hard_tanh_layer.hpp
  leaky_relu_layer.hpp
  linear_layer.hpp
  conv_layer.hpp
  pooling_layer.hpp
  recurrent_layer.hpp
  lstm_layer.hpp
  sparse_bias_layer.hpp
  sparse_input_layer.hpp
  sparse_output_layer.hpp
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all mlpack sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
