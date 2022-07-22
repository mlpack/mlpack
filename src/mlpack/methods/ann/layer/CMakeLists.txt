# Define the files we need to compile
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  adaptive_max_pooling.hpp
  adaptive_max_pooling_impl.hpp
  adaptive_mean_pooling.hpp
  adaptive_mean_pooling_impl.hpp
  add.hpp
  add_impl.hpp
  add_merge.hpp
  add_merge_impl.hpp
  alpha_dropout.hpp
  alpha_dropout_impl.hpp
  base_layer.hpp
  celu.hpp
  celu_impl.hpp
  concat.hpp
  concat_impl.hpp
  concatenate.hpp
  concatenate_impl.hpp
  convolution.hpp
  convolution_impl.hpp
  dropconnect.hpp
  dropconnect_impl.hpp
  dropout.hpp
  dropout_impl.hpp
  elu.hpp
  elu_impl.hpp
  identity.hpp
  identity_impl.hpp
  layer.hpp
  layer_types.hpp
  leaky_relu.hpp
  leaky_relu_impl.hpp
  linear.hpp
  linear_impl.hpp
  linear_no_bias.hpp
  linear_no_bias_impl.hpp
  linear3d.hpp
  linear3d_impl.hpp
  log_softmax.hpp
  log_softmax_impl.hpp
  lstm.hpp
  lstm_impl.hpp
  max_pooling.hpp
  max_pooling_impl.hpp
  mean_pooling.hpp
  mean_pooling_impl.hpp
  multi_layer.hpp
  multi_layer_impl.hpp
  noisylinear.hpp
  noisylinear_impl.hpp
  padding.hpp
  radial_basis_function.hpp
  radial_basis_function_impl.hpp
  serialization.hpp
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all mlpack sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
