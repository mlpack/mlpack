# Define the files we need to compile
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  add.hpp
  add_impl.hpp
  add_merge.hpp
  add_merge_impl.hpp
  base_layer.hpp
  concat.hpp
  concat_impl.hpp
  concat_performance.hpp
  concat_performance_impl.hpp
  constant.hpp
  constant_impl.hpp
  convolution.hpp
  convolution_impl.hpp
  dropconnect.hpp
  dropconnect_impl.hpp
  dropout.hpp
  dropout_impl.hpp
  glimpse.hpp
  glimpse_impl.hpp
  hard_tanh.hpp
  hard_tanh_impl.hpp
  join.hpp
  join_impl.hpp
  layer.hpp
  layer_traits.hpp
  layer_visitor.hpp
  layer_visitor_impl.hpp
  layer_types.hpp
  leaky_relu.hpp
  leaky_relu_impl.hpp
  linear.hpp
  linear_impl.hpp
  linear_no_bias.hpp
  linear_no_bias_impl.hpp
  log_softmax.hpp
  log_softmax_impl.hpp
  lookup.hpp
  lookup_impl.hpp
  lstm.hpp
  lstm_impl.hpp
  max_pooling.hpp
  max_pooling_impl.hpp
  mean_pooling.hpp
  mean_pooling_impl.hpp
  mean_squared_error.hpp
  mean_squared_error_impl.hpp
  multiply_constant.hpp
  multiply_constant_impl.hpp
  negative_log_likelihood.hpp
  negative_log_likelihood_impl.hpp
  recurrent.hpp
  recurrent_impl.hpp
  recurrent_attention.hpp
  recurrent_attention_impl.hpp
  reinforce_normal.hpp
  reinforce_normal_impl.hpp
  select.hpp
  select_impl.hpp
  sequential.hpp
  sequential_impl.hpp
  vr_class_reward_impl.hpp
  vr_class_reward_impl.hpp
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all mlpack sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
