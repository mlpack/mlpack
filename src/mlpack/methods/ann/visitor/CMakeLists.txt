# Define the files we need to compile
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  add_visitor.hpp
  add_visitor_impl.hpp
  backward_visitor.hpp
  backward_visitor_impl.hpp
  delete_visitor.hpp
  delete_visitor_impl.hpp
  delta_visitor.hpp
  delta_visitor_impl.hpp
  deterministic_set_visitor.hpp
  deterministic_set_visitor_impl.hpp
  forward_visitor.hpp
  forward_visitor_impl.hpp
  gradient_set_visitor.hpp
  gradient_set_visitor_impl.hpp
  gradient_update_visitor.hpp
  gradient_update_visitor_impl.hpp
  gradient_visitor.hpp
  gradient_visitor_impl.hpp
  gradient_zero_visitor.hpp
  gradient_zero_visitor_impl.hpp
  load_output_parameter_visitor.hpp
  load_output_parameter_visitor_impl.hpp
  output_height_visitor.hpp
  output_height_visitor_impl.hpp
  output_parameter_visitor.hpp
  output_parameter_visitor_impl.hpp
  output_width_visitor.hpp
  output_width_visitor_impl.hpp
  parameters_set_visitor.hpp
  parameters_set_visitor_impl.hpp
  parameters_visitor.hpp
  parameters_visitor_impl.hpp
  reset_visitor.hpp
  reset_visitor_impl.hpp
  reward_set_visitor.hpp
  reward_set_visitor_impl.hpp
  save_output_parameter_visitor.hpp
  save_output_parameter_visitor_impl.hpp
  set_input_height_visitor.hpp
  set_input_height_visitor_impl.hpp
  set_input_width_visitor.hpp
  set_input_width_visitor_impl.hpp
  weight_set_visitor.hpp
  weight_set_visitor_impl.hpp
  weight_size_visitor.hpp
  weight_size_visitor_impl.hpp
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all mlpack sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
