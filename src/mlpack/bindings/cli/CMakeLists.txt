# Define the files we need to compile.
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  add_to_po.hpp
  cli_option.hpp
  default_param.hpp
  default_param_impl.hpp
  delete_allocated_memory.hpp
  end_program.hpp
  get_allocated_memory.hpp
  get_param.hpp
  get_raw_param.hpp
  get_printable_param.hpp
  get_printable_param_impl.hpp
  get_printable_param_name.hpp
  get_printable_param_name_impl.hpp
  get_printable_param_value.hpp
  get_printable_param_value_impl.hpp
  map_parameter_name.hpp
  output_param.hpp
  output_param_impl.hpp
  parameter_type.hpp
  parse_command_line.hpp
  print_doc_functions.hpp
  print_doc_functions_impl.hpp
  print_help.hpp
  print_help.cpp
  set_param.hpp
  string_type_param.hpp
  string_type_param_impl.hpp
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append source (with directory name) to list of all mlpack sources (used at the
# parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
