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
  print_type_doc.hpp
  print_type_doc_impl.hpp
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

# This macro adds a command-line executable with the given name.  It assumes
# that the file with main() is in <name>_main.cpp, and produces an output
# program with the name mlpack_<name>.
macro (add_cli_executable name)
if (BUILD_CLI_EXECUTABLES)
  add_executable(mlpack_${name}
    ${name}_main.cpp
  )
  target_link_libraries(mlpack_${name}
    mlpack
    ${ARMADILLO_LIBRARIES}
    ${Boost_LIBRARIES}
    ${COMPILER_SUPPORT_LIBRARIES}
  )
  # Make sure that we set BINDING_TYPE to cli so the command-line program is
  # compiled with the correct int main() call.
  set_target_properties(mlpack_${name} PROPERTIES COMPILE_FLAGS
      -DBINDING_TYPE=BINDING_TYPE_CLI)
  install(TARGETS mlpack_${name} RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")

  # If man documentation is being generated, make sure this is a dependency.
  if (TXT2MAN)
    add_dependencies(man mlpack_${name})
  endif ()
endif ()
endmacro ()
