# This macro adds a command-line executable with the given name.  It assumes
# that the file with main() is in <name>_main.cpp, and produces an output
# program with the name mlpack_<name>.
macro (add_cli_executable directory name)
if (BUILD_CLI_EXECUTABLES)
  add_executable(mlpack_${name}
    ${directory}/${name}_main.cpp
  )
  # Build mlpack CLI binding binaries statically.
  if(NOT BUILD_SHARED_LIBS)
    target_link_libraries(mlpack_${name} -static
      ${MLPACK_LIBRARIES}
    )
  else()
    # Build mlpack CLI binding binaries dynamically.
    target_link_libraries(mlpack_${name}
      ${MLPACK_LIBRARIES}
    )
  endif()
  # Make sure that we set BINDING_TYPE to cli so the command-line program is
  # compiled with the correct int main() call.
  set_target_properties(mlpack_${name} PROPERTIES COMPILE_FLAGS
      -DBINDING_TYPE=BINDING_TYPE_CLI -DMLPACK_PRINT_INFO -DMLPACK_PRINT_WARN)
  install(TARGETS mlpack_${name} RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")

  # If man documentation is being generated, make sure this is a dependency.
  if (TXT2MAN)
    add_dependencies(man mlpack_${name})
  endif ()
endif ()
endmacro ()
