macro (not_found_return message)
  message(STATUS "${message}")

  macro (add_markdown_docs name languages)
    # Do nothing.
  endmacro()

  return ()
endmacro ()

# We don't need to find any libraries or anything to generate markdown
# documentation.

# Move sources to the right directory.
set(SOURCES
  "binding_info.hpp"
  "binding_info.cpp"
  "default_param.hpp"
  "get_binding_name.hpp"
  "get_binding_name.cpp"
  "get_param.hpp"
  "get_printable_param.hpp"
  "get_printable_param_name.hpp"
  "get_printable_param_name_impl.hpp"
  "md_option.hpp"
  "print_doc_functions.hpp"
  "print_doc_functions_impl.hpp"
  "print_docs.hpp"
  "print_docs.cpp"
  "program_doc_wrapper.hpp"
)

add_custom_target(markdown_copy)
add_custom_command(TARGET markdown_copy PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        ${CMAKE_BINARY_DIR}/src/mlpack/bindings/markdown/)
foreach(file ${SOURCES})
  add_custom_command(TARGET markdown_copy PRE_BUILD
      COMMAND ${CMAKE_COMMAND} ARGS -E copy
          ${CMAKE_CURRENT_SOURCE_DIR}/${file}
          ${CMAKE_BINARY_DIR}/src/mlpack/bindings/markdown/)
endforeach()

# Create the add_markdown_docs() macro.  It's meant to be used as
# 'add_markdown_docs(knn, "cli;python;julia")', for instance.
macro (add_markdown_docs name languages)
  # First, we should filter any languages that aren't being compiled in the
  # build.  TODO: probably this requires modifying other CMakeLists.txts for
  # each of the bindings so that we know what languages are enabled and what
  # languages aren't.

  # Next, we should use configure_file() to generate each
  # generate_markdown.<binding>.cpp.
  set(BINDING ${name})
  set(LANGUAGES_PUSH_BACK_CODE "")
  set(PROGRAM_MAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${name}_main.cpp")
  foreach (lang ${languages})
    set(LANGUAGES_PUSH_BACK_CODE "${LANGUAGES_PUSH_BACK_CODE}\n  languages.push_back(\"${lang}\");")
  endforeach ()

  configure_file(${CMAKE_SOURCE_DIR}/src/mlpack/bindings/markdown/generate_markdown.binding.hpp.in
      ${CMAKE_BINARY_DIR}/src/mlpack/bindings/markdown/generate_markdown.${name}.hpp)
  configure_file(${CMAKE_SOURCE_DIR}/src/mlpack/bindings/markdown/generate_markdown.binding.cpp.in
      ${CMAKE_BINARY_DIR}/src/mlpack/bindings/markdown/generate_markdown.${name}.cpp)

  # Lastly, that generate_markdown.<binding>.cpp should be added to the list of
  # files to be compiled for the 'generate_markdown' target.
  set (MARKDOWN_SRCS ${MARKDOWN_SRCS}
      ${CMAKE_BINARY_DIR}/src/mlpack/bindings/markdown/generate_markdown.${name}.hpp
      ${CMAKE_BINARY_DIR}/src/mlpack/bindings/markdown/generate_markdown.${name}.cpp)
  set (MARKDOWN_SRCS ${MARKDOWN_SRCS} PARENT_SCOPE)
  set (MARKDOWN_NAMES ${MARKDOWN_NAMES} ${name})
  set (MARKDOWN_NAMES ${MARKDOWN_NAMES} PARENT_SCOPE)
endmacro ()

# After all the methods/ directories have been traversed, we can add the
# 'generate_markdown' target.
function (post_markdown_setup)
  # We need to generate the program file.
  message(STATUS "MARKDOWN_NAMES: ${MARKDOWN_NAMES}")
  foreach (name ${MARKDOWN_NAMES})
    set (MARKDOWN_INCLUDE_CODE "${MARKDOWN_INCLUDE_CODE}\n#include \"generate_markdown.${name}.hpp\"")
    set (MARKDOWN_HEADER_CODE "${MARKDOWN_HEADER_CODE}\n Print${name}Headers();")
    set (MARKDOWN_CALL_CODE "${MARKDOWN_CALL_CODE}\n  Print${name}Docs();")
  endforeach ()

  configure_file(${CMAKE_SOURCE_DIR}/src/mlpack/bindings/markdown/generate_markdown.cpp.in
      ${CMAKE_BINARY_DIR}/src/mlpack/bindings/markdown/generate_markdown.cpp)

  add_executable(generate_markdown
      "${CMAKE_BINARY_DIR}/src/mlpack/bindings/markdown/generate_markdown.cpp")
  target_link_libraries(generate_markdown mlpack ${MLPACK_LIBRARIES})
  add_dependencies(generate_markdown markdown_copy)
  set_target_properties(generate_markdown PROPERTIES
      COMPILE_FLAGS -DBINDING_TYPE=BINDING_TYPE_MARKDOWN
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/src/mlpack/bindings/markdown/)

  add_custom_target(markdown ALL DEPENDS generate_markdown)
  add_custom_command(TARGET markdown PRE_BUILD
      COMMAND ${CMAKE_COMMAND}
          -DPROGRAM=${CMAKE_BINARY_DIR}/src/mlpack/bindings/markdown/generate_markdown
          -DOUTPUT_FILE=${CMAKE_BINARY_DIR}/src/mlpack/bindings/markdown/mlpack.md
          -P ${CMAKE_SOURCE_DIR}/CMake/RunProgram.cmake
      COMMENT "Generating Markdown documentation for mlpack bindings...")
endfunction ()
