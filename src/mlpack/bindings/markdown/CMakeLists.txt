macro (not_found_return message)
  message(STATUS "${message}")

  macro (add_markdown_docs directory name languages category)
    # Do nothing.
  endmacro()

  function (post_markdown_setup)
    # Do nothing.
  endfunction ()

  return ()
endmacro ()

if (NOT BUILD_MARKDOWN_BINDINGS)
  not_found_return("Not building Markdown bindings.")
endif ()

# We don't need to find any libraries or anything to generate markdown
# documentation.

# Add sources for Markdown bindings.
set(SOURCES
  "binding_info.hpp"
  "binding_info.cpp"
  "default_param.hpp"
  "get_param.hpp"
  "get_printable_param.hpp"
  "get_printable_param_name.hpp"
  "get_printable_param_name_impl.hpp"
  "get_printable_type.hpp"
  "md_option.hpp"
  "mlpack_main.hpp"
  "print_doc_functions.hpp"
  "print_doc_functions_impl.hpp"
  "print_docs.hpp"
  "print_docs.cpp"
  "print_param_table.hpp"
  "print_param_table.cpp"
  "print_type_doc.hpp"
  "program_doc_wrapper.hpp"
  "replace_all_copy.hpp"
)

# Copy all Markdown sources to the build directory.
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
# See the file 'Categories.cmake' for valid categories that can be used
# by the macro.
macro (add_markdown_docs directory name languages wrappers)

  set(category ${${name}_CATEGORY})

  # Next, we should use configure_file() to generate each
  # generate_markdown.<binding>.cpp.  We need to loop over all the languages for
  # this binding to do that.
  set(BINDING ${name})

  # pushes language names to a std::vector<string>.
  set(LANGUAGES_PUSH_BACK_CODE "")

  # pushes whether wrapper is to be added for this language
  # to a std::vector<bool>.
  set(WRAPPER_PUSH_BACK_CODE "")

  # pushes method names in a group to a std::vector<string>.
  set(VALID_METHODS_PUSH_BACK_CODE "")

  # includes _main.cpp files for all methods.
  set(INCLUDE_MAIN_FILES "")

  # used to create WRAPPER_PUSH_BACK_CODE.
  set(ADD_WRAPPER_LANGUAGES "")

  # list of languages for which wrapper is to be added.
  set(WRAPPERS ${wrappers})

  foreach (lang ${languages})
    set(LANGUAGES_PUSH_BACK_CODE
        "${LANGUAGES_PUSH_BACK_CODE}\n  languages.push_back(\"${lang}\");")
    set(MARKDOWN_ALL_LANGUAGES_LIST ${MARKDOWN_ALL_LANGUAGES_LIST} ${lang})
  endforeach ()
  list(REMOVE_DUPLICATES MARKDOWN_ALL_LANGUAGES_LIST)

  foreach (lang ${languages})
    list(FIND WRAPPERS "${lang}" _idx)
    if(${_idx} EQUAL -1)
      set(ADD_WRAPPER_LANGUAGES ${ADD_WRAPPER_LANGUAGES} "false")
    else()
      set(ADD_WRAPPER_LANGUAGES ${ADD_WRAPPER_LANGUAGES} "true")
    endif()
  endforeach()

  foreach (val ${ADD_WRAPPER_LANGUAGES})
    set(WRAPPER_PUSH_BACK_CODE
        "${WRAPPER_PUSH_BACK_CODE}\n  addWrapperDocs.push_back(${val});")
  endforeach()
  
  if(NOT ${WRAPPERS} EQUAL "")
    foreach(method ${${name}_VALID_METHODS})
      set(VALID_METHODS_PUSH_BACK_CODE
          "${VALID_METHODS_PUSH_BACK_CODE}\n  validMethods.push_back(\"${method}\");")
    endforeach()
    foreach(loc ${${name}_METHOD_MAIN_FILES})
      set(INCLUDE_MAIN_FILES
          "${INCLUDE_MAIN_FILES}\n#include <${loc}>")
    endforeach()
  endif()

  set(INCLUDE_MAIN_FILES
      "${INCLUDE_MAIN_FILES}\n#include <${CMAKE_CURRENT_SOURCE_DIR}/${directory}/${name}_main.cpp>")

  # Do the actual file configuration.
  set(BINDING_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/markdown)
  set(BINDING_BINARY_DIR ${CMAKE_BINARY_DIR}/src/mlpack/bindings/markdown)

  configure_file(${BINDING_SOURCE_DIR}/generate_markdown.binding.hpp.in
      ${BINDING_BINARY_DIR}/generate_markdown.${name}.hpp)
  configure_file(${BINDING_SOURCE_DIR}/generate_markdown.binding.cpp.in
      ${BINDING_BINARY_DIR}/generate_markdown.${name}.cpp)

  # Lastly, that generate_markdown.<binding>.cpp should be added to the list of
  # files to be compiled for the 'generate_markdown' target.  We also need to
  # add information about this binding to a set of variables we have to track.
  set (MARKDOWN_SRCS ${MARKDOWN_SRCS}
      ${CMAKE_BINARY_DIR}/src/mlpack/bindings/markdown/generate_markdown.${name}.hpp
      ${CMAKE_BINARY_DIR}/src/mlpack/bindings/markdown/generate_markdown.${name}.cpp)
  set (MARKDOWN_SRCS ${MARKDOWN_SRCS} PARENT_SCOPE)
  set (MARKDOWN_NAMES ${MARKDOWN_NAMES} ${name})
  set (MARKDOWN_NAMES ${MARKDOWN_NAMES} PARENT_SCOPE)
  set (MARKDOWN_NAME_CATEGORIES ${MARKDOWN_NAME_CATEGORIES} ${category})
  set (MARKDOWN_NAME_CATEGORIES ${MARKDOWN_NAME_CATEGORIES} PARENT_SCOPE)
  set (MARKDOWN_ALL_LANGUAGES_LIST ${MARKDOWN_ALL_LANGUAGES_LIST} PARENT_SCOPE)
  set (ADD_WRAPPER_LANGUAGES ${ADD_WRAPPER_LANGUAGES} PARENT_SCOPE)
endmacro ()

# After all the methods/ directories have been traversed, we can add the
# 'generate_markdown' target.  This function is run at the bottom of
# methods/CMakeLists.txt.
function (post_markdown_setup)
  # We need to generate the program file.  This consists of generating three
  # things:
  #
  # - MARKDOWN_INCLUDES: the list of files to be included.
  # - MARKDOWN_HEADER_CODE: the code used to print the header/sidebar.
  # - MARKDOWN_CALL_CODE: the code to actually call the functions to print
  #       documentation for each binding.

  # Iterate over categories of binding.
  list(LENGTH MARKDOWN_NAMES NUM_MARKDOWN_BINDINGS)
  list(LENGTH CATEGORIES NUM_MARKDOWN_CATEGORIES)
  math(EXPR cat_limit "${NUM_MARKDOWN_CATEGORIES} - 1")
  foreach (i RANGE ${cat_limit})
    list (GET CATEGORIES ${i} cat)
    # Put the things in this category in a div.
    string(CONCAT header_code "${MARKDOWN_HEADER_CODE}  cout << "
        "\"<div class=\\\"category\\\" markdown=\\\"1\\\">\" << endl;\n  "
        "cout << \"<h5>${cat}:</h5>\" << endl;\n")
    set(MARKDOWN_HEADER_CODE ${header_code})

    # Add an option for this binding.
    math(EXPR range_limit "${NUM_MARKDOWN_BINDINGS} - 1")
    foreach (j RANGE ${range_limit})
      list (GET MARKDOWN_NAME_CATEGORIES ${j} category)
      if (NOT category STREQUAL cat)
        continue ()
      endif ()

      list (GET MARKDOWN_NAMES ${j} name)
      set (MARKDOWN_HEADER_CODE
          "${MARKDOWN_HEADER_CODE}\n  Print${name}Headers();")
    endforeach ()

    set (MARKDOWN_HEADER_CODE
        "${MARKDOWN_HEADER_CODE}\n  cout << \"</div>\" << endl << endl;")
  endforeach ()

  foreach (name ${MARKDOWN_NAMES})
    set (MARKDOWN_INCLUDE_CODE
        "${MARKDOWN_INCLUDE_CODE}\n#include \"generate_markdown.${name}.hpp\"")
    set (MARKDOWN_CALL_CODE "${MARKDOWN_CALL_CODE}\n  Print${name}Docs();")
  endforeach ()

  set(BINDING_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/markdown)
  set(BINDING_BINARY_DIR ${CMAKE_BINARY_DIR}/src/mlpack/bindings/markdown)

  configure_file(${BINDING_SOURCE_DIR}/generate_markdown.cpp.in
      ${BINDING_BINARY_DIR}/generate_markdown.cpp)

  # Remember that this is being run from some other directory, so we have to be
  # explicit with the locations of the files we are compiling against.
  add_executable(generate_markdown
      "${BINDING_BINARY_DIR}/generate_markdown.cpp"
      ${MARKDOWN_SRCS}
      "${BINDING_SOURCE_DIR}/binding_info.hpp"
      "${BINDING_SOURCE_DIR}/binding_info.cpp"
      "${BINDING_SOURCE_DIR}/default_param.hpp"
      "${BINDING_SOURCE_DIR}/get_param.hpp"
      "${BINDING_SOURCE_DIR}/get_printable_param.hpp"
      "${BINDING_SOURCE_DIR}/get_printable_param_name.hpp"
      "${BINDING_SOURCE_DIR}/get_printable_param_name_impl.hpp"
      "${BINDING_SOURCE_DIR}/md_option.hpp"
      "${BINDING_SOURCE_DIR}/print_doc_functions.hpp"
      "${BINDING_SOURCE_DIR}/print_doc_functions_impl.hpp"
      "${BINDING_SOURCE_DIR}/print_docs.hpp"
      "${BINDING_SOURCE_DIR}/print_docs.cpp"
      "${BINDING_SOURCE_DIR}/print_param_table.hpp"
      "${BINDING_SOURCE_DIR}/print_param_table.cpp"
      "${BINDING_SOURCE_DIR}/program_doc_wrapper.hpp"
      "${BINDING_SOURCE_DIR}/replace_all_copy.hpp"
      "${BINDING_SOURCE_DIR}/generate_markdown.cpp")
  target_link_libraries(generate_markdown ${MLPACK_LIBRARIES})
  add_dependencies(generate_markdown markdown_copy)
  set_target_properties(generate_markdown PROPERTIES
      COMPILE_FLAGS -DBINDING_TYPE=BINDING_TYPE_MARKDOWN
      RUNTIME_OUTPUT_DIRECTORY ${BINDING_BINARY_DIR})

  add_custom_target(markdown ALL
      ${CMAKE_COMMAND} -E make_directory
          ${CMAKE_BINARY_DIR}/doc/
      COMMAND ${CMAKE_COMMAND}
          -DPROGRAM=${BINDING_BINARY_DIR}/generate_markdown
          -DOUTPUT_FILE=${CMAKE_BINARY_DIR}/doc/mlpack.md
          -P ${CMAKE_SOURCE_DIR}/CMake/RunProgram.cmake
      COMMAND ${CMAKE_COMMAND} -E copy
          ${BINDING_SOURCE_DIR}/res/change_language.js
          ${CMAKE_BINARY_DIR}/doc/res/change_languages.js
      COMMAND ${CMAKE_COMMAND} -E copy
          ${BINDING_SOURCE_DIR}/res/menu_bg.png
          ${CMAKE_BINARY_DIR}/doc/res/menu_bg.png
      COMMAND ${CMAKE_COMMAND} -E copy
          ${BINDING_SOURCE_DIR}/res/formatting.css
          ${CMAKE_BINARY_DIR}/doc/res/formatting.css
      DEPENDS generate_markdown
      COMMENT "Generating Markdown documentation for mlpack bindings...")
endfunction ()
