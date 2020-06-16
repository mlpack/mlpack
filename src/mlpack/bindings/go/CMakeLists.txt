macro (not_found_return message)
  message(STATUS "${message}")
  if (NOT BUILD_GO_SHLIB)
    macro (add_go_binding name)
      # Do nothing.
    endmacro ()

    return()
  endif()
endmacro ()

# If we are not supposed to make Go bindings, define the macro so it does
# nothing and leave this file.
if (NOT BUILD_GO_BINDINGS)
  not_found_return("Not building Go bindings.")
endif ()

if (BUILD_GO_BINDINGS)
  ## We need to check here if Golang is even available.  Although actually
  ## technically, I'm not sure if we even need to know!  For the tests though we
  ## do.  So it's probably a good idea to check.
  if (FORCE_BUILD_GO_BINDINGS)
    find_package(Go 1.11.0)
    find_package(Gonum)
    if (NOT GO_FOUND OR NOT GONUM_FOUND)
      unset(BUILD_GO_BINDINGS CACHE)
      message(FATAL_ERROR "Go or Gonum not found; unable to build Go bindings!")
    endif()
  else ()
    find_package(Go 1.11.0)
    find_package(Gonum)
    if (NOT GO_FOUND OR NOT GONUM_FOUND)
      unset(BUILD_GO_BINDINGS CACHE)
    endif()
  endif ()

  if (NOT GO_FOUND)
    not_found_return("Go not found; not building Go bindings.")
  endif ()

  if (NOT GONUM_FOUND)
    not_found_return("Gonum not found; not building Go bindings.")
  endif ()

  add_custom_target(go)

  # Create model.go with package defination.
  file(WRITE
      "${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/src/mlpack.org/v1/mlpack/models.go"
      "package mlpack"
      "\n\n"
      "/*\n")
endif()

if (BUILD_GO_SHLIB)

  # Nothing in this directory will be compiled into mlpack.
  set(BINDING_SOURCES
    get_type.hpp
    get_param.hpp
    get_printable_param.hpp
    go_option.hpp
    mlpack/arma_util.h
    mlpack/arma_util.hpp
    mlpack/cli_util.h
    mlpack/cli_util.hpp
    print_class_defn.hpp
    print_cpp.cpp
    print_cpp.hpp
    print_defn_input.hpp
    print_defn_output.hpp
    print_doc.hpp
    print_doc_functions.hpp
    print_doc_functions_impl.hpp
    print_go.hpp
    print_go.cpp
    print_h.hpp
    print_h.cpp
    print_input_processing.hpp
    print_method_config.hpp
    print_method_init.hpp
    print_output_processing.hpp
    camel_case.hpp
    strip_type.hpp
  )

  # These are all the files we need to compile Go bindings for mlpack that are
  # not a part of mlpack itself.
  set(CGO_SOURCES
    mlpack/arma_util.go
    mlpack/cli_util.go
    mlpack/doc.go
    mlpack/numcsv.go
    mlpack/go.mod
    mlpack/go.sum
  )

  # These are all the files we need to compile Go bindings for mlpack that are
  # not a part of mlpack itself.
  set(CAPI_SOURCES
    mlpack/capi/arma_util.h
    mlpack/capi/cli_util.h
  )

  set(TEST_SOURCES
    tests/go_binding_test.go
  )
  add_custom_target(go_shlib ALL DEPENDS mlpack)
  add_custom_target(go_copy ALL DEPENDS mlpack)

  # Copy necessary files after making the mlpack/ directory.
  add_custom_command(TARGET go_copy PRE_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory
          ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/src/mlpack.org/v1/mlpack/
      COMMAND ${CMAKE_COMMAND} -E make_directory
          ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/src/mlpack.org/v1/mlpack/capi/
      COMMAND ${CMAKE_COMMAND} -E make_directory
          ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/src/mlpack.org/v1/mlpack/tests/)

  foreach(go_file ${CAPI_SOURCES})
  add_custom_command(TARGET go_copy PRE_BUILD
      COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different
          ${CMAKE_CURRENT_SOURCE_DIR}/${go_file}
          ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/src/mlpack.org/v1/mlpack/capi/)
  endforeach()

  add_custom_command(TARGET go_copy PRE_BUILD
      COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different
          $<TARGET_FILE:mlpack>
          ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/src/mlpack.org/v1/mlpack/)

  foreach(cgo_file ${CGO_SOURCES})
  add_custom_command(TARGET go_copy PRE_BUILD
      COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different
          ${CMAKE_CURRENT_SOURCE_DIR}/${cgo_file}
          ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/src/mlpack.org/v1/mlpack/)
  endforeach()

  if (BUILD_TESTS OR BUILD_GO_SHLIB)
    foreach(test_file ${TEST_SOURCES})
      add_custom_command(TARGET go_copy PRE_BUILD
          COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different
              ${CMAKE_CURRENT_SOURCE_DIR}/${test_file}
              ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/src/mlpack.org/v1/mlpack/tests/)
    endforeach ()
  endif ()

  add_library(go_util SHARED
          ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/go/mlpack/capi/arma_util.cpp
          ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/go/mlpack/capi/cli_util.cpp)
  target_link_libraries(go_util mlpack ${MLPACK_LIBRARIES})
  target_compile_definitions(go_util PUBLIC "BINDING_TYPE=BINDING_TYPE_GO")
  set_target_properties(go_util PROPERTIES
     LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/src/mlpack.org/v1/mlpack/)

  # Set the include directories correctly.
  get_property(GO_INCLUDE_DIRECTORIES DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
     PROPERTY INCLUDE_DIRECTORIES)
  set (GO_INCLDIRS "${GO_INCLUDE_DIRECTORIES}")

  install(TARGETS go_util
      RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
      LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
      ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
endif()

# Define a global list of models.
define_property(GLOBAL PROPERTY GO_MODELS
    BRIEF_DOCS "Global list of models"
    FULL_DOCS "Global list of models"
)

# Initialize list.
set_property(GLOBAL PROPERTY GO_MODELS "")

# Add a macro to build a go binding.
macro (add_go_binding name)
if (BUILD_GO_BINDINGS)

  # Include all .h that define model to models.go. 
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/${name}_main.cpp" MAIN_FILE)
  if (MAIN_FILE MATCHES "PARAM_MODEL")
    file(APPEND
        "${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/src/mlpack.org/v1/mlpack/models.go"
        "#include <capi/${name}.h>\n")
  endif()

  # Append content to the list.
  set_property(GLOBAL APPEND PROPERTY GO_MODELS ${CMAKE_CURRENT_SOURCE_DIR}/${name}_main.cpp)

  # Create .h file for C API, e.g. pca.h.
  add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/generate_h_${name}.cpp
     COMMAND ${CMAKE_COMMAND}
         -DGENERATE_BINDING_IN=${CMAKE_SOURCE_DIR}/src/mlpack/bindings/go/generate_h.cpp.in
         -DGENERATE_BINDING_OUT=${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/generate_h_${name}.cpp
         -DPROGRAM_MAIN_FILE=${CMAKE_CURRENT_SOURCE_DIR}/${name}_main.cpp
         -DPROGRAM_NAME=${name}
         -P ${CMAKE_SOURCE_DIR}/CMake/ConfigureGoHCPP.cmake
     DEPENDS ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/go/generate_h.cpp.in)

  add_executable(generate_h_${name}
      ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/generate_h_${name}.cpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/go/print_h.hpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/go/print_h.cpp)
  target_link_libraries(generate_h_${name} mlpack ${MLPACK_LIBRARIES})
  set_target_properties(generate_h_${name} PROPERTIES COMPILE_FLAGS
      -DBINDING_TYPE=BINDING_TYPE_GO)
  add_custom_command(TARGET generate_h_${name} POST_BUILD
      COMMAND ${CMAKE_COMMAND}
          -DGENERATE_BINDING_PROGRAM=${CMAKE_BINARY_DIR}/bin/generate_h_${name}
          -DBINDING_OUTPUT_FILE=${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/src/mlpack.org/v1/mlpack/capi/${name}.h
          -P ${CMAKE_SOURCE_DIR}/CMake/GenerateGoBinding.cmake)

  # Create .go file, pca.go.
  add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/generate_go_${name}.cpp
     COMMAND ${CMAKE_COMMAND}
         -DGENERATE_BINDING_IN=${CMAKE_SOURCE_DIR}/src/mlpack/bindings/go/generate_go.cpp.in
         -DGENERATE_BINDING_OUT=${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/generate_go_${name}.cpp
         -DPROGRAM_MAIN_FILE=${CMAKE_CURRENT_SOURCE_DIR}/${name}_main.cpp
         -DPROGRAM_NAME=${name}
         -P ${CMAKE_SOURCE_DIR}/CMake/ConfigureGoHCPP.cmake
     DEPENDS ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/go/generate_go.cpp.in)

  add_executable(generate_go_${name}
      ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/generate_go_${name}.cpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/go/print_go.hpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/go/print_go.cpp)
  target_link_libraries(generate_go_${name} mlpack ${MLPACK_LIBRARIES})
  set_target_properties(generate_go_${name} PROPERTIES COMPILE_FLAGS
      -DBINDING_TYPE=BINDING_TYPE_GO)
  add_custom_command(TARGET generate_go_${name} POST_BUILD
      COMMAND ${CMAKE_COMMAND}
          -DGENERATE_BINDING_PROGRAM=${CMAKE_BINARY_DIR}/bin/generate_go_${name}
          -DBINDING_OUTPUT_FILE=${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/src/mlpack.org/v1/mlpack/${name}.go
          -P ${CMAKE_SOURCE_DIR}/CMake/GenerateGoBinding.cmake)

  add_dependencies(generate_h_${name} generate_go_${name})
  add_dependencies(go generate_h_${name})
endif ()

if(BUILD_GO_SHLIB)
  # Create .cpp file for C API, e.g. pca.cpp.
  add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/build/generate_cpp_${name}.cpp
     COMMAND ${CMAKE_COMMAND}
         -DGENERATE_BINDING_IN=${CMAKE_SOURCE_DIR}/src/mlpack/bindings/go/generate_cpp.cpp.in
         -DGENERATE_BINDING_OUT=${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/build/generate_cpp_${name}.cpp
         -DPROGRAM_MAIN_FILE=${CMAKE_CURRENT_SOURCE_DIR}/${name}_main.cpp
         -DPROGRAM_NAME=${name}
         -P ${CMAKE_SOURCE_DIR}/CMake/ConfigureGoHCPP.cmake
     DEPENDS ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/go/generate_cpp.cpp.in)

  add_executable(generate_cpp_${name}
      ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/build/generate_cpp_${name}.cpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/go/print_cpp.hpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/go/print_cpp.cpp)
  target_link_libraries(generate_cpp_${name} mlpack ${MLPACK_LIBRARIES})
  set_target_properties(generate_cpp_${name} PROPERTIES COMPILE_FLAGS
      -DBINDING_TYPE=BINDING_TYPE_GO)
  add_custom_command(TARGET generate_cpp_${name} POST_BUILD
      COMMAND ${CMAKE_COMMAND}
          -DGENERATE_BINDING_PROGRAM=${CMAKE_BINARY_DIR}/bin/generate_cpp_${name}
          -DBINDING_OUTPUT_FILE=${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/build/${name}.cpp
          -P ${CMAKE_SOURCE_DIR}/CMake/GenerateGoBinding.cmake)

  add_dependencies(generate_cpp_${name} go_copy)

  # Create, e.g., libmlpack_go_pca.so.
  add_library(mlpack_go_${name} SHARED
      ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/build/${name}.cpp)
  set_source_files_properties(${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/build/${name}.cpp
      PROPERTIES GENERATED TRUE)
  target_link_libraries(mlpack_go_${name} mlpack ${MLPACK_LIBRARIES})
  target_compile_definitions(mlpack_go_${name} PUBLIC "BINDING_TYPE=BINDING_TYPE_GO")
  set_target_properties(mlpack_go_${name} PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/src/mlpack.org/v1/mlpack/)

  install(TARGETS mlpack_go_${name}
      RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
      LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
      ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")

  add_dependencies(mlpack_go_${name} generate_cpp_${name})
  add_dependencies(mlpack_go_${name} go_util)
  add_dependencies(go_shlib mlpack_go_${name})

  if (BUILD_GO_BINDINGS)
    add_dependencies(mlpack_go_${name} generate_h_${name})
    add_dependencies(go mlpack_go_${name})
  endif()
endif()

endmacro ()

if (BUILD_TESTS OR BUILD_GO_SHLIB)
  add_subdirectory(tests)
endif ()
