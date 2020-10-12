macro (not_found_return message)
  message(STATUS "${message}")
  if (NOT BUILD_GO_SHLIB)
    macro (add_go_binding name)
      # Do nothing.
    endmacro ()

    return()
  endif()
endmacro ()

macro (post_go_setup)
  if (BUILD_GO_BINDINGS)
    # Once `GO_MODELS` is populated, generate `models.go`.
    file(APPEND
        "${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/src/mlpack.org/v1/mlpack/models.go"
        "*/\n"
        "import \"C\"\n\n"
        "import (\n"
        "  \"runtime\"\n"
        "  \"unsafe\"\n"
        ")\n\n")

    include("${CMAKE_SOURCE_DIR}/CMake/go/AppendModel.cmake")
    # Read list content.
    get_property(MODELS GLOBAL PROPERTY GO_MODELS)
    foreach (models IN LISTS MODELS)
      append_model(
        "${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/src/mlpack.org/v1/mlpack/models.go"
         ${models})
    endforeach()
  endif()
endmacro ()

# If we are not supposed to make Go bindings, define the macro so it does
# nothing and leave this file.
if (NOT BUILD_GO_BINDINGS)
  not_found_return("Not building Go bindings.")
endif ()

if (BUILD_GO_BINDINGS)

  find_package(Go 1.11.0)
  if (NOT GO_FOUND)
    set(GO_NOT_FOUND_MSG "${GO_NOT_FOUND_MSG}\n    - Go")
  endif ()
  find_package(Gonum)
  if (NOT GONUM_FOUND)
    set(GO_NOT_FOUND_MSG "${GO_NOT_FOUND_MSG}\n    - Gonum")
  endif ()

  ## We need to check here if Golang is even available.  Although actually
  ## technically, I'm not sure if we even need to know!  For the tests though we
  ## do.  So it's probably a good idea to check.
  if (FORCE_BUILD_GO_BINDINGS)
    if (NOT GO_FOUND OR NOT GONUM_FOUND)
      unset(BUILD_GO_BINDINGS CACHE)
      set(BUILD_GO_SHLIB OFF)
      message(FATAL_ERROR "\nCould not Build Go Bindings; the following modules are not available: ${GO_NOT_FOUND_MSG}")
    endif()
  else ()
    if (NOT GO_FOUND OR NOT GONUM_FOUND)
      unset(BUILD_GO_BINDINGS CACHE)
      set(BUILD_GO_SHLIB OFF)
      not_found_return("Not building Go bindings; the following modules are not available: ${GO_NOT_FOUND_MSG}")
    endif()
  endif ()

  add_custom_target(go)

  # All the bindings will build under "src/mlpack.org/v1/mlpack";  So if user build
  # go-bindings from source, then he/she can use the same import path as documented
  # on "mlpack.org" by setting GOPATH=/path/to/mlpack/build/src/mlpack/bindings/go.
  # Create model.go with package definition.
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
    mlpack/io_util.h
    mlpack/io_util.hpp
    print_class_defn.hpp
    print_defn_input.hpp
    print_defn_output.hpp
    print_doc.hpp
    print_doc_functions.hpp
    print_doc_functions_impl.hpp
    print_go.hpp
    print_go.cpp
    print_input_processing.hpp
    print_method_config.hpp
    print_method_init.hpp
    print_output_processing.hpp
    strip_type.hpp
  )

  # These are all the files we need to compile Go bindings for mlpack that are
  # not a part of mlpack itself.
  set(CGO_SOURCES
    mlpack/arma_util.go
    mlpack/io_util.go
    mlpack/doc.go
    mlpack/numcsv.go
    mlpack/go.mod
    mlpack/go.sum
  )

  # These are all the files we need to compile Go bindings for mlpack that are
  # not a part of mlpack itself.
  set(CAPI_SOURCES
    mlpack/capi/arma_util.h
    mlpack/capi/io_util.h
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

  add_library(mlpack_go_util SHARED
          ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/go/mlpack/capi/arma_util.cpp
          ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/go/mlpack/capi/io_util.cpp)
  target_link_libraries(mlpack_go_util mlpack ${MLPACK_LIBRARIES})
  target_compile_definitions(mlpack_go_util PUBLIC "BINDING_TYPE=BINDING_TYPE_GO")
  set_target_properties(mlpack_go_util PROPERTIES
     LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/src/mlpack.org/v1/mlpack/)

  # Set the include directories correctly.
  get_property(GO_INCLUDE_DIRECTORIES DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
     PROPERTY INCLUDE_DIRECTORIES)
  set (GO_INCLDIRS "${GO_INCLUDE_DIRECTORIES}")

  install(TARGETS mlpack_go_util
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

  # Create ${name}.h.
  add_custom_command(OUTPUT
      ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/src/mlpack.org/v1/mlpack/capi/${name}.h
      COMMAND ${CMAKE_COMMAND}
          -DPROGRAM_NAME=${name}
          -DPROGRAM_MAIN_FILE=${CMAKE_CURRENT_SOURCE_DIR}/${name}_main.cpp
          -DSOURCE_DIR=${CMAKE_SOURCE_DIR}
          -DGO_IN=${CMAKE_SOURCE_DIR}/src/mlpack/bindings/go/go_method.h.in
          -DGO_OUT=${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/src/mlpack.org/v1/mlpack/capi/${name}.h
          -P ${CMAKE_SOURCE_DIR}/CMake/go/ConfigureGoHCPP.cmake
      DEPENDS ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/go/go_method.h.in
              ${CMAKE_SOURCE_DIR}/CMake/go/ConfigureGoHCPP.cmake)

  # Create .go file, pca.go.
  add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/generate_go_${name}.cpp
     COMMAND ${CMAKE_COMMAND}
         -DGENERATE_CPP_IN=${CMAKE_SOURCE_DIR}/src/mlpack/bindings/go/generate_go.cpp.in
         -DGENERATE_CPP_OUT=${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/generate_go_${name}.cpp
         -DPROGRAM_MAIN_FILE=${CMAKE_CURRENT_SOURCE_DIR}/${name}_main.cpp
         -DPROGRAM_NAME=${name}
         -P ${CMAKE_SOURCE_DIR}/CMake/ConfigureGenerate.cmake
     DEPENDS ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/go/generate_go.cpp.in
             ${CMAKE_SOURCE_DIR}/CMake/ConfigureGenerate.cmake)

  add_executable(generate_go_${name}
      ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/src/mlpack.org/v1/mlpack/capi/${name}.h
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
          -P ${CMAKE_SOURCE_DIR}/CMake/GenerateBinding.cmake)

  add_dependencies(go generate_go_${name})
endif ()

if(BUILD_GO_SHLIB)
  # Create ${name}.cpp.
  add_custom_command(OUTPUT
      ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/build/${name}.cpp
      COMMAND ${CMAKE_COMMAND}
          -DPROGRAM_NAME=${name}
          -DPROGRAM_MAIN_FILE=${CMAKE_CURRENT_SOURCE_DIR}/${name}_main.cpp
          -DSOURCE_DIR=${CMAKE_SOURCE_DIR}
          -DGO_IN=${CMAKE_SOURCE_DIR}/src/mlpack/bindings/go/go_method.cpp.in
          -DGO_OUT=${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/build/${name}.cpp
          -P ${CMAKE_SOURCE_DIR}/CMake/go/ConfigureGoHCPP.cmake
      DEPENDS ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/go/go_method.cpp.in
              ${CMAKE_SOURCE_DIR}/CMake/go/ConfigureGoHCPP.cmake)

  # Build libmlpack_go_${name}.so.
  add_library(mlpack_go_${name} SHARED
      ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/build/${name}.cpp)
  target_link_libraries(mlpack_go_${name} mlpack mlpack_go_util)
  set_target_properties(mlpack_go_${name} PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/src/mlpack.org/v1/mlpack/")


  install(TARGETS mlpack_go_${name}
      RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
      LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
      ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")

  add_dependencies(mlpack_go_${name} go_copy)
  add_dependencies(mlpack_go_${name} mlpack_go_util)
  add_dependencies(go_shlib mlpack_go_${name})

  if (BUILD_GO_BINDINGS)
    add_dependencies(mlpack_go_${name} generate_go_${name})
    add_dependencies(go mlpack_go_${name})
  endif()
endif()

endmacro ()

if (BUILD_TESTS OR BUILD_GO_SHLIB)
  add_subdirectory(tests)
endif ()
