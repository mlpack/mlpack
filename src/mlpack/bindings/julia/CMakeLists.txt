if (BUILD_JULIA_BINDINGS)
  ## We need to check here if Julia is even available.  Although actually
  ## technically, I'm not sure if we even need to know!  For the tests though we
  ## do.  So it's probably a good idea to check.
  find_package(Julia 0.7.0)

  if (NOT JULIA_FOUND)
    # We can't build anything, so define the macro to do nothing.
    macro (add_julia_binding name)
      # Do nothing.
    endmacro ()

    return ()
  endif ()

  add_custom_target(julia ALL)

  add_custom_command(TARGET julia PRE_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory
          ${CMAKE_BINARY_DIR}/src/mlpack/bindings/julia/mlpack/src/)
  add_custom_command(TARGET julia PRE_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
          ${CMAKE_CURRENT_SOURCE_DIR}/mlpack/REQUIRE
          ${CMAKE_BINARY_DIR}/src/mlpack/bindings/julia/mlpack/REQUIRE)

  add_library(mlpack_julia_util
    julia_util.h
    julia_util.cpp)
  target_link_libraries(mlpack_julia_util mlpack)
  set_target_properties(mlpack_julia_util PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/src/mlpack/bindings/julia/mlpack/src/")
  add_dependencies(julia mlpack_julia_util)

  # Now configure Project.toml.
  file(READ "${CMAKE_SOURCE_DIR}/src/mlpack/core/util/version.hpp"
      VERSION_HPP_CONTENTS)
  string(REGEX REPLACE ".*#define MLPACK_VERSION_MAJOR ([0-9]+).*" "\\1"
      MLPACK_VERSION_MAJOR "${VERSION_HPP_CONTENTS}")
  string(REGEX REPLACE ".*#define MLPACK_VERSION_MINOR ([0-9]+).*" "\\1"
      MLPACK_VERSION_MINOR "${VERSION_HPP_CONTENTS}")
  string(REGEX REPLACE ".*#define MLPACK_VERSION_PATCH [\"]?([0-9x]+)[\"]?.*"
      "\\1" MLPACK_VERSION_PATCH "${VERSION_HPP_CONTENTS}")

  set(PACKAGE_VERSION
      "${MLPACK_VERSION_MAJOR}.${MLPACK_VERSION_MINOR}.${MLPACK_VERSION_PATCH}")

  get_property(CYTHON_INCLUDE_DIRECTORIES DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      PROPERTY INCLUDE_DIRECTORIES)
  configure_file(${CMAKE_SOURCE_DIR}/src/mlpack/bindings/julia/mlpack/Project.toml.in
                 ${CMAKE_BINARY_DIR}/src/mlpack/bindings/julia/mlpack/Project.toml)
  # Configure cli.jl.in with the right suffix for libraries.
  configure_file(${CMAKE_SOURCE_DIR}/src/mlpack/bindings/julia/mlpack/cli.jl.in
                 ${CMAKE_BINARY_DIR}/src/mlpack/bindings/julia/mlpack/src/cli.jl)

  # Create the empty mlpack.jl file that we will fill with includes.
  file(WRITE
      "${CMAKE_BINARY_DIR}/src/mlpack/bindings/julia/mlpack/src/mlpack.jl"
      "module mlpack\n\n"
      "module util\n\n"
      "include(\"cli.jl\")\n")

  file(WRITE
      "${CMAKE_BINARY_DIR}/src/mlpack/bindings/julia/mlpack/src/functions.jl"
      "# This file imports all of the mlpack functions into the mlpack module."
      "\n\n")
endif ()

macro (add_julia_binding name)
if (BUILD_JULIA_BINDINGS)
  # We have to take multiple steps.

  # 1. Generate julia_${name}.h and julia_${name}.cpp.
  add_custom_command(OUTPUT
      ${CMAKE_BINARY_DIR}/src/mlpack/bindings/julia/mlpack/build/julia_${name}.h
      ${CMAKE_BINARY_DIR}/src/mlpack/bindings/julia/mlpack/build/julia_${name}.cpp
      COMMAND ${CMAKE_COMMAND}
          -DPROGRAM_NAME="${name}"
          -DPROGRAM_MAIN_FILE="${CMAKE_CURRENT_SOURCE_DIR}/${name}_main.cpp"
          -DJULIA_H_IN="${CMAKE_SOURCE_DIR}/src/mlpack/bindings/julia/julia_method.h.in"
          -DJULIA_H_OUT="${CMAKE_BINARY_DIR}/src/mlpack/bindings/julia/mlpack/build/julia_${name}.h"
          -DJULIA_CPP_IN="${CMAKE_SOURCE_DIR}/src/mlpack/bindings/julia/julia_method.cpp.in"
          -DJULIA_CPP_OUT="${CMAKE_BINARY_DIR}/src/mlpack/bindings/julia/mlpack/build/julia_${name}.cpp"
          -P ${CMAKE_SOURCE_DIR}/CMake/julia/ConfigureJuliaHCPP.cmake
      DEPENDS ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/julia/julia_method.h.in
              ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/julia/julia_method.cpp.in
              ${CMAKE_SOURCE_DIR}/CMake/julia/ConfigureJuliaHCPP.cmake)

  # 2. Build libmlpack_julia_${name}.so.
  add_library(mlpack_julia_${name}
      ${CMAKE_BINARY_DIR}/src/mlpack/bindings/julia/mlpack/build/julia_${name}.h
      ${CMAKE_BINARY_DIR}/src/mlpack/bindings/julia/mlpack/build/julia_${name}.cpp)
  target_link_libraries(mlpack_julia_${name} mlpack mlpack_julia_util)
  set_target_properties(mlpack_julia_${name} PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/src/mlpack/bindings/julia/mlpack/src/")

  # 3. Generate ${name}.jl.
  add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/src/mlpack/bindings/julia/mlpack/build/generate_jl_${name}.cpp
      COMMAND ${CMAKE_COMMAND}
          -DNAME=${name}
          -DGENERATE_CPP_IN=${CMAKE_SOURCE_DIR}/src/mlpack/bindings/julia/generate_jl.cpp.in
          -DGENERATE_CPP_OUT=${CMAKE_BINARY_DIR}/src/mlpack/bindings/julia/mlpack/build/generate_jl_${name}.cpp
          -DPROGRAM_MAIN_FILE=${CMAKE_CURRENT_SOURCE_DIR}/${name}_main.cpp
          -DMLPACK_JL_LIB_SUFFIX=${CMAKE_SHARED_LIBRARY_SUFFIX}
          -P ${CMAKE_SOURCE_DIR}/CMake/ConfigureGenerate.cmake
      DEPENDS ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/julia/generate_jl.cpp.in
              ${CMAKE_SOURCE_DIR}/CMake/ConfigureGenerate.cmake)

  add_executable(generate_jl_${name}
      ${CMAKE_BINARY_DIR}/src/mlpack/bindings/julia/mlpack/build/generate_jl_${name}.cpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/julia/print_jl.hpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/julia/print_jl.cpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/julia/get_julia_type.hpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/julia/get_param.hpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/julia/get_printable_param.hpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/julia/julia_option.hpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/julia/print_doc_functions.hpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/julia/print_doc_functions_impl.hpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/julia/print_input_param.hpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/julia/print_input_processing.hpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/julia/print_input_processing_impl.hpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/julia/print_output_processing.hpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/julia/print_output_processing_impl.hpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/julia/print_param_defn.hpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/julia/print_type_doc.hpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/julia/print_type_doc_impl.hpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/julia/default_param.hpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/julia/default_param_impl.hpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/julia/strip_type.hpp)
  target_link_libraries(generate_jl_${name} mlpack ${MLPACK_LIBRARIES})
  set_target_properties(generate_jl_${name} PROPERTIES
      COMPILE_FLAGS "-DBINDING_TYPE=BINDING_TYPE_JL"
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/src/mlpack/bindings/julia/mlpack/build/bin/")
  add_custom_command(TARGET generate_jl_${name} POST_BUILD
      COMMAND ${CMAKE_COMMAND}
          -DGENERATE_BINDING_PROGRAM="${CMAKE_BINARY_DIR}/src/mlpack/bindings/julia/mlpack/build/bin/generate_jl_${name}"
          -DBINDING_OUTPUT_FILE=${CMAKE_BINARY_DIR}/src/mlpack/bindings/julia/mlpack/src/${name}.jl
          -P ${CMAKE_SOURCE_DIR}/CMake/GenerateBinding.cmake)

  # Add the generate_jl_${name} target to the list of targets that are built
  # when 'make julia' is typed.
  add_dependencies(julia mlpack_julia_${name})
  add_dependencies(julia generate_jl_${name})

  # Append the binding to the mlpack.jl file.  (Note that this is done at
  # configuration time.)
  file(APPEND
      "${CMAKE_BINARY_DIR}/src/mlpack/bindings/julia/mlpack/src/mlpack.jl"
      "include(\"${name}.jl\")\n")

  # Append the code to define the function in the module.
  file(APPEND
      "${CMAKE_BINARY_DIR}/src/mlpack/bindings/julia/mlpack/src/functions.jl"
      "${name} = util.${name}\n")
endif ()
endmacro ()

if (BUILD_TESTS)
  add_subdirectory(tests)
endif ()
