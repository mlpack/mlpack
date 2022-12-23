macro (not_found_return message)
  message(STATUS "${message}")
  macro (add_python_binding directory name)
    # Do nothing.
  endmacro ()
  macro (add_python_wrapper directory group_name)
    # Do nothing.
  endmacro ()

  return()
endmacro ()

macro (post_python_bindings)
  # If we are building Python bindings, we have to configure setup.py but only
  # after we've recursed into methods/.
  if (BUILDING_PYTHON_BINDINGS)
    get_property(CYTHON_INCLUDE_DIRECTORIES DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        PROPERTY INCLUDE_DIRECTORIES)
    add_custom_target(python_configure
        COMMAND ${CMAKE_COMMAND}
            -D GENERATE_CPP_IN=${CMAKE_SOURCE_DIR}/src/mlpack/bindings/python/setup.py.in
            -D GENERATE_CPP_OUT=${CMAKE_BINARY_DIR}/src/mlpack/bindings/python/setup.py
            -D PACKAGE_VERSION="${PACKAGE_VERSION}"
            -D MLPACK_LIBRARIES="${MLPACK_LIBRARIES}"
            -D MLPACK_PYXS="${MLPACK_PYXS}"
            -D OpenMP_CXX_FLAGS="${OpenMP_CXX_FLAGS}"
            -D DISABLE_CFLAGS="${DISABLE_CFLAGS}"
            -D CYTHON_INCLUDE_DIRECTORIES="${CYTHON_INCLUDE_DIRECTORIES}"
            -D CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
            -D EXTRA_CXX_FLAGS="-DMLPACK_PRINT_INFO -DMLPACK_PRINT_WARN"
            -D OUTPUT_DIR=${CMAKE_BINARY_DIR}
            -P "${CMAKE_SOURCE_DIR}/CMake/ConfigureFile.cmake"
        BYPRODUCTS "${CMAKE_BINARY_DIR}/src/mlpack/bindings/python/setup.py"
        COMMENT "Configuring setup.py...")
    add_dependencies(python_configure python_copy)
    add_dependencies(python_configured python_configure)

    # Append the package version to __init__.py after all the imports are loaded.
    file(APPEND ${CMAKE_BINARY_DIR}/src/mlpack/bindings/python/mlpack/__init__.py
    "__version__='${PACKAGE_VERSION}'\n")
  endif ()
endmacro ()

# If we are not supposed to make Python bindings, define the macro so it does
# nothing and leave this file.
if (NOT BUILD_PYTHON_BINDINGS)
  not_found_return("Not building Python bindings.")
endif ()

# Generate Python setuptools file.
# Import find_python_module.
include(${CMAKE_SOURCE_DIR}/CMake/FindPythonModule.cmake)
find_package(PythonInterp)
if (NOT PYTHON_EXECUTABLE)
  set(PY_NOT_FOUND_MSG "${PY_NOT_FOUND_MSG}\n    - Python")
endif()
find_python_module(distutils)
if (NOT PY_DISTUTILS)
  set(PY_NOT_FOUND_MSG "${PY_NOT_FOUND_MSG}\n    - distutils")
endif ()
find_python_module(Cython 0.24)
if (NOT PY_CYTHON)
  set(PY_NOT_FOUND_MSG "${PY_NOT_FOUND_MSG}\n    - Cython")
endif ()
find_python_module(numpy)
if (NOT PY_NUMPY)
  set(PY_NOT_FOUND_MSG "${PY_NOT_FOUND_MSG}\n    - numpy")
endif ()
find_python_module(pandas 0.15.0)
if (NOT PY_PANDAS)
  set(PY_NOT_FOUND_MSG "${PY_NOT_FOUND_MSG}\n    - pandas")
endif ()

## We need to check here if Python and other dependencies is even available, as
## it is require to build python-bindings.
if (FORCE_BUILD_PYTHON_BINDINGS)
  if (NOT PYTHON_EXECUTABLE OR NOT PY_DISTUTILS OR NOT PY_CYTHON OR NOT PY_NUMPY
      OR NOT PY_PANDAS)
    unset(BUILD_PYTHON_BINDINGS CACHE)
    message(FATAL_ERROR "\nCould not Build Python Bindings; the following modules are not available: ${PY_NOT_FOUND_MSG}")
  endif()
else()
  if (NOT PYTHON_EXECUTABLE OR NOT PY_DISTUTILS OR NOT PY_CYTHON OR NOT PY_NUMPY
      OR NOT PY_PANDAS)
    unset(BUILD_PYTHON_BINDINGS CACHE)
    not_found_return("Not building Python bindings; the following modules are not available: ${PY_NOT_FOUND_MSG}")
  endif()
endif()

set(BUILDING_PYTHON_BINDINGS ON PARENT_SCOPE)

# Nothing in this directory will be compiled into mlpack.
# These are all the files we need to compile Cython bindings for mlpack that are
# not a part of mlpack itself.
set(CYTHON_SOURCES
  # mlpack/__init__.py is not included here---that is copied separately.
  mlpack/arma_numpy.pxd
  mlpack/arma_numpy.pyx
  mlpack/arma.pxd
  mlpack/arma_util.hpp
  mlpack/io.pxd
  mlpack/io_util.hpp
  mlpack/matrix_utils.py
  mlpack/serialization.hpp
  mlpack/serialization.pxd
  mlpack/params.pxd
  mlpack/preprocess_json_params.py
  mlpack/timers.pxd
)

set(TEST_SOURCES
  tests/dataset_info_test.py
  tests/test_python_binding.py
)

# Set the include directories correctly.
get_property(CYTHON_INCLUDE_DIRECTORIES DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    PROPERTY INCLUDE_DIRECTORIES)
set (CYTHON_INCLDIRS "${CYTHON_INCLUDE_DIRECTORIES}")

# By default, Python appears to compile with -DNDEBUG, but if we are in debug
# mode we don't want that.  We also want to disable MLPACK_HAS_BFD_DL if it is
# set.
if (DEBUG)
  set(DISABLE_CFLAGS "NDEBUG;MLPACK_HAS_BFD_DL" PARENT_SCOPE)
endif ()

add_custom_target(python ALL DEPENDS python_copy)
add_custom_target(python_copy ALL)
# The python_configure target is added later; this is a dummy target.
add_custom_target(python_configured ALL)

# Copy necessary files after making the mlpack/ directory.
add_custom_command(TARGET python_copy PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        ${CMAKE_BINARY_DIR}/src/mlpack/bindings/python/mlpack/)
foreach(cython_file ${CYTHON_SOURCES})
  add_custom_command(TARGET python_copy PRE_BUILD
      COMMAND ${CMAKE_COMMAND} ARGS -E copy
          ${CMAKE_CURRENT_SOURCE_DIR}/${cython_file}
          ${CMAKE_BINARY_DIR}/src/mlpack/bindings/python/mlpack/)
endforeach()
add_custom_command(TARGET python_copy PRE_BUILD
    COMMAND ${CMAKE_COMMAND} ARGS -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/setup.cfg
        ${CMAKE_BINARY_DIR}/src/mlpack/bindings/python/
    BYPRODUCTS ${CMAKE_BINARY_DIR}/src/mlpack/bindings/python/mlpack/${cython_file})
add_custom_command(TARGET python_copy PRE_BUILD
    COMMAND ${CMAKE_COMMAND} ARGS -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/copy_artifacts.py
        ${CMAKE_BINARY_DIR}/src/mlpack/bindings/python/)
add_custom_command(TARGET python_copy PRE_BUILD
    COMMAND ${CMAKE_COMMAND} ARGS -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/setup_readme.md
        ${CMAKE_BINARY_DIR}/src/mlpack/bindings/python/)

if (BUILD_TESTS)
  foreach(test_file ${TEST_SOURCES})
    add_custom_command(TARGET python_copy PRE_BUILD
        COMMAND ${CMAKE_COMMAND} ARGS -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/${test_file}
            ${CMAKE_BINARY_DIR}/src/mlpack/bindings/python/tests/
        BYPRODUCTS ${CMAKE_BINARY_DIR}/src/mlpack/bindings/python/tests/${test_file})
  endforeach ()
endif ()

# Install any dependencies via setuptools automatically.
add_custom_command(TARGET python_configured POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E env NO_BUILD=1 ${PYTHON_EXECUTABLE}
        ${CMAKE_BINARY_DIR}/src/mlpack/bindings/python/setup.py build
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/src/mlpack/bindings/python/)

# Then do the actual build.
add_custom_command(TARGET python POST_BUILD
    COMMAND ${PYTHON_EXECUTABLE}
        ${CMAKE_BINARY_DIR}/src/mlpack/bindings/python/setup.py build_ext
    DEPENDS mlpack/arma_numpy.pxd
            mlpack/arma_numpy.pyx
            mlpack/arma.pxd
            mlpack/arma_util.hpp
            mlpack/io.pxd
            mlpack/io_util.hpp
            mlpack/matrix_utils.py
            mlpack/preprocess_json_params.py
            mlpack
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/src/mlpack/bindings/python/)

# Copy the built artifacts, so that it is also an in-place build.
add_custom_command(TARGET python POST_BUILD
    COMMAND ${PYTHON_EXECUTABLE}
        ${CMAKE_BINARY_DIR}/src/mlpack/bindings/python/copy_artifacts.py
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/src/mlpack/bindings/python/)

add_dependencies(python python_configured)

# Configure installation script file.
if (NOT PYTHON_INSTALL_PREFIX)
  set(PYTHON_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
endif ()

execute_process(COMMAND ${PYTHON_EXECUTABLE}
    "${CMAKE_CURRENT_SOURCE_DIR}/print_python_version.py"
    "${PYTHON_INSTALL_PREFIX}"
    OUTPUT_VARIABLE CMAKE_PYTHON_PATH)
string(STRIP "${CMAKE_PYTHON_PATH}" CMAKE_PYTHON_PATH)
install(CODE "set(ENV{PYTHONPATH} ${CMAKE_PYTHON_PATH})")
install(CODE "set(PYTHON_EXECUTABLE \"${PYTHON_EXECUTABLE}\")")
install(CODE "set(CMAKE_BINARY_DIR \"${CMAKE_BINARY_DIR}\")")

install(CODE "set(PYTHON_INSTALL_PREFIX \"${PYTHON_INSTALL_PREFIX}\")")
install(CODE "execute_process(COMMAND mkdir -p $ENV{DESTDIR}${CMAKE_PYTHON_PATH})")
install(SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/PythonInstall.cmake")

# Prepare __init__.py for having all of the convenience imports appended to it.
file(COPY mlpack/__init__.py DESTINATION
    ${CMAKE_BINARY_DIR}/src/mlpack/bindings/python/mlpack/)

if (WIN32)
  # Copy all necessary DLLs to the Python build directory.
  foreach (dir ${DLL_COPY_DIRS})
    file(GLOB dll_dir_files "${dir}/*.dll")
    file(COPY ${dll_dir_files} DESTINATION ${CMAKE_BINARY_DIR}/src/mlpack/bindings/python/mlpack/)
  endforeach ()

  foreach (dll ${DLL_COPY_LIBS})
    file(COPY ${dll} DESTINATION ${CMAKE_BINARY_DIR}/src/mlpack/bindings/python/mlpack/)
  endforeach ()
endif ()

# Add a macro to build a python binding.
macro (add_python_binding directory name)
  if (BUILD_PYTHON_BINDINGS)
    set (MLPACK_PYXS ${MLPACK_PYXS} "${name}.pyx")
    set (MLPACK_PYXS ${MLPACK_PYXS} PARENT_SCOPE)
    add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/src/mlpack/bindings/python/generate_pyx_${name}.cpp
        COMMAND ${CMAKE_COMMAND}
            -DGENERATE_CPP_IN=${CMAKE_SOURCE_DIR}/src/mlpack/bindings/python/generate_pyx.cpp.in
            -DGENERATE_CPP_OUT=${CMAKE_BINARY_DIR}/src/mlpack/bindings/python/generate_pyx_${name}.cpp
            -DPROGRAM_MAIN_FILE=${CMAKE_CURRENT_SOURCE_DIR}/${directory}/${name}_main.cpp
            -DPROGRAM_NAME=${name}
            -P ${CMAKE_SOURCE_DIR}/CMake/ConfigureFile.cmake
        DEPENDS ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/python/generate_pyx.cpp.in
                ${CMAKE_SOURCE_DIR}/CMake/ConfigureFile.cmake)

    add_executable(generate_pyx_${name}
        ${CMAKE_BINARY_DIR}/src/mlpack/bindings/python/generate_pyx_${name}.cpp
        ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/python/print_pyx.hpp
        ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/python/print_pyx.cpp)
    if (BUILD_SHARED_LIBS)
      target_link_libraries(generate_pyx_${name} ${MLPACK_LIBRARIES})
    else ()
      target_link_libraries(generate_pyx_${name} -static ${MLPACK_LIBRARIES})
    endif ()
    set_target_properties(generate_pyx_${name} PROPERTIES COMPILE_FLAGS
        -DBINDING_TYPE=BINDING_TYPE_PYX)
    add_custom_command(TARGET generate_pyx_${name} POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            -DPROGRAM=$<TARGET_FILE:generate_pyx_${name}>
            -DOUTPUT_FILE=${CMAKE_BINARY_DIR}/src/mlpack/bindings/python/mlpack/${name}.pyx
            -P ${CMAKE_SOURCE_DIR}/CMake/RunProgram.cmake)

    # Build the pyx.  Since distutils doesn't support a parallel build, we'll
    # enforce it here.  Although this will always be rebuilt, that's okay because
    # distutils will determine whether or not it *actually* needs to be rebuilt.
    add_custom_target(build_pyx_${name}
        ${PYTHON_EXECUTABLE}
            ${CMAKE_BINARY_DIR}/src/mlpack/bindings/python/setup.py
            build_ext --module=${name}.pyx
        DEPENDS generate_pyx_${name}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/src/mlpack/bindings/python/
        COMMENT "Building Cython binding ${name}.so...")

    add_dependencies(python build_pyx_${name})
    add_dependencies(build_pyx_${name} generate_pyx_${name})
    add_dependencies(generate_pyx_${name} python_configured)

    # Add the convenience import to __init__.py.  Note that this happens during
    # configuration.
    file(APPEND ${CMAKE_BINARY_DIR}/src/mlpack/bindings/python/mlpack/__init__.py
        "from .${name} import ${name}\n")
  endif ()
endmacro()

# this macro creates .so files for all methods in the given group.
# if using add_python_wrapper(), then must group bindings using group_bindings()
macro (add_python_wrapper directory group_name)

  foreach (method ${${group_name}_VALID_METHODS})
    add_python_binding(${directory} "${group_name}_${method}")
  endforeach()

  if(EXISTS ${CMAKE_BINARY_DIR}/src/mlpack/bindings/python/${group_name}_main_includes.hpp)
    file(REMOVE ${CMAKE_BINARY_DIR}/src/mlpack/bindings/python/${group_name}_main_includes.hpp)
  endif()

  foreach(loc ${${group_name}_METHOD_MAIN_FILES})
    file(APPEND ${CMAKE_BINARY_DIR}/src/mlpack/bindings/python/${group_name}_main_includes.hpp
        "#include <${loc}>\n")
  endforeach()

  file(APPEND ${CMAKE_BINARY_DIR}/src/mlpack/bindings/python/mlpack/__init__.py
      "from .${group_name} import *\n")

  add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/src/mlpack/bindings/python/generate_py_wrapper_${group_name}.cpp
  COMMAND ${CMAKE_COMMAND}
      -DGENERATE_CPP_IN=${CMAKE_SOURCE_DIR}/src/mlpack/bindings/python/generate_py_wrapper.cpp.in
      -DGENERATE_CPP_OUT=${CMAKE_BINARY_DIR}/src/mlpack/bindings/python/generate_py_wrapper_${group_name}.cpp
      -DGROUP_NAME=${group_name}
      -DGROUP_VALID_METHODS="${${group_name}_VALID_METHODS}"
      -DCATEGORY="${${group_name}_CATEGORY}"
      -P ${CMAKE_SOURCE_DIR}/CMake/ConfigureFile.cmake
  DEPENDS ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/python/generate_py_wrapper.cpp.in
          ${CMAKE_SOURCE_DIR}/CMake/ConfigureFile.cmake)

  add_executable(generate_py_wrapper_${group_name}
      ${CMAKE_BINARY_DIR}/src/mlpack/bindings/python/generate_py_wrapper_${group_name}.cpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/python/print_wrapper_py.hpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/python/print_wrapper_py.cpp)

  if (BUILD_SHARED_LIBS)
    target_link_libraries(generate_py_wrapper_${group_name} ${MLPACK_LIBRARIES})
  else ()
    target_link_libraries(generate_py_wrapper_${group_name} -static
        ${MLPACK_LIBRARIES})
  endif ()

  add_custom_command(TARGET generate_py_wrapper_${group_name} POST_BUILD
      COMMAND ${CMAKE_COMMAND}
          -DPROGRAM=$<TARGET_FILE:generate_py_wrapper_${group_name}>
          -DOUTPUT_FILE=${CMAKE_BINARY_DIR}/src/mlpack/bindings/python/mlpack/${group_name}.py
          -P ${CMAKE_SOURCE_DIR}/CMake/RunProgram.cmake)
    add_dependencies(python generate_py_wrapper_${group_name})
    add_dependencies(generate_py_wrapper_${group_name} python_configured)
endmacro ()

# Add a test.
if (BUILD_PYTHON_BINDINGS)
  if (BUILD_TESTS)
    add_test(NAME python_bindings_test
        COMMAND ${PYTHON_EXECUTABLE}
            ${CMAKE_BINARY_DIR}/src/mlpack/bindings/python/setup.py test
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/src/mlpack/bindings/python/)
    set_tests_properties(python_bindings_test
        PROPERTIES ENVIRONMENT "NO_BUILD=1;LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:${CMAKE_BINARY_DIR}/lib/")
  endif ()
endif ()

if (BUILD_TESTS)
  add_subdirectory(tests)
endif ()

set(MLPACK_PYXS "arma_numpy.pyx" ${MLPACK_PYXS} PARENT_SCOPE)
