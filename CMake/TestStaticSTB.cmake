# Author: Omar Shrit

#[=======================================================================[.rst:
TestForSTB
----------

Test to verify if the available version of STB contains a working static
implementation that can be used from multiple translation units.

::

    CMAKE_HAS_WORKING_STATIC_STB - defined by the results
#]=======================================================================]

if(NOT DEFINED CMAKE_HAS_WORKING_STATIC_STB)
  message(STATUS "Check that STB static implementation mode links correctly...")

  set(TEST_CXX_DEFS "")
  if (NOT STB_INCLUDE_NEEDS_STB_SUFFIX)
    set(TEST_CXX_DEFS "-DMLPACK_HAS_NO_STB_DIR")
  endif ()

  try_compile(CMAKE_HAS_WORKING_STATIC_STB
      ${CMAKE_BINARY_DIR}/CMakeFiles/CMakeTmp/
      SOURCES
        ${CMAKE_SOURCE_DIR}/CMake/stb/main.cpp
        ${CMAKE_SOURCE_DIR}/CMake/stb/a.cpp
        ${CMAKE_SOURCE_DIR}/CMake/stb/b.cpp
      CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${STB_IMAGE_INCLUDE_DIR}"
      COMPILE_DEFINITIONS "${TEST_CXX_DEFS}"
      OUTPUT_VARIABLE out)
  if (CMAKE_HAS_WORKING_STATIC_STB)
    message(STATUS "Check that STB static implementation mode links "
        "correctly... success")
    message(STATUS
        "Determining if STB's static implementation can link correctly passed "
        "with the following output:\n${out}\n\n")
    set(CMAKE_HAS_WORKING_STATIC_STB 1 CACHE INTERNAL
        "Does STB static implementation mode link correctly")
    file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
        "Determining if STB's static implementation can link correctly passed "
        "with the following output:\n${out}\n\n")
  else ()
    message(STATUS "Check that STB static implementation mode links "
        "correctly... fail")
    message(STATUS
        "Determining if STB's static implementation can link correctly failed "
        "with the following output:\n${out}\n\n")
    set(CMAKE_HAS_WORKING_STATIC_STB 0 CACHE INTERNAL
        "Does STB static implementation mode link correctly")
    file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
        "Determining if STB's static implementation can link correctly failed "
        "with the following output:\n${out}\n\n")
  endif ()
endif()
