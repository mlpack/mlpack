# This macro allows grouping a set of bindings into a single class.
macro(group_bindings directory name methods)
  set(${name}_METHOD_MAIN_FILES "")
  set(${name}_VALID_METHODS "")

  foreach (method ${methods})
    set(filename "${CMAKE_CURRENT_SOURCE_DIR}/${directory}/${name}_${method}_main.cpp")
    if(EXISTS ${filename})
      list(APPEND ${name}_METHOD_MAIN_FILES
          ${CMAKE_CURRENT_SOURCE_DIR}/${directory}/${name}_${method}_main.cpp)
      list(APPEND ${name}_VALID_METHODS ${method})
    else()
      message(FATAL_ERROR "file: ${filename} does not exist!!")
    endif()
  endforeach()
endmacro()

# All we have to do is recurse into the subdirectories.
set(DIRS
  cli
  julia
  markdown
  python
  go
  R
  tests
)

foreach(dir ${DIRS})
  add_subdirectory(${dir})
endforeach()

# Some of these variables may be set in the lower-level test directories, and we
# need to pass their values back up so they are accessible in
# src/mlpack/methods/.
set(MLPACK_SRCS ${MLPACK_SRCS} PARENT_SCOPE)
set(MLPACK_TEST_SRCS ${MLPACK_TEST_SRCS} PARENT_SCOPE)
set(MLPACK_PYXS ${MLPACK_PYXS} PARENT_SCOPE)
set(DISABLE_CFLAGS ${DISABLE_CFLAGS} PARENT_SCOPE)
set(BUILDING_PYTHON_BINDINGS ${BUILDING_PYTHON_BINDINGS} PARENT_SCOPE)
set(JULIA_GROUP_BINDINGS ${JULIA_GROUP_BINDINGS} PARENT_SCOPE)

include(Categories.cmake)
set(CATEGORIES ${CATEGORIES} PARENT_SCOPE)

macro(add_category name category)
  # First, make sure that the category is a valid category.
  list(FIND CATEGORIES ${category} cat_index)
  if (${cat_index} EQUAL -1)
    message(FATAL_ERROR "add_category(): unknown category ${category}!"
        "  See the categories in "
        "src/mlpack/bindings/Categories.cmake.")
  endif ()
  set(${name}_CATEGORY ${category})
endmacro()
