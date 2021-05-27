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

set(MARKDOWN_CATEGORIES ${MARKDOWN_CATEGORIES} PARENT_SCOPE)
set(MLPACK_SRCS ${MLPACK_SRCS} PARENT_SCOPE)
set(MLPACK_PYXS ${MLPACK_PYXS} PARENT_SCOPE)
set(DISABLE_CFLAGS ${DISABLE_CFLAGS} PARENT_SCOPE)
set(BUILDING_PYTHON_BINDINGS ${BUILDING_PYTHON_BINDINGS} PARENT_SCOPE)

macro(group_bindings name methods)
  set(${name}_method_main_files "")
  set(${name}_valid_methods "")

  foreach (method ${methods})
    message(WARNING ${method})
    set(filename "${CMAKE_CURRENT_SOURCE_DIR}/${name}_${method}_main.cpp")
    if(EXISTS ${filename})
      list(APPEND ${name}_method_main_files ${CMAKE_CURRENT_SOURCE_DIR}/${name}_${method}_main.cpp)
      list(APPEND ${name}_valid_methods ${method})
    else()
      message(FATAL_ERROR "file: ${filename} does not exist!!")
    endif()
  endforeach()
endmacro()