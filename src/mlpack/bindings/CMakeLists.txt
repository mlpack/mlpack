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
