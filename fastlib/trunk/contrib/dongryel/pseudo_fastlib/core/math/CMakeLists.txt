cmake_minimum_required(VERSION 2.8)

set(SOURCES
  discrete.cc
  discrete.h
  geometry.cc
  geometry.h
  math_lib.h
  math_lib_impl.h
  range.h
)

set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(CORE_SRCS ${CORE_SRCS} ${DIR_SRCS} PARENT_SCOPE)
