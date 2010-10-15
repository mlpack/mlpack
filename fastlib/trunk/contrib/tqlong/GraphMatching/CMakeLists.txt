cmake_minimum_required(VERSION 2.8)

set(SOURCES
  feature_detector
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

SET(graph_matching_SOURCES graphmatching_main.cpp main_window.cc image_view.cc)
SET(graph_matching_HEADERS main_window.h image_view.h)
QT4_WRAP_CPP(graph_matching_HEADERS_MOC ${graph_matching_HEADERS})

add_executable(graph_matching EXCLUDE_FROM_ALL
  ${graph_matching_SOURCES}
  ${graph_matching_HEADERS_MOC}
)
add_dependencies(graph_matching
  mlpack_contrib
)
# link dependencies of executable
target_link_libraries(graph_matching
  mlpack_contrib
  fastlib
  ${QT_LIBRARIES}
  ${OpenCV_LIBS}
)
