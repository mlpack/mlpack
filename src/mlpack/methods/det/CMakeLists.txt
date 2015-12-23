# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
  # the DET class
  dtree.hpp
  dtree.cpp

  # the util file
  dt_utils.hpp
  dt_utils.cpp
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all mlpack sources (used at the parent scope)
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# executable
add_executable(mlpack_det
  det_main.cpp
)
# link dependencies of executable
target_link_libraries(mlpack_det
  mlpack
)
install(TARGETS mlpack_det RUNTIME DESTINATION bin)
