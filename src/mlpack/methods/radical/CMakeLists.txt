# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
set(SOURCES
   radical.hpp
   radical.cpp
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all mlpack sources (used at the parent scope)
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_executable(mlpack_radical
  radical_main.cpp
)
target_link_libraries(mlpack_radical
  mlpack
)
install(TARGETS mlpack_radical RUNTIME DESTINATION bin)
