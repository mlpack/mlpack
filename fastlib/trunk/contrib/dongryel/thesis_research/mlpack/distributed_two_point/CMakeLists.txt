cmake_minimum_required(VERSION 2.6)

set(SOURCES
  distributed_two_point_arguments.h
  distributed_two_point_dev.h
  distributed_two_point.h
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)


add_executable(distributed_two_point
               EXCLUDE_FROM_ALL
               distributed_two_point.cc
)

# link dependencies 
target_link_libraries(distributed_two_point
                      core
                      ${TRILINOS_LIBS}
                      ${Boost_LIBRARIES}
)

add_executable(distributed_two_point_test
	EXCLUDE_FROM_ALL
	distributed_two_point_test.cc
)
target_link_libraries(distributed_two_point_test
	core
	${TRILINOS_LIBS}
	${Boost_LIBRARIES}
)
