cmake_minimum_required(VERSION 2.8)
project(FASTLIB C CXX Fortran)

set(SOURCES
  abstract_point.h
  dense_matrix.h
  dense_point.h
  distributed_table.h
  distributed_table_message.h
  mailbox.h
  memory_mapped_file.h
  point_request_message.h
  table.h
  transform.h
)

set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(CORE_SRCS ${CORE_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# test executable: distributed_table-test
add_executable(distributed_table-test
               EXCLUDE_FROM_ALL
               distributed_table.test.cc
              )
# link dependencies of test executable
target_link_libraries(distributed_table-test
                      core
                      ${Boost_LIBRARIES}
                     )
