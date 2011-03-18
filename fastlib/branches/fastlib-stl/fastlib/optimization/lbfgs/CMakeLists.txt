cmake_minimum_required(VERSION 2.8)

set(SOURCES
  lbfgs_impl.h
  lbfgs.h
  quasi_newton_hessian_update.h
  trust_region_dev.h
  trust_region.h
  test_functions.h
  test_functions.cc
)

set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(FASTLIB_SRCS ${FASTLIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# test executable
add_executable(lbfgs_test
  EXCLUDE_FROM_ALL
  lbfgs_test.cc
)
# link dependencies of test executable
target_link_libraries(lbfgs_test
  fastlib
)
