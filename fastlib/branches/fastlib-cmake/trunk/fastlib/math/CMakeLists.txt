cmake_minimum_required(VERSION 2.6)

include_directories( $ENV{FASTLIBPATH}		     
                     ${CMAKE_CURRENT_BINARY_DIR}/../base/config
)

set(TMP_HEADERS 
    discrete.h
    geometry.h
    kernel.h
    math_lib.h
    math_lib_impl.h
    statistics.h
)
set(TMP_CXX 
    discrete.cc
    geometry.cc
    statistics.cc
)
set(TMP_SRCS ${TMP_HEADERS} ${TMP_CXX})

set(MATH_SRCS)
foreach(file ${TMP_SRCS})
  set(MATH_SRCS ${MATH_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(FASTLIB_SRCS ${FASTLIB_SRCS} ${MATH_SRCS} PARENT_SCOPE )
add_library( math ${MATH_SRCS} )

add_dependencies( math col base )

##!! doesn't exist??
# add_executable( math_test
# 		math_test.cc
# )
# 
# target_link_libraries( math_test
#                math
#                #col
#                #base	#needed?
# )

install(FILES ${TMP_HEADERS} DESTINATION include/fastlib/math)
