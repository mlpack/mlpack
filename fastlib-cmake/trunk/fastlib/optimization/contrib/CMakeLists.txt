
set( TMP_HEADERS 
    optimizers.h
)
set( TMP_CXX 
    optimizers.cc
    )
set( TMP_SRCS ${TMP_HEADERS} ${TMP_CXX})

set(FL_OPT_CONTRIB_SRCS)
foreach(file ${TMP_SRCS})
  set(FL_OPT_CONTRIB_SRCS ${FL_OPT_CONTRIB_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
set(FASTLIB_SRCS ${FASTLIB_SRCS} ${FL_OPT_CONTRIB_SRCS} PARENT_SCOPE )

add_library( fastlib_contrib_optimization ${FL_OPT_CONTRIB_SRCS} )
add_dependencies( fastlib_contrib_optimization fx la )

#target_link_libraries( la base col blas lapack )
#target_link_libraries( fastlib_contrib_optimization )

install(FILES ${TMP_HEADERS} DESTINATION include/fastlib/optimization/contrib)
