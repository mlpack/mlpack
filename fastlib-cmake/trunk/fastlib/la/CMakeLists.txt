# file( DOWNLOAD            
#       http://www.netlib.org/blas/blas.tgz 
#       ${CMAKE_CURRENT_BINARY_DIR}/blas.tgz
# )

# file( DOWNLOAD
#       http://www.netlib/org/lapack/lapack.tgz
#       ${CMAKE_CURRENT_BINARY_DIR}/lapack.tgz   
# )

set( TMP_HEADERS 
    blas.h	clapack.h  la.h  matrix.h  uselapack.h
)
set( TMP_CXX 
    uselapack.cc  #uselapack_test.cc
    )
set( TMP_SRCS ${TMP_HEADERS} ${TMP_CXX})

set(LA_SRCS)
foreach(file ${TMP_SRCS})
  set(LA_SRCS ${LA_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
set(FASTLIB_SRCS ${FASTLIB_SRCS} ${LA_SRCS} PARENT_SCOPE )
add_library( la ${LA_SRCS} )

#target_link_libraries( la base col blas lapack )
target_link_libraries( la base col ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} )

install(FILES ${TMP_HEADERS} DESTINATION include/fastlib/la)
