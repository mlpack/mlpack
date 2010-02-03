cmake_minimum_required(VERSION 2.6)

## include_directories( $ENV{FASTLIBPATH}		     
##                      ${CMAKE_CURRENT_BINARY_DIR}/../base/config
## )

set( TMP_HEADERS 
    mvu_objectives.h
    mvu_objectives_impl.h
)
set( TMP_CXX 
    
)	
set( TMP_SRCS ${TMP_HEADERS} ${TMP_CXX})
set(mvu_SRCS)
foreach(file ${TMP_SRCS})
  set(mvu_SRCS ${mvu_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

## set(MLPACK_SRCS ${MLPACK_SRCS} ${mvu_SRCS} PARENT_SCOPE )
add_library( mvu ${mvu_SRCS} )
set_target_properties( mvu PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries( mvu
    allknn allkfn  #TODO
    ${MLPACK_STD_LIBS}
)

add_executable( ncmvu
    main.cc
)
target_link_libraries( ncmvu
    mvu
#   lbfgs  #TODO
    ${MLPACK_STD_LIBS}
)
## install(FILES ${TMP_HEADERS} DESTINATION include/fastlib/mvu)
