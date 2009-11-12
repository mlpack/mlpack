set(FASTLIB_SRCS ${FASTLIB_SRCS} PARENT_SCOPE )
message("in optim: ${FASTLIB_SRCS}")

#add_subdirectory( lbfgs )
add_subdirectory( optpp )

##TODO depends on everything - bleh
#add_library( fastlib_native_optimization
#    optimizers.cc
#    optimizers.h
#)
#target_link_libraries( fastlib_native_optimization ${MLPACK_STD_LIBS} )

