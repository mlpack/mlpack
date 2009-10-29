
#add_subdirectory( lbfgs )
add_subdirectory( opt++ )

add_library( fastlib_native_optimization
    optimizers.cc
    optimizers.h
)
target_link_libraries( fastlib_native_optimization ${MLPACK_STD_LIBS} )

