add_subdirectory( lbfgs )

add_library( optimization
    optimizers.cc
    optimizers.h
)
target_link_libraries( optimization ${MLPACK_STD_LIBS} )

