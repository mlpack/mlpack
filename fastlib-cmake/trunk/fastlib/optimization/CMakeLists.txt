
add_subdirectory( contrib )
add_subdirectory( lbfgs )
add_subdirectory( optpp )

set(FASTLIB_SRCS ${FASTLIB_SRCS} PARENT_SCOPE )

add_custom_target( optimization )
add_dependencies( optimization 
    fastlib_contrib_optimization
    fastlib_lbfgs
    fastlib_optpp
)

