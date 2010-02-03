add_library( kalman
    kalman_helper.cc kalman.cc
    kalman_helper.h kalman.h
)
target_link_libraries( kalman ${MLPACK_STD_LIBS} )

add_executable( kalman_main
    kalman_main.cc
)
target_link_libraries( kalman_main kalman )
