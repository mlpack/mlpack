
add_library( infomax_ica
    infomax_ica.cc
    infomax_ica.h
)
target_link_libraries( infomax_ica ${MLPACK_STD_LIBS} )

add_executable( infomax_ica_main
    main.cc
)
target_link_libraries( infomax_ica_main infomax_ica )

add_executable( infomax_ica_test_main
    test_main.cc
)
target_link_libraries( infomax_ica_test_main infomax_ica )

