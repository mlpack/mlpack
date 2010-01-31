add_library( mog_em
    mog_em.cc
    mog_em.h
    phi.h
    math_functions.h
)
target_link_libraries( mog_em ${MLPACK_STD_LIBS} )

add_library( mog_l2e
    mog_l2e.cc
    mog_l2e.h
    phi.h
)
target_link_libraries( mog_l2e ${MLPACK_STD_LIBS} )

add_executable( mog_em_main
    mog_em_main.cc
)
target_link_libraries( mog_em_main mog_em )
#target_link_libraries( mog_em_main mog_em ${MLPACK_STD_LIBS} )

add_executable( mog_l2e_main
    mog_l2e_main.cc
)
target_link_libraries( mog_l2e_main
    mog_l2e
#   optimization
#    ${MLPACK_STD_LIBS}
)
