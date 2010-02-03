add_library( nbc
    simple_nbc.h
    phi.h
    math_functions.h
)
#    tests = ["test_simple_nbc_main.cc"]
set_target_properties( nbc PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries( nbc ${MLPACK_STD_LIBS} )

add_executable( nbc_main
    nbc_main.cc
)
#target_link_libraries( nbc_main nbc ${MLPACK_STD_LIBS} )
target_link_libraries( nbc_main nbc )
