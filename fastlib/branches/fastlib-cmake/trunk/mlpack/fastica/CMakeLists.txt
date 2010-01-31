add_library( fastica_lib
    fastica.h lin_alg.h
)
set_target_properties( fastica_lib PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries( fastica_lib ${MLPACK_STD_LIBS} )

add_executable( fastica
    fastica_main.cc
    lin_alg.h
)
target_link_libraries( fastica fastica_lib )
