add_library(allnn
    allnn.h
)
set_target_properties( allnn PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries( allnn ${MLPACK_STD_LIBS} )

add_executable( allnn_test allnn_test.cc )
target_link_libraries( allnn_test allnn )

add_executable( allnn_main main.cc )
target_link_libraries( allnn_main allnn )
