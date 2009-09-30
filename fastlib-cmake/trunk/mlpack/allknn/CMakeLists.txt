add_library(allknn
    allknn.h
)
set_target_properties( allknn PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries( allknn ${MLPACK_STD_LIBS} )

##TODO
add_executable( allknn_test
    allknn_test.cc
)
target_link_libraries( allknn_test allknn )

add_executable( allknn_exe
    main.cc
)
target_link_libraries( allknn_exe allknn )
