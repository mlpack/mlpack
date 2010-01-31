add_library( svm
    svm_main.cc
    smo.h svm.h
)
target_link_libraries( svm ${MLPACK_STD_LIBS} )

add_executable( svm_main
    svm_main.cc
    smo.h svm.h
)
target_link_libraries( svm_main svm )

