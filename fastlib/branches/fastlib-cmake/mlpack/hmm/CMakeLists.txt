add_library( hmm
    support.cc discreteHMM.cc gaussianHMM.cc mixgaussHMM.cc mixtureDST.cc
    support.h discreteHMM.h gaussianHMM.h mixgaussHMM.h mixtureDST.h
)
target_link_libraries( hmm ${MLPACK_STD_LIBS} )

add_executable( generate generate.cc )
target_link_libraries( generate hmm )

add_executable( loglik loglik.cc ) 
target_link_libraries( loglik hmm )

add_executable( viterbi viterbi.cc )
target_link_libraries( viterbi hmm )

add_executable( train train.cc )
target_link_libraries( train hmm )
