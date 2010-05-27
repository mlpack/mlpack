cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   discreteHMM.h
   discreteHMM.cc
   gaussianHMM.h
   gaussianHMM.cc
   mixgaussHMM.h
   mixgaussHMM.cc
   mixtureDST.h
   mixtureDST.cc
   support.h
   support.cc
   hmm_documentation.h
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# executables
add_executable(hmm_generate
   generate.cc
)
target_link_libraries(hmm_generate
   mlpack
   fastlib
)

add_executable(hmm_loglik
   loglik.cc
)
target_link_libraries(hmm_loglik
   mlpack
   fastlib
)

add_executable(hmm_viterbi
   viterbi.cc
)
target_link_libraries(hmm_viterbi
   mlpack
   fastlib
)

add_executable(hmm_train
   train.cc
)
target_link_libraries(hmm_train
   mlpack
   fastlib
)
