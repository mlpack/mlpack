cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   support.h
   support.cc
   discreteHMM.h
   discreteHMM.cc
   gaussianHMM.h
   gaussianHMM.cc
   mixgaussHMM.h
   mixgaussHMM.cc
   mixtureDST.h
   mixtureDST.cc
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# all executables suffixed with _niche because I believe they already all exist in fastlib
# executable
add_executable(generate_niche
   generate.cc
)
# link dependencies of executable
target_link_libraries(generate_niche
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(loglik_niche
   loglik.cc
)
target_link_libraries(loglik_niche
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(viterbi_niche
   viterbi.cc
)
target_link_libraries(viterbi_niche
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(train_niche
   train.cc
)
target_link_libraries(train_niche
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(mmf_niche
   mmf.cc
)
target_link_libraries(mmf_niche
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(mmf1_niche
   mmf1.cc
)
target_link_libraries(mmf1_niche
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(mmf2_niche
   mmf2.cc
)
target_link_libraries(mmf2_niche
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(mmf3_niche
   mmf3.cc
)
target_link_libraries(mmf3_niche
   fastlib
   mlpack
   mlpack_contrib
)
