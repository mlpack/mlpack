cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile.
# Anything not in this list will not be compiled into MLPACK.
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
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all MLPACK sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# executables
add_executable(generate
  generate.cc
)
target_link_libraries(generate
  mlpack
)

add_executable(loglik
  loglik.cc
)
target_link_libraries(loglik
  mlpack
)

add_executable(viterbi
  viterbi.cc
)
target_link_libraries(viterbi
  mlpack
)

add_executable(train
  train.cc
)
target_link_libraries(train
  mlpack
)
