cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile.
# Anything not in this list will not be compiled into MLPACK.
set(SOURCES
  discreteHMM.hpp
  discreteHMM.cpp
  gaussianHMM.hpp
  gaussianHMM.cpp
  mixgaussHMM.hpp
  mixgaussHMM.cpp
  mixtureDST.hpp
  mixtureDST.cpp
  support.hpp
  support.cpp
  hmm.hpp
  hmm_impl.hpp
  distributions/discrete_distribution.hpp
  distributions/discrete_distribution_impl.hpp
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
  generate.cpp
)
target_link_libraries(generate
  mlpack
)

add_executable(loglik
  loglik.cpp
)
target_link_libraries(loglik
  mlpack
)

add_executable(viterbi
  viterbi.cpp
)
target_link_libraries(viterbi
  mlpack
)

add_executable(train
  train.cpp
)
target_link_libraries(train
  mlpack
)
