# Define the files we need to compile.
# Anything not in this list will not be compiled into MLPACK.
set(SOURCES
  hmm.hpp
  hmm_impl.hpp
  hmm_regression.hpp
  hmm_regression_impl.hpp
  hmm_util.hpp
  hmm_util_impl.hpp
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all MLPACK sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_executable(mlpack_hmm_train
  hmm_train_main.cpp
)
target_link_libraries(mlpack_hmm_train
  mlpack
)

add_executable(mlpack_hmm_loglik
  hmm_loglik_main.cpp
)
target_link_libraries(mlpack_hmm_loglik
  mlpack
)

add_executable(mlpack_hmm_viterbi
  hmm_viterbi_main.cpp
)
target_link_libraries(mlpack_hmm_viterbi
  mlpack
)

add_executable(mlpack_hmm_generate
  hmm_generate_main.cpp
)
target_link_libraries(mlpack_hmm_generate
  mlpack
)

install(TARGETS
    mlpack_hmm_train
    mlpack_hmm_loglik
    mlpack_hmm_viterbi
    mlpack_hmm_generate
  RUNTIME DESTINATION bin)
