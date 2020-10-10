# Define the files we need to compile.
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  hmm.hpp
  hmm_impl.hpp
  hmm_model.hpp
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
# Append sources (with directory name) to list of all mlpack sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_cli_executable(hmm_train)
add_python_binding(hmm_train)
add_julia_binding(hmm_train)
add_go_binding(hmm_train)
add_r_binding(hmm_train)
add_markdown_docs(hmm_train "cli;python;julia;go;r" "misc. / other")

add_cli_executable(hmm_loglik)
add_python_binding(hmm_loglik)
add_julia_binding(hmm_loglik)
add_go_binding(hmm_loglik)
add_r_binding(hmm_loglik)
add_markdown_docs(hmm_loglik "cli;python;julia;go;r" "misc. / other")

add_cli_executable(hmm_viterbi)
add_python_binding(hmm_viterbi)
add_julia_binding(hmm_viterbi)
add_go_binding(hmm_viterbi)
add_r_binding(hmm_viterbi)
add_markdown_docs(hmm_viterbi "cli;python;julia;go;r" "misc. / other")

add_cli_executable(hmm_generate)
add_python_binding(hmm_generate)
add_julia_binding(hmm_generate)
add_go_binding(hmm_generate)
add_r_binding(hmm_generate)
add_markdown_docs(hmm_generate "cli;python;julia;go;r" "misc. / other")
