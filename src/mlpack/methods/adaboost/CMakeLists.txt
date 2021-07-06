# Define the files we need to compile.
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  adaboost.hpp
  adaboost_impl.hpp
  adaboost_model.hpp
  adaboost_model.cpp
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all mlpack sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_type(adaboost "classification")
group_bindings(adaboost "fit;predict;predict_proba")
add_python_binding(adaboost)

add_cli_executable(adaboost_fit)
add_julia_binding(adaboost_fit)
add_go_binding(adaboost_fit)
add_r_binding(adaboost_fit)
add_markdown_docs(adaboost_fit "cli;python;julia;go;r" "classification")

add_cli_executable(adaboost_predict)
add_julia_binding(adaboost_predict)
add_go_binding(adaboost_predict)
add_r_binding(adaboost_predict)
add_markdown_docs(adaboost_predict "cli;python;julia;go;r" "classification")

add_cli_executable(adaboost_predict_proba)
add_julia_binding(adaboost_predict_proba)
add_go_binding(adaboost_predict_proba)
add_r_binding(adaboost_predict_proba)
add_markdown_docs(adaboost_predict_proba "cli;python;julia;go;r" "classification")
