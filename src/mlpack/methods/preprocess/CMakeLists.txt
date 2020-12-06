# Define the files we need to compile.
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
scaling_model.hpp
scaling_model_impl.hpp
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all mlpack sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

#add_cli_executable(preprocess_stats)

add_cli_executable(preprocess_split)
add_python_binding(preprocess_split)
add_julia_binding(preprocess_split)
add_go_binding(preprocess_split)
add_r_binding(preprocess_split)
add_markdown_docs(preprocess_split "cli;python;julia;go;r" "preprocessing")

add_cli_executable(preprocess_binarize)
add_python_binding(preprocess_binarize)
add_julia_binding(preprocess_binarize)
add_go_binding(preprocess_binarize)
add_r_binding(preprocess_binarize)
add_markdown_docs(preprocess_binarize "cli;python;julia;go;r" "preprocessing")

add_cli_executable(preprocess_describe)
add_python_binding(preprocess_describe)
add_julia_binding(preprocess_describe)
add_go_binding(preprocess_describe)
add_r_binding(preprocess_describe)
add_markdown_docs(preprocess_describe "cli;python;julia;go;r" "preprocessing")

#add_cli_executable(preprocess_scan)

add_cli_executable(preprocess_imputer)
#add_go_binding(preprocess_imputer)
#add_python_binding(preprocess_imputer)
#add_julia_binding(preprocess_imputer)
#add_r_binding(preprocess_imputer)
add_markdown_docs(preprocess_imputer "cli" "preprocessing")

add_cli_executable(preprocess_scale)
add_python_binding(preprocess_scale)
add_go_binding(preprocess_scale)
add_julia_binding(preprocess_scale)
add_r_binding(preprocess_scale)
add_markdown_docs(preprocess_scale "cli;python;julia;go;r" "preprocessing")

add_cli_executable(preprocess_one_hot_encoding)
add_python_binding(preprocess_one_hot_encoding)
add_go_binding(preprocess_one_hot_encoding)
add_julia_binding(preprocess_one_hot_encoding)
add_r_binding(preprocess_one_hot_encoding)
add_markdown_docs(preprocess_one_hot_encoding "cli;python;julia;go;r"
    "preprocessing")

if (STB_AVAILABLE)
  add_cli_executable(image_converter)
  add_python_binding(image_converter)
  add_julia_binding(image_converter)
  add_go_binding(image_converter)
  add_r_binding(image_converter)
  add_markdown_docs(image_converter "cli;python;julia;go;r" "preprocessing")
endif ()
