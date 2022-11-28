# Utility macro to add bindings for all languages.
macro(add_all_bindings directory name type)
  add_category(${name} ${type})
  add_cli_executable(${directory} ${name})
  add_python_binding(${directory} ${name})
  add_julia_binding(${directory} ${name})
  add_go_binding(${directory} ${name})
  add_r_binding(${directory} ${name})
  add_markdown_docs(${directory} ${name} "cli;python;julia;go;r" "")
endmacro()

# First, define all the "regular" bindings that exist for all languages.
add_all_bindings(approx_kfn approx_kfn "geometry")
add_all_bindings(bayesian_linear_regression bayesian_linear_regression
    "regression")
add_all_bindings(cf cf "misc. / other")
add_all_bindings(dbscan dbscan "clustering")
add_all_bindings(decision_tree decision_tree "classification")
add_all_bindings(det det "misc. / other")
add_all_bindings(emst emst "geometry")
add_all_bindings(fastmks fastmks "geometry")
add_all_bindings(gmm gmm_train "clustering")
add_all_bindings(gmm gmm_generate "clustering")
add_all_bindings(gmm gmm_probability "clustering")
add_all_bindings(hmm hmm_train "misc. / other")
add_all_bindings(hmm hmm_generate "misc. / other")
add_all_bindings(hmm hmm_loglik "misc. / other")
add_all_bindings(hmm hmm_viterbi "misc. / other")
add_all_bindings(hoeffding_trees hoeffding_tree "clustering")
add_all_bindings(kde kde "misc. / other")
add_all_bindings(kernel_pca kernel_pca "transformations")
add_all_bindings(kmeans kmeans "clustering")
add_all_bindings(lars lars "regression")
add_all_bindings(linear_svm linear_svm "classification")
add_all_bindings(lmnn lmnn "transformations")
add_all_bindings(local_coordinate_coding local_coordinate_coding
    "transformations")
add_all_bindings(logistic_regression logistic_regression "classification")
add_all_bindings(lsh lsh "geometry")
add_all_bindings(mean_shift mean_shift "clustering")
add_all_bindings(naive_bayes nbc "classification")
add_all_bindings(nca nca "transformations")
add_all_bindings(neighbor_search knn "geometry")
add_all_bindings(neighbor_search kfn "geometry")
add_all_bindings(nmf nmf "misc. / other")
add_all_bindings(pca pca "transformations")
add_all_bindings(perceptron perceptron "classification")
add_all_bindings(preprocess preprocess_split "preprocessing")
add_all_bindings(preprocess preprocess_binarize "preprocessing")
add_all_bindings(preprocess preprocess_describe "preprocessing")
add_all_bindings(preprocess preprocess_scale "preprocessing")
add_all_bindings(preprocess preprocess_one_hot_encoding "preprocessing")
add_all_bindings(radical radical "transformations")
add_all_bindings(random_forest random_forest "classification")
add_all_bindings(rann krann "geometry")
add_all_bindings(softmax_regression softmax_regression "classification")
add_all_bindings(sparse_coding sparse_coding "transformations")

# Now, define the "special" bindings that are different somehow.

# Adaboost has a Python wrapper class that encapsulates its functionality in a
# few methods.
add_category(adaboost "classification")
group_bindings(adaboost adaboost "train;classify;probabilities")
add_python_wrapper(adaboost adaboost)

add_cli_executable(adaboost adaboost)
add_julia_binding(adaboost adaboost)
add_go_binding(adaboost adaboost)
add_r_binding(adaboost adaboost)
add_markdown_docs(adaboost adaboost "cli;python;julia;go;r" "python")

# Linear regression has a Python wrapper class that encapsulates its
# functionality in a few methods.
add_category(linear_regression "regression")
group_bindings(linear_regression linear_regression "train;predict")
add_python_wrapper(linear_regression linear_regression)

add_cli_executable(linear_regression linear_regression)
add_julia_binding(linear_regression linear_regression)
add_go_binding(linear_regression linear_regression)
add_r_binding(linear_regression linear_regression)
add_markdown_docs(linear_regression linear_regression "cli;python;julia;go;r"
    "python")

# The imputer is only defined for CLI bindings right now.
add_category(preprocess_imputer "preprocessing")
add_cli_executable(preprocess preprocess_imputer)
add_markdown_docs(preprocess preprocess_imputer "cli" "")

# The image converter is only enabled if STB is available.
if (STB_AVAILABLE)
  add_all_bindings(preprocess image_converter "preprocessing")
endif()

# Range search provides a vector of vector of results, and this is only
# supported for the CLI bindings.
add_category(range_search "geometry")
add_cli_executable(range_search range_search)
add_markdown_docs(range_search range_search "cli" "")
