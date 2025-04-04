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
add_all_bindings(approx_kfn approx_kfn "Geometry")
add_all_bindings(bayesian_linear_regression bayesian_linear_regression
    "Regression")
add_all_bindings(cf cf "Misc. / Other")
add_all_bindings(dbscan dbscan "Clustering")
add_all_bindings(decision_tree decision_tree "Classification")
add_all_bindings(det det "Misc. / Other")
add_all_bindings(emst emst "Geometry")
add_all_bindings(fastmks fastmks "Geometry")
add_all_bindings(gmm gmm_train "Clustering")
add_all_bindings(gmm gmm_generate "Clustering")
add_all_bindings(gmm gmm_probability "Clustering")
add_all_bindings(hmm hmm_train "Misc. / Other")
add_all_bindings(hmm hmm_generate "Misc. / Other")
add_all_bindings(hmm hmm_loglik "Misc. / Other")
add_all_bindings(hmm hmm_viterbi "Misc. / Other")
add_all_bindings(hoeffding_trees hoeffding_tree "Classification")
add_all_bindings(preprocess image_converter "Preprocessing")
add_all_bindings(kde kde "Misc. / Other")
add_all_bindings(kernel_pca kernel_pca "Transformations")
add_all_bindings(kmeans kmeans "Clustering")
add_all_bindings(lars lars "Regression")
add_all_bindings(linear_svm linear_svm "Classification")
add_all_bindings(lmnn lmnn "Transformations")
add_all_bindings(local_coordinate_coding local_coordinate_coding
    "Transformations")
add_all_bindings(logistic_regression logistic_regression "Classification")
add_all_bindings(lsh lsh "Geometry")
add_all_bindings(mean_shift mean_shift "Clustering")
add_all_bindings(naive_bayes nbc "Classification")
add_all_bindings(nca nca "Transformations")
add_all_bindings(neighbor_search knn "Geometry")
add_all_bindings(neighbor_search kfn "Geometry")
add_all_bindings(nmf nmf "Misc. / Other")
add_all_bindings(pca pca "Transformations")
add_all_bindings(perceptron perceptron "Classification")
add_all_bindings(preprocess preprocess_split "Preprocessing")
add_all_bindings(preprocess preprocess_binarize "Preprocessing")
add_all_bindings(preprocess preprocess_describe "Preprocessing")
add_all_bindings(preprocess preprocess_scale "Preprocessing")
add_all_bindings(preprocess preprocess_one_hot_encoding "Preprocessing")
add_all_bindings(radical radical "Transformations")
add_all_bindings(random_forest random_forest "Classification")
add_all_bindings(rann krann "Geometry")
add_all_bindings(softmax_regression softmax_regression "Classification")
add_all_bindings(sparse_coding sparse_coding "Transformations")

# Now, define the "special" bindings that are different somehow.

# Adaboost has a Python wrapper class that encapsulates its functionality in a
# few methods.
add_category(adaboost "Classification")
group_bindings(adaboost adaboost "train;classify;probabilities")
add_python_wrapper(adaboost adaboost)

add_cli_executable(adaboost adaboost)
add_julia_binding(adaboost adaboost)
add_go_binding(adaboost adaboost)
add_r_binding(adaboost adaboost)
add_markdown_docs(adaboost adaboost "cli;python;julia;go;r" "python")

# Linear Regression has a Python wrapper class that encapsulates its
# functionality in a few methods.
add_category(linear_regression "Regression")
group_bindings(linear_regression linear_regression "train;predict")
add_python_wrapper(linear_regression linear_regression)

add_cli_executable(linear_regression linear_regression)
add_julia_binding(linear_regression linear_regression)
add_go_binding(linear_regression linear_regression)
add_r_binding(linear_regression linear_regression)
add_markdown_docs(linear_regression linear_regression "cli;python;julia;go;r"
    "python")

# The imputer is only defined for CLI bindings right now.
add_category(preprocess_imputer "Preprocessing")
add_cli_executable(preprocess preprocess_imputer)
add_markdown_docs(preprocess preprocess_imputer "cli" "")

# Range search provides a vector of vector of results, and this is only
# supported for the CLI bindings.
add_category(range_search "Geometry")
add_cli_executable(range_search range_search)
add_markdown_docs(range_search range_search "cli" "")
