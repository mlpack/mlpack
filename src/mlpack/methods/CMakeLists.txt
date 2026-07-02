# Utility macro to add bindings for all languages.
macro(add_all_bindings directory name type)
  add_category(${name} ${type})
  add_cli_executable(${directory} ${name})
  add_python_binding(${directory} ${name})
  add_julia_binding(${directory} ${name} "false")
  add_go_binding(${directory} ${name})
  set(extra_args "${ARGN}")
  if (NOT extra_args)
    add_r_binding(${directory} ${name} "include")
  else ()
    add_r_binding(${directory} ${name} ${extra_args})
  endif ()
  add_markdown_docs(${directory} ${name} "cli;python;julia;go;r" "")
endmacro()

# Refactored classification and regression methods do not use the macro above.
#
# Commented lines below retained for spacing and visual seperation from other
# methods using aggregation macro.
#
add_category(adaboost "Classification")
group_bindings(adaboost adaboost "train;classify;probabilities")
add_python_wrapper(adaboost adaboost)
add_julia_class(adaboost adaboost)
add_cli_executable(adaboost adaboost)
add_go_binding(adaboost adaboost)
add_r_binding(adaboost adaboost "include")
add_r_binding(adaboost adaboost_train "skip")
add_r_binding(adaboost adaboost_classify "skip")
add_r_binding(adaboost adaboost_probabilities "skip")
add_markdown_docs(adaboost adaboost "cli;python;julia;go;r" "python;r;julia")
#
add_all_bindings(approx_kfn approx_kfn "Geometry")
#
add_category(bayesian_linear_regression "Regression")
group_bindings(bayesian_linear_regression bayesian_linear_regression
  "train;predict")
add_python_wrapper(bayesian_linear_regression bayesian_linear_regression)
add_julia_class(bayesian_linear_regression bayesian_linear_regression)
add_cli_executable(bayesian_linear_regression bayesian_linear_regression)
add_go_binding(bayesian_linear_regression bayesian_linear_regression)
add_r_binding(bayesian_linear_regression bayesian_linear_regression "include")
add_r_binding(bayesian_linear_regression bayesian_linear_regression_train
  "skip")
add_r_binding(bayesian_linear_regression bayesian_linear_regression_predict
  "skip")
add_markdown_docs(bayesian_linear_regression bayesian_linear_regression
  "cli;python;julia;go;r" "python;r;julia")
#
add_all_bindings(cf cf "Misc. / Other")
add_all_bindings(dbscan dbscan "Clustering")
#
add_category(decision_tree "Classification")
group_bindings(decision_tree decision_tree "train;classify;probabilities")
add_python_wrapper(decision_tree decision_tree)
add_julia_class(decision_tree decision_tree)
add_cli_executable(decision_tree decision_tree)
add_go_binding(decision_tree decision_tree)
add_r_binding(decision_tree decision_tree "include")
add_r_binding(decision_tree decision_tree_train "skip")
add_r_binding(decision_tree decision_tree_classify "skip")
add_r_binding(decision_tree decision_tree_probabilities "skip")
add_markdown_docs(decision_tree decision_tree "cli;python;julia;go;r"
    "python;r;julia")
#
add_all_bindings(det det "Misc. / Other")
add_all_bindings(emst emst "Geometry")
add_all_bindings(fastmks fastmks "Geometry")
add_all_bindings(gmm gmm_train "Clustering")
add_all_bindings(gmm gmm_generate "Clustering" "skip")
add_all_bindings(gmm gmm_probability "Clustering" "skip")
add_all_bindings(hmm hmm_train "Misc. / Other")
add_all_bindings(hmm hmm_generate "Misc. / Other" "skip")
add_all_bindings(hmm hmm_loglik "Misc. / Other" "skip")
add_all_bindings(hmm hmm_viterbi "Misc. / Other" "skip")
#
add_category(hoeffding_tree "Classification")
group_bindings(hoeffding_trees hoeffding_tree "train;classify;probabilities")
add_python_wrapper(hoeffding_trees hoeffding_tree)
add_julia_class(hoeffding_trees hoeffding_tree)
add_cli_executable(hoeffding_trees hoeffding_tree)
add_go_binding(hoeffding_trees hoeffding_tree)
add_r_binding(hoeffding_trees hoeffding_tree "include")
add_r_binding(hoeffding_trees hoeffding_tree_train "skip")
add_r_binding(hoeffding_trees hoeffding_tree_classify "skip")
add_r_binding(hoeffding_trees hoeffding_tree_probabilities "skip")
add_markdown_docs(hoeffding_trees hoeffding_tree "cli;python;julia;go;r"
    "python;r;julia")
#
add_all_bindings(preprocess image_converter "Preprocessing")
add_all_bindings(kde kde "Misc. / Other")
add_all_bindings(kernel_pca kernel_pca "Transformations")
add_all_bindings(kmeans kmeans "Clustering")
#
add_category(lars "Regression")
group_bindings(lars lars "train;predict")
add_python_wrapper(lars lars)
add_julia_class(lars lars)
add_cli_executable(lars lars)
add_go_binding(lars lars)
add_r_binding(lars lars "include")
add_r_binding(lars lars_train "skip")
add_r_binding(lars lars_predict "skip")
add_markdown_docs(lars lars "cli;python;julia;go;r" "python;r;julia")
#
add_category(linear_regression "Regression")
group_bindings(linear_regression linear_regression "train;predict")
add_python_wrapper(linear_regression linear_regression)
add_julia_class(linear_regression linear_regression)
add_cli_executable(linear_regression linear_regression)
add_go_binding(linear_regression linear_regression)
add_r_binding(linear_regression linear_regression "include")
add_r_binding(linear_regression linear_regression_train "skip")
add_r_binding(linear_regression linear_regression_predict "skip")
add_markdown_docs(linear_regression linear_regression "cli;python;julia;go;r"
    "python;r;julia")
#
add_category(linear_svm "Classification")
group_bindings(linear_svm linear_svm "train;classify;scores")
add_python_wrapper(linear_svm linear_svm)
add_julia_class(linear_svm linear_svm)
add_cli_executable(linear_svm linear_svm)
add_go_binding(linear_svm linear_svm)
add_r_binding(linear_svm linear_svm "include")
add_r_binding(linear_svm linear_svm_train "skip")
add_r_binding(linear_svm linear_svm_classify "skip")
add_r_binding(linear_svm linear_svm_scores "skip")
add_markdown_docs(linear_svm linear_svm "cli;python;julia;go;r"
    "python;r;julia")
#
add_all_bindings(lmnn lmnn "Transformations")
add_all_bindings(local_coordinate_coding local_coordinate_coding
    "Transformations")
#
add_category(logistic_regression "Classification")
group_bindings(logistic_regression logistic_regression
  "train;classify;probabilities")
add_python_wrapper(logistic_regression logistic_regression)
add_julia_class(logistic_regression logistic_regression)
add_cli_executable(logistic_regression logistic_regression)
add_go_binding(logistic_regression logistic_regression)
add_r_binding(logistic_regression logistic_regression "include")
add_r_binding(logistic_regression logistic_regression_train "skip")
add_r_binding(logistic_regression logistic_regression_classify "skip")
add_r_binding(logistic_regression logistic_regression_probabilities "skip")
add_markdown_docs(logistic_regression logistic_regression
  "cli;python;julia;go;r" "python;r;julia")
#
add_all_bindings(lsh lsh "Geometry")
add_all_bindings(mean_shift mean_shift "Clustering")
#
add_category(nbc "Classification")
group_bindings(naive_bayes nbc "train;classify;probabilities")
add_python_wrapper(naive_bayes nbc)
add_julia_class(naive_bayes nbc)
add_cli_executable(naive_bayes nbc)
add_go_binding(naive_bayes nbc)
add_r_binding(naive_bayes nbc "include")
add_r_binding(naive_bayes nbc_train "skip")
add_r_binding(naive_bayes nbc_classify "skip")
add_r_binding(naive_bayes nbc_probabilities "skip")
add_markdown_docs(naive_bayes nbc "cli;python;julia;go;r" "python;r;julia")
#
add_all_bindings(nca nca "Transformations")
add_all_bindings(neighbor_search knn "Geometry")
add_all_bindings(neighbor_search kfn "Geometry")
add_all_bindings(nmf nmf "Misc. / Other")
add_all_bindings(pca pca "Transformations")
#
add_category(perceptron "Classification")
group_bindings(perceptron perceptron "train;classify")
add_python_wrapper(perceptron perceptron)
add_julia_class(perceptron perceptron)
add_cli_executable(perceptron perceptron)
add_go_binding(perceptron perceptron)
add_r_binding(perceptron perceptron "include")
add_r_binding(perceptron perceptron_train "skip")
add_r_binding(perceptron perceptron_classify "skip")
add_markdown_docs(perceptron perceptron "cli;python;julia;go;r"
    "python;r;julia")
#
add_all_bindings(preprocess preprocess_split "Preprocessing")
add_all_bindings(preprocess preprocess_binarize "Preprocessing")
add_all_bindings(preprocess preprocess_describe "Preprocessing")
add_all_bindings(preprocess preprocess_scale "Preprocessing")
add_all_bindings(preprocess preprocess_one_hot_encoding "Preprocessing")
add_all_bindings(radical radical "Transformations")
#
add_category(random_forest "Classification")
group_bindings(random_forest random_forest "train;classify;probabilities")
add_python_wrapper(random_forest random_forest)
add_julia_class(random_forest random_forest)
add_cli_executable(random_forest random_forest)
add_go_binding(random_forest random_forest)
add_r_binding(random_forest random_forest "include")
add_r_binding(random_forest random_forest_train "skip")
add_r_binding(random_forest random_forest_classify "skip")
add_r_binding(random_forest random_forest_probabilities "skip")
add_markdown_docs(random_forest random_forest "cli;python;julia;go;r"
    "python;r;julia")
#
add_all_bindings(rann krann "Geometry")
#
#add_all_bindings(softmax_regression softmax_regression "Classification")
add_category(softmax_regression "Classification")
group_bindings(softmax_regression softmax_regression "train;classify;probabilities")
add_python_wrapper(softmax_regression softmax_regression)
add_julia_class(softmax_regression softmax_regression)
add_cli_executable(softmax_regression softmax_regression)
add_go_binding(softmax_regression softmax_regression)
add_r_binding(softmax_regression softmax_regression "include")
add_r_binding(softmax_regression softmax_regression_train "skip")
add_r_binding(softmax_regression softmax_regression_classify "skip")
add_r_binding(softmax_regression softmax_regression_probabilities "skip")
add_markdown_docs(softmax_regression softmax_regression "cli;python;julia;go;r"
    "python;r;julia")
#
add_all_bindings(sparse_coding sparse_coding "Transformations")

# Now, define the "special" bindings that are different somehow.

# The imputer is only defined for CLI bindings right now.
add_category(preprocess_imputer "Preprocessing")
add_cli_executable(preprocess preprocess_imputer)
add_markdown_docs(preprocess preprocess_imputer "cli" "")

# Range search provides a vector of vector of results, and this is only
# supported for the CLI bindings.
add_category(range_search "Geometry")
add_cli_executable(range_search range_search)
add_markdown_docs(range_search range_search "cli" "")
