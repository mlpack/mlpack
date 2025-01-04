<object data="../img/pipeline-top-4.svg" type="image/svg+xml" id="pipeline-top">
</object>

# Modeling

mlpack contains numerous different machine learning algorithms that can be used
for modeling.

*Note: this section is under construction and not all functionality is
documented yet.*

## Classification

Classify points as discrete labels (`0`, `1`, `2`, ...).

 * [`AdaBoost`](methods/adaboost.md): Adaptive Boosting
 * [`DecisionTree`](methods/decision_tree.md): ID3-style decision tree
   classifier
 * [`HoeffdingTree`](methods/hoeffding_tree.md): streaming/incremental decision
   tree classifier
 * [`LinearSVM`](methods/linear_svm.md): simple linear support vector machine
   classifier
 * [`LogisticRegression`](methods/logistic_regression.md): L2-regularized
   logistic regression (two-class only)
 * [`NaiveBayesClassifier`](methods/naive_bayes_classifier.md): simple
   multi-class naive Bayes classifier
 * [`Perceptron`](methods/perceptron.md): simple Perceptron classifier
 * [`RandomForest`](methods/random_forest.md): parallelized random forest
   classifier
 * [`SoftmaxRegression`](methods/softmax_regression.md): L2-regularized
   softmax regression (i.e. multi-class logistic regression)

## Regression

Predict continuous values.

 * [`BayesianLinearRegression`](methods/bayesian_linear_regression.md):
   Bayesian L2-penalized linear regression
 * [`DecisionTreeRegressor`](methods/decision_tree_regressor.md): ID3-style
   decision tree regressor
 * [`LARS`](methods/lars.md): Least Angle Regression (LARS), L1-regularized and
   L2-regularized
 * [`LinearRegression`](methods/linear_regression.md): L2-regularized linear
   regression (ridge regression)

## Clustering

***NOTE:*** this documentation is still under construction and so some
algorithms that mlpack implements are not yet listed here.  For now, see
[the mlpack/methods directory](https://github.com/mlpack/mlpack/tree/master/src/mlpack/methods)
for a full list of algorithms.

Group points into clusters.

 * [`MeanShift`](methods/mean_shift.md): clustering with the density-based mean
   shift algorithm

## Geometric algorithms

***NOTE:*** this documentation is still under construction and so no geometric
algorithms in mlpack are documented yet.  For now, see
[the mlpack/methods directory](https://github.com/mlpack/mlpack/tree/master/src/mlpack/methods)
for a full list of algorithms.

Computations based on distance metrics.

<!-- TODO: add some -->
