/**
 * @file methods/linear_svm/linear_svm_function_impl.hpp
 * @author Shikhar Bhardwaj
 * @author Ayush Chamoli
 *
 * Implementation of the hinge loss function for training a linear SVM with the
 * parallel SGD algorithm
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_LINEAR_SVM_LINEAR_SVM_FUNCTION_IMPL_HPP
#define MLPACK_METHODS_LINEAR_SVM_LINEAR_SVM_FUNCTION_IMPL_HPP

#include <mlpack/core/math/make_alias.hpp>
#include <mlpack/core/math/shuffle_data.hpp>

// In case it hasn't been included yet.
#include "linear_svm_function.hpp"

namespace mlpack {

template<typename MatType, typename ParametersType>
LinearSVMFunction<MatType, ParametersType>::LinearSVMFunction(
    const MatType& datasetIn,
    const arma::Row<size_t>& labels,
    const size_t numClasses,
    const double lambda,
    const double delta,
    const bool fitIntercept) :
    numClasses(numClasses),
    lambda(lambda),
    delta(delta),
    fitIntercept(fitIntercept)
{
  MakeAlias(dataset, datasetIn, datasetIn.n_rows, datasetIn.n_cols, 0, false);

  InitializeWeights(initialPoint, dataset.n_rows, numClasses, fitIntercept);
  initialPoint *= 0.005;

  // Calculate the label matrix.
  GetGroundTruthMatrix(labels, groundTruth);
}

/**
 * Initializes parameter weights to random values taken from a scaled standard
 * normal distribution. The weights cannot be initialized to zero, as that will
 * lead to each class output being the same.
 */
template<typename MatType, typename ParametersType>
void LinearSVMFunction<MatType, ParametersType>::InitializeWeights(
    ParametersType& weights,
    const size_t featureSize,
    const size_t numClasses,
    const bool fitIntercept)
{
  // Initialize values to 0.005 * r. 'r' is a matrix of random values taken from
  // a Gaussian distribution with mean zero and variance one.
  if (fitIntercept)
    weights.randn(featureSize + 1, numClasses);
  else
    weights.randn(featureSize, numClasses);
  weights *= 0.005;
}

/**
 * This is equivalent to applying the indicator function to the training
 * labels. The output is in the form of a matrix, which leads to simpler
 * calculations in the Evaluate() and Gradient() methods.
 */
template<typename MatType, typename ParametersType>
void LinearSVMFunction<MatType, ParametersType>::GetGroundTruthMatrix(
    const arma::Row<size_t>& labels,
    typename LinearSVMFunction<MatType, ParametersType>::SparseMatType&
        groundTruth) const
{
  // Calculate the ground truth matrix according to the labels passed. The
  // ground truth matrix is a matrix of dimensions 'numClasses * numExamples',
  // where each column contains a single entry of '1', marking the label
  // corresponding to that example.

  // Row pointers and column pointers corresponding to the entries.
  arma::uvec rowPointers(labels.n_elem);
  arma::uvec colPointers(labels.n_elem + 1);

  // colPointers[0] needs to be set to 0.
  colPointers[0] = 0;

  // Row pointers are the labels of the examples, and column pointers are the
  // number of cumulative entries made uptil that column.
  for (size_t i = 0; i < labels.n_elem; ++i)
  {
    rowPointers(i) = labels(i);
    colPointers(i + 1) = i + 1;
  }

  // All entries are '1'.
  DenseColType values;
  values.ones(labels.n_elem);

  // Calculate the matrix.
  groundTruth = SparseMatType(rowPointers, colPointers, values, numClasses,
                              labels.n_elem);
}

/**
 * Shuffle the data.
 */
template<typename MatType, typename ParametersType>
void LinearSVMFunction<MatType, ParametersType>::Shuffle()
{
  // Determine new ordering.
  arma::uvec ordering = arma::shuffle(arma::linspace<arma::uvec>(0,
      dataset.n_cols - 1, dataset.n_cols));

  // Re-sort data.
  MatType newData = dataset.cols(ordering);
  ClearAlias(dataset);
  dataset = std::move(newData);

  // Assemble data for batch constructor.  We need reverse orderings though...
  arma::uvec reverseOrdering(ordering.n_elem);
  for (size_t i = 0; i < ordering.n_elem; ++i)
    reverseOrdering[ordering[i]] = i;

  arma::umat newLocations(2, groundTruth.n_nonzero);
  DenseColType values(groundTruth.n_nonzero);
  typename SparseMatType::const_iterator it = groundTruth.begin();
  size_t loc = 0;
  while (it != groundTruth.end())
  {
    newLocations(0, loc) = it.row();
    newLocations(1, loc) = reverseOrdering(it.col());
    values(loc) = (*it);

    ++it;
    ++loc;
  }

  groundTruth = SparseMatType(newLocations, values, groundTruth.n_rows,
                              groundTruth.n_cols);
}

template<typename MatType, typename ParametersType>
typename LinearSVMFunction<MatType, ParametersType>::ElemType
LinearSVMFunction<MatType, ParametersType>::Evaluate(
    const ParametersType& parameters) const
{
  // The objective function is the hinge loss function and it is
  // calculated over all the training examples.

  // Calculate the loss and regularization terms.
  // L_i = Σ_i Σ_m max(0, Δ + (w_m x_i + b_m) - (w_{y_i} x_i + b_{y_i}))
  // where (m != y_i)
  ElemType loss, regularization;

  // Scores for each class are evaluated.
  DenseMatType scores;

  // Check intercept condition.
  if (!fitIntercept)
  {
    scores = parameters.t() * dataset;
  }
  else
  {
    // When using `fitIntercept` we need to add the `b_i` term explicitly.
    // The first `parameters.n_rows - 1` rows of parameters holds the value
    // of Weights `w_i`, and the last row holds `b_i`.
    // On calculating the score, we add `b_i` term to each element of
    // `i_th` row of `scores`.
    scores = parameters.rows(0, dataset.n_rows - 1).t() * dataset
        + repmat(parameters.row(dataset.n_rows).t(), 1, dataset.n_cols);
  }

  // Evaluate the margin by the following steps:
  //  - Subtracting the score of correct class from all the class scores.
  //  - Adding the margin parameter `delta`.
  //  - Removing the `delta` parameter from correct class label in each
  //    column.
  DenseMatType margin = scores - (repmat(ones(numClasses).t()
      * (scores % groundTruth), numClasses, 1)) + delta
      - (delta * groundTruth);

  // The Hinge Loss Function
  loss = accu(arma::clamp(margin, 0.0, DBL_MAX)) / dataset.n_cols;

  // Adding the regularization term.
  constexpr ElemType half = ((ElemType) 0.5);
  regularization = half * lambda * dot(parameters, parameters);

  return loss + regularization;
}

template<typename MatType, typename ParametersType>
typename LinearSVMFunction<MatType, ParametersType>::ElemType
LinearSVMFunction<MatType, ParametersType>::Evaluate(
    const ParametersType& parameters,
    const size_t firstId,
    const size_t batchSize) const
{
  const size_t lastId = firstId + batchSize - 1;

  // Calculate the loss and regularization terms.
  ElemType loss, regularization, cost;

  // Scores for each class are evaluated.
  DenseMatType scores;

  // Check intercept condition.
  if (!fitIntercept)
  {
    scores = parameters.t() * dataset.cols(firstId, lastId);
  }
  else
  {
    scores = parameters.rows(0, dataset.n_rows - 1).t()
        * dataset.cols(firstId, lastId)
        + repmat(parameters.row(dataset.n_rows).t(), 1,
        dataset.n_cols);
  }

  DenseMatType margin = scores - (repmat(ones(numClasses).t()
      * (scores % groundTruth.cols(firstId, lastId)), numClasses, 1))
      + delta - (delta * groundTruth.cols(firstId, lastId));

  // The Hinge Loss Function
  loss = accu(arma::clamp(margin, 0.0, DBL_MAX));
  loss /= batchSize;

  // Adding the regularization term.
  constexpr ElemType half = ((ElemType) 0.5);
  regularization = half * lambda * dot(parameters, parameters);

  cost = loss + regularization;
  return cost;
}

template<typename MatType, typename ParametersType>
template<typename GradType>
void LinearSVMFunction<MatType, ParametersType>::Gradient(
    const ParametersType& parameters,
    GradType& gradient) const
{
  // The objective is to minimize the loss, which is evaluated as the sum
  // of all the positive elements of `margin` matrix.
  // So, we focus of these positive elements and reduce them.
  // Also, we need to increase the score of the correct class.

  // Scores for each class are evaluated.
  DenseMatType scores;

  if (!fitIntercept)
  {
    scores = parameters.t() * dataset;
  }
  else
  {
    scores = parameters.rows(0, dataset.n_rows - 1).t() * dataset
        + repmat(parameters.row(dataset.n_rows).t(), 1, dataset.n_cols);
  }

  DenseMatType margin = scores - (repmat(ones(numClasses).t()
      * (scores % groundTruth), numClasses, 1)) + delta
      - (delta * groundTruth);

  // An element of `mask` matrix holds `1` corresponding to
  // each positive element of `margin` matrix.
  DenseMatType mask = margin.for_each([](arma::mat::elem_type& val)
      { val = (val > 0) ? 1: 0; });

  DenseMatType difference = groundTruth
      % (-repmat(sum(mask), numClasses, 1)) + mask;

  // The gradient is evaluated as follows:
  //  - Add `x_i` to `w_j` if `margin_i_m`is positive.
  //  - Subtract `x_i` from `w_y_i` for each positive
  //    `margin_i_j`.
  //  - Take the average over the size of dataset.
  //  - Add the regularization parameter.

  // Check intercept condition
  if (!fitIntercept)
  {
    gradient = dataset * difference.t();
  }
  else
  {
    gradient.set_size(arma::size(parameters));
    gradient.submat(0, 0, parameters.n_rows - 2, parameters.n_cols - 1) =
        dataset * difference.t();
    gradient.row(parameters.n_rows - 1) =
        ones<DenseRowType>(dataset.n_cols) * difference.t();
  }

  gradient /= dataset.n_cols;

  // Adding the regularization contribution to the gradient.
  gradient += lambda * parameters;
}

template<typename MatType, typename ParametersType>
template<typename GradType>
void LinearSVMFunction<MatType, ParametersType>::Gradient(
    const ParametersType& parameters,
    const size_t firstId,
    GradType& gradient,
    const size_t batchSize) const
{
  const size_t lastId = firstId + batchSize - 1;

  // Scores for each class are evaluated.
  DenseMatType scores;

  // Check intercept condition.
  if (!fitIntercept)
  {
    scores = parameters.t() * dataset.cols(firstId, lastId);
  }
  else
  {
    scores = parameters.rows(0, dataset.n_rows - 1).t()
        * dataset.cols(firstId, lastId)
        + repmat(parameters.row(dataset.n_rows).t(), 1, batchSize);
  }

  DenseMatType margin = scores - (repmat(ones(numClasses).t()
      * (scores % groundTruth.cols(firstId, lastId)), numClasses, 1))
      + delta - (delta * groundTruth.cols(firstId, lastId));

  // For each sample, find the total number of classes where
  // ( margin > 0 ).
  DenseMatType mask = margin.for_each([](arma::mat::elem_type& val)
      { val = (val > 0) ? 1: 0; });

  DenseMatType difference = groundTruth.cols(firstId, lastId)
      % (-repmat(sum(mask), numClasses, 1)) + mask;

  // Check intercept condition
  if (!fitIntercept)
  {
    gradient = dataset.cols(firstId, lastId) * difference.t();
  }
  else
  {
    gradient.set_size(arma::size(parameters));
    gradient.submat(0, 0, parameters.n_rows - 2, parameters.n_cols - 1) =
        dataset.cols(firstId, lastId) * difference.t();
    gradient.row(parameters.n_rows - 1) =
        ones<DenseRowType>(batchSize) * difference.t();
  }

  gradient /= batchSize;

  // Adding the regularization contribution to the gradient.
  gradient += lambda * parameters;
}

template<typename MatType, typename ParametersType>
template<typename GradType>
typename LinearSVMFunction<MatType, ParametersType>::ElemType
LinearSVMFunction<MatType, ParametersType>::EvaluateWithGradient(
    const ParametersType& parameters,
    GradType& gradient) const
{
  ElemType loss, regularization, cost;

  // Scores for each class are evaluated.
  DenseMatType scores;

  if (!fitIntercept)
  {
    scores = parameters.t() * dataset;
  }
  else
  {
    scores = parameters.rows(0, dataset.n_rows - 1).t() * dataset
        + repmat(parameters.row(dataset.n_rows).t(), 1,
        dataset.n_cols);
  }

  DenseMatType margin = scores - (repmat(
      ones<DenseColType>(numClasses).t() * (scores % groundTruth),
      numClasses, 1)) + delta - (delta * groundTruth);

  // For each sample, find the total number of classes where
  // ( margin > 0 ).
  DenseMatType mask = margin.for_each([](ElemType& val)
      { val = (val > 0) ? 1: 0; });

  DenseMatType difference = groundTruth
      % (-repmat(sum(mask), numClasses, 1)) + mask;

  // Check intercept condition
  if (!fitIntercept)
  {
    gradient = dataset * difference.t();
  }
  else
  {
    gradient.set_size(arma::size(parameters));
    gradient.submat(0, 0, parameters.n_rows - 2, parameters.n_cols - 1) =
            dataset * difference.t();
    gradient.row(parameters.n_rows - 1) =
            ones<DenseRowType>(dataset.n_cols) * difference.t();
  }

  gradient /= dataset.n_cols;

  // Adding the regularization contribution to the gradient.
  gradient += lambda * parameters;

  // The Hinge Loss Function
  loss = accu(arma::clamp(margin, 0.0, DBL_MAX));
  loss /= dataset.n_cols;

  // Adding the regularization term.
  constexpr ElemType half = ((ElemType) 0.5);
  regularization = half * lambda * dot(parameters, parameters);

  cost = loss + regularization;
  return cost;
}

template<typename MatType, typename ParametersType>
template<typename GradType>
typename LinearSVMFunction<MatType, ParametersType>::ElemType
LinearSVMFunction<MatType, ParametersType>::EvaluateWithGradient(
    const ParametersType& parameters,
    const size_t firstId,
    GradType& gradient,
    const size_t batchSize) const
{
  const size_t lastId = firstId + batchSize - 1;

  // Calculate the loss and regularization terms.
  ElemType loss, regularization, cost;

  // Scores for each class are evaluated.
  DenseMatType scores;

  // Check intercept condition.
  if (!fitIntercept)
  {
    scores = parameters.t() * dataset.cols(firstId, lastId);
  }
  else
  {
    scores = parameters.rows(0, dataset.n_rows - 1).t()
        * dataset.cols(firstId, lastId)
        + repmat(parameters.row(dataset.n_rows).t(), 1, (lastId - firstId + 1));
  }

  DenseMatType margin = scores - (repmat(ones(numClasses).t()
      * (scores % groundTruth.cols(firstId, lastId)), numClasses, 1))
      + delta - (delta * groundTruth.cols(firstId, lastId));

  // For each sample, find the total number of classes where
  // ( margin > 0 ).
  DenseMatType mask = margin.for_each([](arma::mat::elem_type& val)
      { val = (val > 0) ? 1: 0; });

  DenseMatType difference = groundTruth.cols(firstId, lastId)
      % (-repmat(sum(mask), numClasses, 1)) + mask;

  // Check intercept condition
  if (!fitIntercept)
  {
    gradient = dataset.cols(firstId, lastId) * difference.t();
  }
  else
  {
    gradient.set_size(arma::size(parameters));
    gradient.submat(0, 0, parameters.n_rows - 2, parameters.n_cols - 1) =
        dataset.cols(firstId, lastId) * difference.t();
    gradient.row(parameters.n_rows - 1) =
        ones<DenseRowType>(batchSize) * difference.t();
  }

  gradient /= batchSize;


  // Adding the regularization contribution to the gradient.
  gradient += lambda * parameters;

  // The Hinge Loss Function
  loss = accu(arma::clamp(margin, 0.0, DBL_MAX));
  loss /= batchSize;

  // Adding the regularization term.
  constexpr ElemType half = ((ElemType) 0.5);
  regularization = half * lambda * dot(parameters, parameters);

  cost = loss + regularization;
  return cost;
}

template<typename MatType, typename ParametersType>
size_t LinearSVMFunction<MatType, ParametersType>::NumFunctions() const
{
  // The number of points in the dataset is the number of functions, as this
  // is a data dependent function.
  return dataset.n_cols;
}

} // namespace mlpack


#endif // MLPACK_METHODS_LINEAR_SVM_LINEAR_SVM_FUNCTION_IMPL_HPP
