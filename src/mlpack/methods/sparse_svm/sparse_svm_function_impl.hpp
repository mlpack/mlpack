/**
 * @file sparse_svm_function_impl.hpp
 * @author Shikhar Bhardwaj
 * @author Ayush Chamoli
 *
 * Implementation of the hinge loss function for training a sparse SVM with the
 * parallel SGD algorithm
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_SPARSE_SVM_SPARSE_SVM_FUNCTION_IMPL_HPP
#define MLPACK_METHODS_SPARSE_SVM_SPARSE_SVM_FUNCTION_IMPL_HPP

#include <mlpack/core/math/make_alias.hpp>
#include <mlpack/core/math/shuffle_data.hpp>

// In case it hasn't been included yet.
#include "sparse_svm_function.hpp"

namespace mlpack {
namespace svm {

template <typename MatType>
SparseSVMFunction<MatType>::SparseSVMFunction(
  const MatType& dataset,
  const arma::Row<size_t>& labels,
  const size_t numClasses,
  const double lambda) :
  dataset(math::MakeAlias(const_cast<MatType&>(dataset), false)),
  labels(math::MakeAlias(const_cast<arma::Row<size_t>&>(labels),
      false)),
  numClasses(numClasses),
  lambda(lambda)
{
  // Initialize the parameters to suitable values.
  initialPoint = InitializeWeights();

  // Calculate the label matrix.
  GetGroundTruthMatrix(labels, groundTruth);
}

/**
 * Initializes parameter weights to random values taken from a scaled standard
 * normal distribution. The weights cannot be initialized to zero, as that will
 * lead to each class output being the same.
 */
template <typename MatType>
const arma::mat SparseSVMFunction<MatType>::InitializeWeights()
{
  return InitializeWeights(dataset.n_rows, numClasses);
}

template <typename MatType>
const arma::mat SparseSVMFunction<MatType>::InitializeWeights(
        const size_t featureSize,
        const size_t numClasses)
{
  arma::mat parameters;
  InitializeWeights(parameters, featureSize, numClasses);
  return parameters;
}

template <typename MatType>
void SparseSVMFunction<MatType>::InitializeWeights(
        arma::mat &weights,
        const size_t featureSize,
        const size_t numClasses)
{
  // Initialize values to 0.005 * r. 'r' is a matrix of random values taken from
  // a Gaussian distribution with mean zero and variance one.
  weights.randn(numClasses, featureSize);
  weights *= 0.005;
}

/**
 * This is equivalent to applying the indicator function to the training
 * labels. The output is in the form of a matrix, which leads to simpler
 * calculations in the Evaluate() and Gradient() methods.
 */
template <typename MatType>
void SparseSVMFunction<MatType>::GetGroundTruthMatrix(
        const arma::Row<size_t>& labels, arma::sp_mat& groundTruth)
{
  // Calculate the ground truth matrix according to the labels passed. The
  // ground truth matrix is a matrix of dimensions 'numClasses * numExamples',
  // where each column contains a single entry of '1', marking the label
  // corresponding to that example.

  // Row pointers and column pointers corresponding to the entries.
  arma::uvec rowPointers(labels.n_elem);
  arma::uvec colPointers(labels.n_elem + 1);

  // Row pointers are the labels of the examples, and column pointers are the
  // number of cumulative entries made uptil that column.
  for (size_t i = 0; i < labels.n_elem; i++)
  {
    rowPointers(i) = labels(i);
    colPointers(i + 1) = i + 1;
  }

  // All entries are '1'.
  arma::vec values;
  values.ones(labels.n_elem);

  // Calculate the matrix.
  groundTruth = arma::sp_mat(rowPointers, colPointers, values, numClasses,
                             labels.n_elem);
}

/**
 * Shuffle the data.
 */
template <typename MatType>
void SparseSVMFunction<MatType>::Shuffle()
{
  // Determine new ordering.
  arma::uvec ordering = arma::shuffle(arma::linspace<arma::uvec>(0,
      dataset.n_cols - 1, dataset.n_cols));

  // Re-sort data.
  arma::mat newData = dataset.cols(ordering);
  math::ClearAlias(dataset);
  dataset = std::move(newData);

  // Assemble data for batch constructor.  We need reverse orderings though...
  arma::uvec reverseOrdering(ordering.n_elem);
  for (size_t i = 0; i < ordering.n_elem; ++i)
    reverseOrdering[ordering[i]] = i;

  arma::umat newLocations(2, groundTruth.n_nonzero);
  arma::vec values(groundTruth.n_nonzero);
  arma::sp_mat::const_iterator it = groundTruth.begin();
  size_t loc = 0;
  while (it != groundTruth.end())
  {
    newLocations(0, loc) = reverseOrdering(it.col());
    newLocations(1, loc) = it.row();
    values(loc) = (*it);

    ++it;
    ++loc;
  }

  groundTruth = arma::sp_mat(newLocations, values, groundTruth.n_rows,
                             groundTruth.n_cols);
}

template <typename MatType>
double SparseSVMFunction<MatType>::Evaluate(
    const arma::mat& parameters)
{
  // The objective function is the hinge loss function and it is
  // calculated over all the training examples.

  // Calculate the loss and regularization terms.
  double loss, regularization, cost;

  arma::mat scores = dataset.t() * parameters;
  arma::mat correct = scores
      % arma::conv_to<arma::mat>::from(groundTruth).t();
  arma::mat margin = scores - arma::repmat(correct
      * arma::ones(numClasses), 1, numClasses) + 1
      - groundTruth.t();

  // The Hinge Loss Function
  loss = arma::accu(arma::clamp(margin, 0.0, margin.max()));
  loss /= dataset.n_cols;

  // Adding the regularization term.
  regularization = 0.5 * lambda * arma::dot(parameters,
      parameters);

  cost = loss + regularization;
  return cost;
}

template <typename MatType>
double SparseSVMFunction<MatType>::Evaluate(
    const arma::mat& parameters,
    const size_t firstId,
    const size_t batchSize)
{
  const size_t lastId = firstId + batchSize - 1;

  // Calculate the loss and regularization terms.
  double loss, regularization, cost;

  arma::mat scores = dataset.cols(firstId, lastId).t()
      * parameters;
  arma::mat correct = scores
      % arma::conv_to<arma::mat>::from(groundTruth).cols(firstId, lastId).t();
  arma::mat margin = scores - arma::repmat(correct
      * arma::ones(numClasses), 1, numClasses) + 1
      - arma::conv_to<arma::mat>::from(groundTruth).cols(firstId, lastId).t();

  // The Hinge Loss Function
  loss = arma::accu(arma::clamp(margin, 0.0, margin.max()));
  loss /= batchSize;

  // Adding the regularization term.
  regularization = 0.5 * lambda * arma::dot(parameters,
      parameters);

  cost = loss + regularization;
  return cost;
}

template <typename MatType>
template <typename GradType>
void SparseSVMFunction<MatType>::Gradient(
    const arma::mat& parameters,
    GradType& gradient)
{
  arma::mat scores = dataset.t() * parameters;
  arma::mat correct = scores
      % arma::conv_to<arma::mat>::from(groundTruth).t();
  arma::mat margin = scores - arma::repmat(correct
      * arma::ones(numClasses), 1, numClasses) + 1
      - groundTruth.t();

  // For each sample, find the total number of classes where
  // ( margin > 0 )
  arma::mat mask = margin.for_each([](arma::mat::elem_type& val)
      { val = (val > 0) ? 1: 0; });

  arma::mat incorrectLabels = arma::conv_to<arma::mat>::from(groundTruth).t()
      % (-arma::repmat(arma::sum(mask, 1), 1, numClasses));
  arma::mat difference = incorrectLabels + mask;

  gradient = dataset * difference;
  gradient /= dataset.n_cols;

  // Adding the regularization contribution to the gradient.
  gradient += lambda * parameters;
}

template <typename MatType>
template <typename GradType>
void SparseSVMFunction<MatType>::Gradient(
    const arma::mat& parameters,
    const size_t firstId,
    GradType& gradient,
    const size_t batchSize)
{
  const size_t lastId = firstId + batchSize - 1;
  arma::mat scores = dataset.cols(firstId, lastId).t()
      * parameters;
  arma::mat correct = scores
      % arma::conv_to<arma::mat>::from(groundTruth).cols(firstId, lastId).t();
  arma::mat margin = scores - arma::repmat(correct
      * arma::ones(numClasses), 1, numClasses) + 1
      - arma::conv_to<arma::mat>::from(groundTruth).cols(firstId, lastId).t();

  // For each sample, find the total number of classes where
  // ( margin > 0 )
  arma::mat mask = margin.for_each([](arma::mat::elem_type& val)
      { val = (val > 0) ? 1: 0; });

  arma::mat incorrectLabels =
      arma::conv_to<arma::mat>::from(groundTruth).cols(firstId, lastId).t() %
      (-arma::repmat(arma::sum(mask.rows(firstId, lastId), 1), 1, numClasses));
  arma::mat difference = incorrectLabels + mask;

  gradient = dataset.cols(firstId, lastId) * difference;
  gradient /= batchSize;

  // Adding the regularization contribution to the gradient.
  gradient += lambda * parameters;
}

template <typename MatType>
template <typename GradType>
double SparseSVMFunction<MatType>::EvaluateWithGradient(
    const arma::mat& parameters,
    GradType& gradient) const
{
  double loss, regularization, cost;

  arma::mat scores = dataset.t() * parameters;
  arma::mat correct = scores
      % arma::conv_to<arma::mat>::from(groundTruth).t();
  arma::mat margin = scores - arma::repmat(correct
       * arma::ones(numClasses), 1, numClasses) + 1
       - groundTruth.t();

  // For each sample, find the total number of classes where
  // ( margin > 0 )
  arma::mat mask = margin.for_each([](arma::mat::elem_type& val)
      { val = (val > 0) ? 1: 0; });

  arma::mat incorrectLabels = arma::conv_to<arma::mat>::from(groundTruth).t()
      % (-arma::repmat(arma::sum(mask, 1), 1, numClasses));
  arma::mat difference = incorrectLabels + mask;

  gradient = dataset * difference;
  gradient /= dataset.n_cols;

  // Adding the regularization contribution to the gradient.
  gradient += lambda * parameters;

  // The Hinge Loss Function
  loss = arma::accu(arma::clamp(margin, 0.0, margin.max()));
  loss /= dataset.n_cols;

  // Adding the regularization term.
  regularization = 0.5 * lambda * arma::dot(parameters,
      parameters);

  cost = loss + regularization;
  return cost;
}

template <typename MatType>
template <typename GradType>
double SparseSVMFunction<MatType>::EvaluateWithGradient(
    const arma::mat& parameters,
    const size_t firstId,
    GradType& gradient,
    const size_t batchSize) const
{
  const size_t lastId = firstId + batchSize - 1;

  // Calculate the loss and regularization terms.
  double loss, regularization, cost;

  arma::mat scores = dataset.cols(firstId, lastId).t()
                     * parameters;
  arma::mat correct = scores
      % arma::conv_to<arma::mat>::from(groundTruth).cols(firstId, lastId).t();
  arma::mat margin = scores - arma::repmat(correct
      * arma::ones(numClasses), 1, numClasses) + 1
      - arma::conv_to<arma::mat>::from(groundTruth).cols(firstId, lastId).t();

  // For each sample, find the total number of classes where
  // ( margin > 0 )
  arma::mat mask = margin.for_each([](arma::mat::elem_type& val)
      { val = (val > 0) ? 1: 0; });

  arma::mat incorrectLabels =
      arma::conv_to<arma::mat>::from(groundTruth).cols(firstId, lastId).t() %
      (-arma::repmat(arma::sum(mask.rows(firstId, lastId), 1), 1, numClasses));
  arma::mat difference = incorrectLabels + mask;

  gradient = dataset.cols(firstId, lastId) * difference;
  gradient /= batchSize;

  // Adding the regularization contribution to the gradient.
  gradient += lambda * parameters;

  // The Hinge Loss Function
  loss = arma::accu(arma::clamp(margin, 0.0, margin.max()));
  loss /= batchSize;

  // Adding the regularization term.
  regularization = 0.5 * lambda * arma::dot(parameters,
      parameters);

  cost = loss + regularization;
  return cost;
}

template <typename MatType>
size_t SparseSVMFunction<MatType>::NumFunctions() const
{
  // The number of points in the dataset is the number of functions, as this
  // is a data dependent function.
  return dataset.n_cols;
}

} // namespace svm
} // namespace mlpack


#endif // MLPACK_METHODS_SPARSE_SVM_SPARSE_SVM_FUNCTION_IMPL_HPP
