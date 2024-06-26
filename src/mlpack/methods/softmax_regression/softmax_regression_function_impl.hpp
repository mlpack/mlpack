/**
 * @file methods/softmax_regression/softmax_regression_function_impl.hpp
 * @author Siddharth Agrawal
 *
 * Implementation of function to be optimized for softmax regression.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_SOFTMAX_REGRESSION_SOFTMAX_REGRESSION_FUNCTION_IMPL_HPP
#define MLPACK_METHODS_SOFTMAX_REGRESSION_SOFTMAX_REGRESSION_FUNCTION_IMPL_HPP

#include "softmax_regression_function.hpp"

namespace mlpack {

template<typename MatType>
inline SoftmaxRegressionFunction<MatType>::SoftmaxRegressionFunction(
    const MatType& dataIn,
    const arma::Row<size_t>& labels,
    const size_t numClasses,
    const double lambda,
    const bool fitIntercept) :
    numClasses(numClasses),
    lambda(lambda),
    fitIntercept(fitIntercept)
{
  MakeAlias(data, dataIn, dataIn.n_rows, dataIn.n_cols, 0, false);

  // Initialize the parameters to suitable values.
  initialPoint = InitializeWeights();

  // Calculate the label matrix.
  GetGroundTruthMatrix(labels, groundTruth);
}

/**
 * Shuffle the data.
 */
template<typename MatType>
inline void SoftmaxRegressionFunction<MatType>::Shuffle()
{
  // Determine new ordering.
  arma::uvec ordering = arma::shuffle(arma::linspace<arma::uvec>(0,
      data.n_cols - 1, data.n_cols));

  // Re-sort data.
  MatType newData = data.cols(ordering);
  ClearAlias(data);
  data = std::move(newData);

  // Assemble data for batch constructor.  We need reverse orderings though...
  arma::uvec reverseOrdering(ordering.n_elem);
  for (size_t i = 0; i < ordering.n_elem; ++i)
    reverseOrdering[ordering[i]] = i;

  arma::umat newLocations(2, groundTruth.n_nonzero);
  arma::Col<ElemType> values(groundTruth.n_nonzero);
  typename SpMatType::const_iterator it = groundTruth.begin();
  size_t loc = 0;
  while (it != groundTruth.end())
  {
    newLocations(0, loc) = it.row();
    newLocations(1, loc) = reverseOrdering(it.col());
    values(loc) = (*it);

    ++it;
    ++loc;
  }

  groundTruth = SpMatType(newLocations, values, groundTruth.n_rows,
      groundTruth.n_cols);
}

/**
 * Initializes parameter weights to random values taken from a scaled standard
 * normal distribution. The weights cannot be initialized to zero, as that will
 * lead to each class output being the same.
 */
template<typename MatType>
inline const typename SoftmaxRegressionFunction<MatType>::DenseMatType
SoftmaxRegressionFunction<MatType>::InitializeWeights()
{
  return InitializeWeights(data.n_rows, numClasses, fitIntercept);
}

template<typename MatType>
inline const typename SoftmaxRegressionFunction<MatType>::DenseMatType
SoftmaxRegressionFunction<MatType>::InitializeWeights(
    const size_t featureSize,
    const size_t numClasses,
    const bool fitIntercept)
{
  DenseMatType parameters;
  InitializeWeights(parameters, featureSize, numClasses, fitIntercept);
  return parameters;
}

template<typename MatType>
inline void SoftmaxRegressionFunction<MatType>::InitializeWeights(
    typename SoftmaxRegressionFunction<MatType>::DenseMatType& weights,
    const size_t featureSize,
    const size_t numClasses,
    const bool fitIntercept)
{
  // Initialize values to 0.005 * r. 'r' is a matrix of random values taken from
  // a Gaussian distribution with mean zero and variance one.
  // If the fitIntercept flag is true, parameters.col(0) is the intercept.
  if (fitIntercept)
    weights.randn(numClasses, featureSize + 1);
  else
    weights.randn(numClasses, featureSize);
  weights *= 0.005;
}

/**
 * This is equivalent to applying the indicator function to the training
 * labels. The output is in the form of a matrix, which leads to simpler
 * calculations in the Evaluate() and Gradient() methods.
 */
template<typename MatType>
inline void SoftmaxRegressionFunction<MatType>::GetGroundTruthMatrix(
    const arma::Row<size_t>& labels,
    typename SoftmaxRegressionFunction<MatType>::SpMatType& groundTruth)
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
  arma::Col<ElemType> values;
  values.ones(labels.n_elem);

  // Calculate the matrix.
  groundTruth = SpMatType(rowPointers, colPointers, values, numClasses,
      labels.n_elem);
}

/**
 * Evaluate the probabilities matrix. If fitIntercept flag is true,
 * it should consider the parameters.cols(0) intercept term.
 */
template<typename MatType>
inline void SoftmaxRegressionFunction<MatType>::GetProbabilitiesMatrix(
    const typename SoftmaxRegressionFunction<MatType>::DenseMatType& parameters,
    typename SoftmaxRegressionFunction<MatType>::DenseMatType& probabilities,
    const size_t start,
    const size_t batchSize) const
{
  DenseMatType hypothesis;

  if (fitIntercept)
  {
    // In order to add the intercept term, we should compute following matrix:
    //     [1; data] = join_cols(ones(1, data.n_cols), data)
    //     hypothesis = exp(parameters * [1; data]).
    //
    // Since the cost of join may be high due to the copy of original data,
    // split the hypothesis computation to two components.
    hypothesis = exp(
        repmat(parameters.col(0), 1, batchSize) +
        parameters.cols(1, parameters.n_cols - 1) *
        data.cols(start, start + batchSize - 1));
  }
  else
  {
    hypothesis = exp(parameters * data.cols(start, start + batchSize - 1));
  }

  probabilities = hypothesis / repmat(sum(hypothesis, 0), numClasses, 1);
}

/**
 * Evaluates the objective function given the parameters.
 */
template<typename MatType>
inline
typename SoftmaxRegressionFunction<MatType>::ElemType
SoftmaxRegressionFunction<MatType>::Evaluate(
    const typename SoftmaxRegressionFunction<MatType>::DenseMatType& parameters)
    const
{
  // The objective function is the negative log likelihood of the model
  // calculated over all the training examples. Mathematically it is as follows:
  // log likelihood = sum(1{y_i = j} * log(probability(j))) / m
  // The sum is over all 'i's and 'j's, where 'i' points to a training example
  // and 'j' points to a particular class. 1{x} is an indicator function whose
  // value is 1 only when 'x' is satisfied, otherwise it is 0.
  // 'm' is the number of training examples.
  // The cost also takes into account the regularization to control the
  // parameter weights.

  // Calculate the class probabilities for each training example. The
  // probabilities for each of the classes are given by:
  // p_j = exp(theta_j' * x_i) / sum(exp(theta_k' * x_i))
  // The sum is calculated over all the classes.
  // x_i is the input vector for a particular training example.
  // theta_j is the parameter vector associated with a particular class.
  DenseMatType probabilities;
  GetProbabilitiesMatrix(parameters, probabilities, 0, data.n_cols);

  // Calculate the log likelihood and regularization terms.
  ElemType logLikelihood, weightDecay, cost;

  logLikelihood = arma::accu(groundTruth % log(probabilities)) /
                  data.n_cols;
  weightDecay = ((typename MatType::elem_type) 0.5) * lambda *
      arma::accu(parameters % parameters);

  // The cost is the sum of the negative log likelihood and the regularization
  // terms.
  cost = -logLikelihood + weightDecay;

  return cost;
}

/**
 * Evaluate the objective function for the given points given the parameters.
 */
template<typename MatType>
inline
typename SoftmaxRegressionFunction<MatType>::ElemType
SoftmaxRegressionFunction<MatType>::Evaluate(
    const typename SoftmaxRegressionFunction<MatType>::DenseMatType& parameters,
    const size_t start,
    const size_t batchSize) const
{
  MatType probabilities;
  GetProbabilitiesMatrix(parameters, probabilities, start, batchSize);

  // Calculate the log likelihood and regularization terms.
  ElemType logLikelihood, weightDecay;

  logLikelihood = arma::accu(groundTruth.cols(start, start + batchSize - 1) %
      log(probabilities)) / batchSize;
  weightDecay = ((typename MatType::elem_type) 0.5) * lambda *
      norm(vectorise(parameters), 2);

  return -logLikelihood + weightDecay;
}

/**
 * Calculates and stores the gradient values given a set of parameters.
 */
template<typename MatType>
template<typename GradType>
inline void SoftmaxRegressionFunction<MatType>::Gradient(
    const typename SoftmaxRegressionFunction<MatType>::DenseMatType& parameters,
    GradType& gradient) const
{
  // Calculate the class probabilities for each training example. The
  // probabilities for each of the classes are given by:
  // p_j = exp(theta_j' * x_i) / sum(exp(theta_k' * x_i))
  // The sum is calculated over all the classes.
  // x_i is the input vector for a particular training example.
  // theta_j is the parameter vector associated with a particular class.
  DenseMatType probabilities;
  GetProbabilitiesMatrix(parameters, probabilities, 0, data.n_cols);

  // Calculate the parameter gradients.
  gradient.set_size(parameters.n_rows, parameters.n_cols);
  if (fitIntercept)
  {
    // Treating the intercept term parameters.col(0) seperately to avoid
    // the cost of building matrix [1; data].
    DenseMatType inner = probabilities - groundTruth;
    gradient.col(0) =
        inner * ones<DenseMatType>(data.n_cols, 1) / data.n_cols +
        lambda * parameters.col(0);
    gradient.cols(1, parameters.n_cols - 1) =
        inner * data.t() / data.n_cols +
        lambda * parameters.cols(1, parameters.n_cols - 1);
  }
  else
  {
    gradient = (probabilities - groundTruth) * data.t() / data.n_cols +
               lambda * parameters;
  }
}

template<typename MatType>
template<typename GradType>
inline void SoftmaxRegressionFunction<MatType>::Gradient(
    const typename SoftmaxRegressionFunction<MatType>::DenseMatType& parameters,
    const size_t start,
    GradType& gradient,
    const size_t batchSize) const
{
  DenseMatType probabilities;
  GetProbabilitiesMatrix(parameters, probabilities, start, batchSize);

  // Calculate the parameter gradients.
  gradient.set_size(parameters.n_rows, parameters.n_cols);
  if (fitIntercept)
  {
    DenseMatType inner = probabilities - groundTruth.cols(start, start +
        batchSize - 1);
    gradient.col(0) =
        inner * ones<DenseMatType>(batchSize, 1) / batchSize +
        lambda * parameters.col(0);
    gradient.cols(1, parameters.n_cols - 1) =
        inner * data.cols(start, start + batchSize - 1).t() / batchSize +
        lambda * parameters.cols(1, parameters.n_cols - 1);
  }
  else
  {
    gradient = (probabilities - groundTruth.cols(start, start + batchSize - 1))
        * data.cols(start, start + batchSize - 1).t() / batchSize
        + lambda * parameters;
  }
}

template<typename MatType>
template<typename GradType>
inline void SoftmaxRegressionFunction<MatType>::PartialGradient(
    const typename SoftmaxRegressionFunction<MatType>::DenseMatType& parameters,
    const size_t j,
    GradType& gradient) const
{
  gradient.zeros(arma::size(parameters));

  DenseMatType probabilities;
  GetProbabilitiesMatrix(parameters, probabilities, 0, data.n_cols);

  // Calculate the required part of the gradient.
  DenseMatType inner = probabilities - groundTruth;
  if (fitIntercept)
  {
    if (j == 0)
    {
      gradient.col(j) =
          inner * ones<DenseMatType>(data.n_cols, 1) / data.n_cols +
          lambda * parameters.col(0);
    }
    else
    {
      gradient.col(j) = inner * data.row(j).t() / data.n_cols + lambda *
          parameters.col(j);
    }
  }
  else
  {
    gradient.col(j) = inner * data.row(j).t() / data.n_cols + lambda *
        parameters.col(j);
  }
}

} // namespace mlpack

#endif
