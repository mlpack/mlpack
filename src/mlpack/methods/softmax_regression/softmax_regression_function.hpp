/**
 * @file methods/softmax_regression/softmax_regression_function.hpp
 * @author Siddharth Agrawal
 *
 * The function to be optimized for softmax regression. Any mlpack optimizer
 * can be used.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_SOFTMAX_REGRESSION_SOFTMAX_REGRESSION_FUNCTION_HPP
#define MLPACK_METHODS_SOFTMAX_REGRESSION_SOFTMAX_REGRESSION_FUNCTION_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/math/make_alias.hpp>

namespace mlpack {

template<typename MatType = arma::mat>
class SoftmaxRegressionFunction
{
 public:
  using ElemType = typename MatType::elem_type;
  using DenseMatType = typename GetDenseMatType<MatType>::type;
  using SpMatType = typename GetSparseMatType<MatType>::type;

  /**
   * Construct the Softmax Regression objective function with the given
   * parameters.
   *
   * @param data Input training data, each column associate with one sample
   * @param labels Labels associated with the feature data.
   * @param numClasses Number of classes for classification.
   * @param lambda L2-regularization constant.
   * @param fitIntercept Intercept term flag.
   */
  SoftmaxRegressionFunction(const MatType& data,
                            const arma::Row<size_t>& labels,
                            const size_t numClasses,
                            const double lambda = 0.0001,
                            const bool fitIntercept = false);

  //! Initializes the parameters of the model to suitable values.
  const DenseMatType InitializeWeights();

  /**
   * Shuffle the dataset.
   */
  void Shuffle();

  /**
   * Initialize Softmax Regression weights (trainable parameters) with the given
   * parameters.
   *
   * @param featureSize The number of features in the training set.
   * @param numClasses Number of classes for classification.
   * @param fitIntercept If true, an intercept is fitted.
   * @return Initialized model weights.
   */
  static const DenseMatType InitializeWeights(const size_t featureSize,
                                              const size_t numClasses,
                                              const bool fitIntercept = false);

  /**
   * Initialize Softmax Regression weights (trainable parameters) with the given
   * parameters.
   *
   * @param weights This will be filled with the initialized model weights.
   * @param featureSize The number of features in the training set.
   * @param numClasses Number of classes for classification.
   * @param fitIntercept Intercept term flag.
   */
  static void InitializeWeights(DenseMatType& weights,
                                const size_t featureSize,
                                const size_t numClasses,
                                const bool fitIntercept = false);

  /**
   * Constructs the ground truth label matrix with the passed labels.
   *
   * @param labels Labels associated with the training data.
   * @param groundTruth Pointer to arma::mat which stores the computed matrix.
   */
  void GetGroundTruthMatrix(const arma::Row<size_t>& labels,
                            SpMatType& groundTruth);

  /**
   * Evaluate the probabilities matrix with the passed parameters.
   * probabilities(i, j) =
   *     @f$ exp(\theta_i * data_j) / sum_k(exp(\theta_k * data_j)) @f$.
   * It represents the probability of data_j belongs to class i.
   *
   * @param parameters Current values of the model parameters.
   * @param probabilities Pointer to arma::mat which stores the probabilities.
   * @param start Index of point to start at.
   * @param batchSize Number of points to calculate probabilities for.
   */
  void GetProbabilitiesMatrix(const DenseMatType& parameters,
                              DenseMatType& probabilities,
                              const size_t start,
                              const size_t batchSize) const;

  /**
   * Evaluates the objective function of the softmax regression model using the
   * given parameters. The cost function has terms for the log likelihood error
   * and the regularization cost. The objective function takes a low value when
   * the model generalizes well for the given training data, while having small
   * parameter values.
   *
   * @param parameters Current values of the model parameters.
   */
  ElemType Evaluate(const DenseMatType& parameters) const;

  /**
   * Evaluate the objective function of the softmax regression model for a
   * subset of the data points using the given parameters. The cost function has
   * terms for the log likelihood error and the regularization cost. The
   * objective function takes a low value when the model generalizes well for
   * the given training data, while having small parameter values.
   *
   * @param parameters Current values of the model parameters.
   * @param start First index of the data points to use.
   * @param batchSize Number of data points to evaluate objective for.
   */
  ElemType Evaluate(const DenseMatType& parameters,
                    const size_t start,
                    const size_t batchSize = 1) const;

  /**
   * Evaluates the gradient values of the objective function given the current
   * set of parameters. The function calculates the probabilities for each class
   * given the parameters, and computes the gradients based on the difference
   * from the ground truth.
   *
   * @param parameters Current values of the model parameters.
   * @param gradient Matrix where gradient values will be stored.
   */
  template<typename GradType>
  void Gradient(const DenseMatType& parameters, GradType& gradient) const;

  /**
   * Evaluate the gradient of the objective function given the current set of
   * parameters, on a subset of the data. The function calculates the
   * probabilities for each class given the parameters, and computes the
   * gradients based on the difference from the ground truth.
   *
   * @param parameters Current values of the model parameters.
   * @param start First index of the data points to use.
   * @param gradient Matrix to store gradient into.
   * @param batchSize Number of data points to evaluate gradient for.
   */
  template<typename GradType>
  void Gradient(const DenseMatType& parameters,
                const size_t start,
                GradType& gradient,
                const size_t batchSize = 1) const;

  /**
   * Evaluates the gradient values of the objective function given the current
   * set of parameters for a single feature indexed by j.
   *
   * @param parameters Current values of the model parameters.
   * @param j The index of the feature with respect to which the partial
   *    gradient is to be computed.
   * @param gradient Out param for the gradient value.
   */
  template<typename GradType>
  void PartialGradient(const DenseMatType& parameters,
                       size_t j,
                       GradType& gradient) const;

  //! Return the initial point for the optimization.
  const DenseMatType& GetInitialPoint() const { return initialPoint; }

  //! Gets the number of classes.
  size_t NumClasses() const { return numClasses; }

  //! Gets the features size of the training data.
  size_t NumFeatures() const
  {
    return initialPoint.n_cols;
  }
  /**
   * Return the number of separable functions (the number of predictor points).
   */
  size_t NumFunctions() const { return data.n_cols; }

  //! Sets the regularization parameter.
  double& Lambda() { return lambda; }
  //! Gets the regularization parameter.
  double Lambda() const { return lambda; }

  //! Gets the intercept flag.
  bool FitIntercept() const { return fitIntercept; }

 private:
  //! Training data matrix.  This is an alias until the data is shuffled.
  MatType data;
  //! Label matrix for the provided data.
  SpMatType groundTruth;
  //! Initial parameter point.
  DenseMatType initialPoint;
  //! Number of classes.
  size_t numClasses;
  //! L2-regularization constant.
  double lambda;
  //! Intercept term flag.
  bool fitIntercept;
};

} // namespace mlpack

// Include implementation.
#include "softmax_regression_function_impl.hpp"

#endif
