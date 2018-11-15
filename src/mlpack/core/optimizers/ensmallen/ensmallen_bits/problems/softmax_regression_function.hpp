/**
 * @file softmax_regression_function.hpp
 * @author Siddharth Agrawal
 *
 * The function to be optimized for softmax regression. Any ensmallen optimizer
 * can be used.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_SOFTMAX_REGRESSION_FUNCTION_HPP
#define ENSMALLEN_PROBLEMS_SOFTMAX_REGRESSION_FUNCTION_HPP

namespace ens {
namespace test {

class SoftmaxRegressionFunction
{
 public:
  /**
   * Construct the Softmax Regression objective function with the given
   * parameters.
   *
   * @param data Input training data, each column associate with one sample
   * @param labels Labels associated with the feature data.
   * @param inputSize Size of the input feature vector.
   * @param numClasses Number of classes for classification.
   * @param lambda L2-regularization constant.
   * @param fitIntercept Intercept term flag.
   */
  SoftmaxRegressionFunction(const arma::mat& data,
                            const arma::Row<size_t>& labels,
                            const size_t numClasses,
                            const double lambda = 0.0001,
                            const bool fitIntercept = false);

  //! Initializes the parameters of the model to suitable values.
  const arma::mat InitializeWeights();

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
  const arma::mat InitializeWeights(const size_t featureSize,
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
  void InitializeWeights(arma::mat &weights,
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
                            arma::sp_mat& groundTruth);

  /**
   * Evaluate the probabilities matrix with the passed parameters.
   * probabilities(i, j) =
   *     exp(\theta_i * data_j) / sum_k(exp(\theta_k * data_j)).
   * It represents the probability of data_j belongs to class i.
   *
   * @param parameters Current values of the model parameters.
   * @param probabilities Pointer to arma::mat which stores the probabilities.
   * @param start Index of point to start at.
   * @param batchSize Number of points to calculate probabilities for.
   */
  void GetProbabilitiesMatrix(const arma::mat& parameters,
                              arma::mat& probabilities,
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
  double Evaluate(const arma::mat& parameters) const;

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
  double Evaluate(const arma::mat& parameters,
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
  void Gradient(const arma::mat& parameters, arma::mat& gradient) const;

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
  void Gradient(const arma::mat& parameters,
                const size_t start,
                arma::mat& gradient,
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
  void PartialGradient(const arma::mat& parameters,
                       size_t j,
                       arma::sp_mat& gradient) const;

  //! Return the initial point for the optimization.
  const arma::mat& GetInitialPoint() const { return initialPoint; }

  //! Gets the number of classes.
  size_t NumClasses() const { return numClasses; }

  //! Gets the features size of the training data.
  size_t NumFeatures() const
  {
    return initialPoint.n_cols;
  }

  //! Sets the regularization parameter.
  double& Lambda() { return lambda; }
  //! Gets the regularization parameter.
  double Lambda() const { return lambda; }

  //! Gets the intercept flag.
  bool FitIntercept() const { return fitIntercept; }

 private:
  //! Training data matrix.  This is an alias until the data is shuffled.
  arma::mat data;
  //! Label matrix for the provided data.
  arma::sp_mat groundTruth;
  //! Initial parameter point.
  arma::mat initialPoint;
  //! Number of classes.
  size_t numClasses;
  //! L2-regularization constant.
  double lambda;
  //! Intercept term flag.
  bool fitIntercept;
};

} // namespace test
} // namespace ens

// Include implementation.
#include "softmax_regression_function_impl.hpp"

#endif // ENSMALLEN_PROBLEMS_SOFTMAX_REGRESSION_FUNCTION_HPP
