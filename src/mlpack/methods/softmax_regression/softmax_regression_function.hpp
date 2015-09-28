/**
 * @file softmax_regression_function.hpp
 * @author Siddharth Agrawal
 *
 * The function to be optimized for softmax regression. Any mlpack optimizer
 * can be used.
 */
#ifndef __MLPACK_METHODS_SOFTMAX_REGRESSION_SOFTMAX_REGRESSION_FUNCTION_HPP
#define __MLPACK_METHODS_SOFTMAX_REGRESSION_SOFTMAX_REGRESSION_FUNCTION_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace regression {

class SoftmaxRegressionFunction
{
 public:
  /**
   * Construct the Softmax Regression objective function with the given
   * parameters.
   *
   * @param data Input training features.
   * @param labels Labels associated with the feature data.
   * @param inputSize Size of the input feature vector.
   * @param numClasses Number of classes for classification.
   * @param lambda L2-regularization constant.
   * @param fitIntercept Intercept term flag.
   */
  SoftmaxRegressionFunction(const arma::mat& data,
                            const arma::vec& labels,
                            const size_t inputSize,
                            const size_t numClasses,
                            const double lambda = 0.0001,
                            const bool fitIntercept = false);

  //! Initializes the parameters of the model to suitable values.
  const arma::mat InitializeWeights();

  /**
   * Constructs the ground truth label matrix with the passed labels.
   *
   * @param labels Labels associated with the training data.
   * @param groundTruth Pointer to arma::mat which stores the computed matrix.
   */
  void GetGroundTruthMatrix(const arma::vec& labels, arma::sp_mat& groundTruth);

  /**
   * Evaluate the probabilities matrix with the passed parameters.
   * probabilities(i, j) =
   *     exp(\theta_i * data_j) / sum_k(exp(\theta_k * data_j)).
   * It represents the probability of data_j belongs to class i.
   *
   * @param parameters Current values of the model parameters.
   * @param probabilities Pointer to arma::mat which stores the probabilities.
   */
  void GetProbabilitiesMatrix(const arma::mat& parameters,
                              arma::mat& probabilities) const;

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
   * Evaluates the gradient values of the objective function given the current
   * set of parameters. The function calculates the probabilities for each class
   * given the parameters, and computes the gradients based on the difference
   * from the ground truth.
   *
   * @param parameters Current values of the model parameters.
   * @param gradient Matrix where gradient values will be stored.
   */
  void Gradient(const arma::mat& parameters, arma::mat& gradient) const;

  //! Return the initial point for the optimization.
  const arma::mat& GetInitialPoint() const {
    return initialPoint;
  }

  //! Sets the size of the input vector.
  size_t& InputSize() {
    return inputSize;
  }
  //! Gets the size of the input vector.
  size_t InputSize() const {
    return inputSize;
  }

  //! Sets the number of classes.
  size_t& NumClasses() {
    return numClasses;
  }
  //! Gets the number of classes.
  size_t NumClasses() const {
    return numClasses;
  }

  //! Sets the regularization parameter.
  double& Lambda() {
    return lambda;
  }
  //! Gets the regularization parameter.
  double Lambda() const {
    return lambda;
  }

  //! Gets the intercept flag.
  bool FitIntercept() const {
    return fitIntercept;
  }

 private:
  //! Training data matrix.
  const arma::mat& data;
  //! Label matrix for the provided data.
  arma::sp_mat groundTruth;
  //! Initial parameter point.
  arma::mat initialPoint;
  //! Size of input feature vector.
  size_t inputSize;
  //! Number of classes.
  size_t numClasses;
  //! L2-regularization constant.
  double lambda;
  //! Intercept term flag.
  bool fitIntercept;
};

}; // namespace regression
}; // namespace mlpack

#endif
