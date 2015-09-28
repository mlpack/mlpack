/**
 * @file softmax_regression.hpp
 * @author Siddharth Agrawal
 *
 * An implementation of softmax regression.
 */
#ifndef __MLPACK_METHODS_SOFTMAX_REGRESSION_SOFTMAX_REGRESSION_HPP
#define __MLPACK_METHODS_SOFTMAX_REGRESSION_SOFTMAX_REGRESSION_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/lbfgs/lbfgs.hpp>

#include "softmax_regression_function.hpp"

namespace mlpack {
namespace regression {

/**
 * Softmax Regression is a classifier which can be used for classification when
 * the data available can take two or more class values. It is a generalization
 * of Logistic Regression (which is used only for binary classification). The
 * model has a different set of parameters for each class, but can be easily
 * converted into a vectorized implementation as has been done in this module.
 * The model can be used for direct classification of feature data or in
 * conjunction with unsupervised learning methods. More technical details about
 * the model can be found on the following webpage:
 *
 * http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression
 *
 * An example on how to use the interface is shown below:
 *
 * @code
 * arma::mat train_data; // Training data matrix.
 * arma::vec labels; // Labels associated with the data.
 * const size_t inputSize = 784; // Size of input feature vector.
 * const size_t numClasses = 10; // Number of classes.
 *
 * // Train the model using default options.
 * SoftmaxRegression<> regressor1(train_data, labels, inputSize, numClasses);
 *
 * const size_t numBasis = 5; // Parameter required for L-BFGS algorithm.
 * const size_t numIterations = 100; // Maximum number of iterations.
 *
 * // Use an instantiated optimizer for the training.
 * SoftmaxRegressionFunction srf(train_data, labels, inputSize, numClasses);
 * L_BFGS<SoftmaxRegressionFunction> optimizer(srf, numBasis, numIterations);
 * SoftmaxRegression<L_BFGS> regressor2(optimizer);
 *
 * arma::mat test_data; // Test data matrix.
 * arma::vec predictions1, predictions2; // Vectors to store predictions in.
 *
 * // Obtain predictions from both the learned models.
 * regressor1.Predict(test_data, predictions1);
 * regressor2.Predict(test_data, predictions2);
 * @endcode
 */

template<
  template<typename> class OptimizerType = mlpack::optimization::L_BFGS
  >
class SoftmaxRegression
{
 public:
  /**
   * Initialize the SoftmaxRegression without performing training.
   * Default value of lambda is 0.0001.
   * Be sure to use Train() before calling Predict() or ComputeAccuracy(),
   * otherwise the results may be meaningless.
   *
   * @param inputSize Size of the input feature vector.
   * @param numClasses Number of classes for classification.
   * @param fitIntercept add intercept term or not.
   */
  SoftmaxRegression(const size_t inputSize,
                    const size_t numClasses,
                    const bool fitIntercept = false);

  /**
   * Construct the SoftmaxRegression class with the provided data and labels.
   * This will train the model.
   *
   * @param fileName name of the files saving the model contents
   * @param name name of the structure to be save
   * @exception If the file cannot be load, the exception will thrown
   */
  SoftmaxRegression(const std::string &fileName,
                    const std::string& name);

  /**
   * Construct the SoftmaxRegression class with the provided data and labels.
   * This will train the model. Optionally, the parameter 'lambda' can be
   * passed, which controls the amount of L2-regularization in the objective
   * function. By default, the model takes a small value.
   *
   * @param data Input training features. Each column associate with one sample
   * @param labels Labels associated with the feature data.
   * @param inputSize Size of the input feature vector.
   * @param numClasses Number of classes for classification.
   * @param lambda L2-regularization constant.
   * @param fitIntercept add intercept term or not.
   */
  SoftmaxRegression(const arma::mat& data,
                    const arma::vec& labels,
                    const size_t inputSize,
                    const size_t numClasses,
                    const double lambda = 0.0001,
                    const bool fitIntercept = false);

  /**
   * Construct the softmax regression model with the given training data. This
   * will train the model. This overload takes an already instantiated optimizer
   * and uses it to train the model. The optimizer should hold an instantiated
   * SoftmaxRegressionFunction object for the function to operate upon. This
   * option should be preferred when the optimizer options are to be changed.
   *
   * @param optimizer Instantiated optimizer with instantiated error function.
   */
  SoftmaxRegression(OptimizerType<SoftmaxRegressionFunction>& optimizer);

  /**
   * Predict the class labels for the provided feature points. The function
   * calculates the probabilities for every class, given a data point. It then
   * chooses the class which has the highest probability among all.
   *
   * @param testData Matrix of data points for which predictions are to be made.
   * @param predictions Vector to store the predictions in.
   */
  void Predict(const arma::mat& testData, arma::vec& predictions);

  /**
   * Computes accuracy of the learned model given the feature data and the
   * labels associated with each data point. Predictions are made using the
   * provided data and are compared with the actual labels.
   *
   * @param testData Matrix of data points using which predictions are made.
   * @param labels Vector of labels associated with the data.
   */
  double ComputeAccuracy(const arma::mat& testData, const arma::vec& labels);

  /**
   * Train the softmax regression model with the given optimizer.
   * The optimizer should hold an instantiated
   * SoftmaxRegressionFunction object for the function to operate upon. This
   * option should be preferred when the optimizer options are to be changed.
   * @param optimizer Instantiated optimizer with instantiated error function.
   * @return Objective value of the final point.
   */
  double Train(OptimizerType<SoftmaxRegressionFunction>& optimizer);

  /**
   * Train the softmax regression with the given training data.
   * @param data Input data with each column as one example.
   * @param labels Labels associated with the feature data.
   * @param numClasses Number of classes for classification.
   * @return Objective value of the final point.
   */
  double Train(const arma::mat &data, const arma::vec& labels,
               const size_t numClasses);

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

  //! Gets the intercept term flag.  We can't change this after training.
  bool FitIntercept() const {
    return fitIntercept;
  }

  //! get the training parameters
  arma::mat& Parameters()
  {
    return parameters;
  }

  //! get the training parameters
  const arma::mat& Parameters() const
  {
    return parameters;
  }

  /**
   * Serialize the SparseAutoencoder
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    using mlpack::data::CreateNVP;

    ar & CreateNVP(parameters, "parameters");
    ar & CreateNVP(inputSize, "inputSize");
    ar & CreateNVP(numClasses, "numClasses");
    ar & CreateNVP(lambda, "lambda");
    ar & CreateNVP(fitIntercept, "fitIntercept");
  }

 private:
  //! Parameters after optimization.
  arma::mat parameters;
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

// Include implementation.
#include "softmax_regression_impl.hpp"

#endif
