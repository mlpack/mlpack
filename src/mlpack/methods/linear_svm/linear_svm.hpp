/**
 * @file methods/linear_svm/linear_svm.hpp
 * @author Ayush Chamoli
 *
 * An implementation of Linear SVM.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_LINEAR_SVM_LINEAR_SVM_HPP
#define MLPACK_METHODS_LINEAR_SVM_LINEAR_SVM_HPP

#include <mlpack/core.hpp>
#include <ensmallen.hpp>

#include "linear_svm_function.hpp"

namespace mlpack {

/**
 * The LinearSVM class implements an L2-regularized support vector machine
 * model, and supports training with multiple optimizers and classification.
 * The class supports different observation types via the MatType template
 * parameter; for instance, support vector classification can be performed
 * on sparse datasets by specifying arma::sp_mat as the MatType parameter.
 *
 * Linear SVM can be used for general classification tasks which will work
 * on multiclass classification. More technical details about
 * the model can be found from the following:
 *
 * @code
 * @inproceedings{weston1999support,
 * title        = {Support vector machines for multi-class pattern
 *                 recognition.},
 * author       = {Weston, Jason and Watkins, Chris},
 * booktitle    = {Proceedings of the 7th European Symposium on Artifical Neural
 *                 Networks (ESANN '99)},
 * volume       = {99},
 * pages        = {219--224},
 * year         = {1999}
 * }
 * @endcode
 *
 * @code
 * @article{cortes1995support,
 * title        = {Support-vector networks},
 * author       = {Cortes, Corinna and Vapnik, Vladimir},
 * journal      = {Machine Learning},
 * volume       = {20},
 * number       = {3},
 * pages        = {273--297},
 * year         = {1995},
 * publisher    = {Springer}
 * }
 * @endcode
 *
 * An example on how to use the interface is shown below:
 *
 * @code
 * arma::mat train_data; // Training data matrix.
 * arma::Row<size_t> labels; // Labels associated with the data.
 * const size_t inputSize = 1000; // Size of input feature vector.
 * const size_t numClasses = 5; // Number of classes.
 *
 * // Train the model using default options.
 * LinearSVM<> lsvm(train_data, labels, inputSize, numClasses, lambda,
 *     delta, L_BFGS());
 *
 * arma::mat test_data;
 * arma::Row<size_t> predictions;
 * lsvm.Classify(test_data, predictions);
 * @endcode
 *
 * @tparam MatType Type of data matrix.
 */
template <typename MatType = arma::mat>
class LinearSVM
{
 public:
  /**
   * Construct the LinearSVM class with the provided data and labels.
   * This will train the model. Optionally, the parameter 'lambda' can be
   * passed, which controls the amount of L2-regularization in the objective
   * function. By default, the model takes a small value.
   *
   * @tparam OptimizerType Desired differentiable separable optimizer
   * @tparam CallbackTypes Types of callback functions.
   * @param data Input training features. Each column associate with one sample
   * @param labels Labels associated with the feature data.
   * @param numClasses Number of classes for classification.
   * @param lambda L2-regularization constant.
   * @param delta Margin of difference between correct class and other classes.
   * @param fitIntercept add intercept term or not.
   * @param optimizer Desired optimizer.
   * @param callbacks Callback functions.
   *      See https://www.ensmallen.org/docs.html#callback-documentation.
   */
  template <typename OptimizerType, typename... CallbackTypes>
  LinearSVM(const MatType& data,
            const arma::Row<size_t>& labels,
            const size_t numClasses,
            const double lambda,
            const double delta,
            const bool fitIntercept,
            OptimizerType optimizer,
            CallbackTypes&&... callbacks);

  /**
   * Construct the LinearSVM class with the provided data and labels.
   * This will train the model. Optionally, the parameter 'lambda' can be
   * passed, which controls the amount of L2-regularization in the objective
   * function. By default, the model takes a small value.
   *
   * @tparam OptimizerType Desired differentiable separable optimizer
   * @param data Input training features. Each column associate with one sample
   * @param labels Labels associated with the feature data.
   * @param numClasses Number of classes for classification.
   * @param lambda L2-regularization constant.
   * @param delta Margin of difference between correct class and other classes.
   * @param fitIntercept add intercept term or not.
   * @param optimizer Desired optimizer.
   */
  template <typename OptimizerType = ens::L_BFGS>
  LinearSVM(const MatType& data,
            const arma::Row<size_t>& labels,
            const size_t numClasses = 2,
            const double lambda = 0.0001,
            const double delta = 1.0,
            const bool fitIntercept = false,
            OptimizerType optimizer = OptimizerType());

  /**
   * Initialize the Linear SVM without performing training.  Default
   * value of lambda is 0.0001.  Be sure to use Train() before calling
   * Classify() or ComputeAccuracy(), otherwise the results may be meaningless.
   *
   * @param inputSize Size of the input feature vector.
   * @param numClasses Number of classes for classification.
   * @param lambda L2-regularization constant.
   * @param delta Margin of difference between correct class and other classes.
   * @param fitIntercept add intercept term or not.
   */
  LinearSVM(const size_t inputSize,
            const size_t numClasses = 0,
            const double lambda = 0.0001,
            const double delta = 1.0,
            const bool fitIntercept = false);
  /**
   * Initialize the Linear SVM without performing training.  Default
   * value of lambda is 0.0001.  Be sure to use Train() before calling
   * Classify() or ComputeAccuracy(), otherwise the results may be meaningless.
   *
   * @param numClasses Number of classes for classification.
   * @param lambda L2-regularization constant.
   * @param delta Margin of difference between correct class and other classes.
   * @param fitIntercept add intercept term or not.
   */
  LinearSVM(const size_t numClasses = 0,
            const double lambda = 0.0001,
            const double delta = 1.0,
            const bool fitIntercept = false);

  /**
   * Classify the given points, returning the predicted labels for each point.
   * The function calculates the probabilities for every class, given a data
   * point. It then chooses the class which has the highest probability among
   * all.
   *
   * @param data Set of points to classify.
   * @param labels Predicted labels for each point.
   */
  void Classify(const MatType& data,
                arma::Row<size_t>& labels) const;

  /**
   * Classify the given points, returning class scores and predicted
   * class label for each point.
   * The function calculates the scores for every class, given a data
   * point. It then chooses the class which has the highest probability among
   * all.
   *
   * @param data Matrix of data points to be classified.
   * @param labels Predicted labels for each point.
   * @param scores Class probabilities for each point.
   */
  void Classify(const MatType& data,
                arma::Row<size_t>& labels,
                arma::mat& scores) const;

  /**
   * Classify the given points, returning class scores for each point.
   *
   * @param data Matrix of data points to be classified.
   * @param scores Class scores for each point.
   */
  void Classify(const MatType& data,
                arma::mat& scores) const;

  /**
   * Classify the given point. The predicted class label is returned.
   * The function calculates the scores for every class, given the point.
   * It then chooses the class which has the highest probability among all.
   *
   * @param point Point to be classified.
   * @return Predicted class label of the point.
   */
  template<typename VecType>
  size_t Classify(const VecType& point) const;

  /**
   * Computes accuracy of the learned model given the feature data and the
   * labels associated with each data point. Predictions are made using the
   * provided data and are compared with the actual labels.
   *
   * @param testData Matrix of data points using which predictions are made.
   * @param testLabels Vector of labels associated with the data.
   * @return Accuracy of the model.
   */
  double ComputeAccuracy(const MatType& testData,
                         const arma::Row<size_t>& testLabels) const;

  /**
   * Train the Linear SVM with the given training data.
   *
   * @tparam OptimizerType Desired optimizer.
   * @tparam CallbackTypes Types of Callback Functions.
   * @param data Input training features. Each column associate with one sample.
   * @param labels Labels associated with the feature data.
   * @param numClasses Number of classes for classification.
   * @param optimizer Desired optimizer.
   * @param callbacks Callback Functions.
   *      See https://www.ensmallen.org/docs.html#callback-documentation.
   * @return Objective value of the final point.
   */
  template <typename OptimizerType, typename... CallbackTypes>
  double Train(const MatType& data,
               const arma::Row<size_t>& labels,
               const size_t numClasses,
               OptimizerType optimizer,
               CallbackTypes&&... callbacks);

  /**
   * Train the Linear SVM with the given training data.
   *
   * @tparam OptimizerType Desired optimizer.
   * @param data Input training features. Each column associate with one sample.
   * @param labels Labels associated with the feature data.
   * @param numClasses Number of classes for classification.
   * @param optimizer Desired optimizer.
   * @return Objective value of the final point.
   */
  template <typename OptimizerType = ens::L_BFGS>
  double Train(const MatType& data,
               const arma::Row<size_t>& labels,
               const size_t numClasses = 2,
               OptimizerType optimizer = OptimizerType());


  //! Sets the number of classes.
  size_t& NumClasses() { return numClasses; }
  //! Gets the number of classes.
  size_t NumClasses() const { return numClasses; }

  //! Sets the regularization parameter.
  double& Lambda() { return lambda; }
  //! Gets the regularization parameter.
  double Lambda() const { return lambda; }

  //! Sets the margin between the correct class and all other classes.
  double& Delta() { return delta; }
  //! Gets the margin between the correct class and all other classes.
  double Delta() const { return delta; }

  //! Sets the intercept term flag.
  bool& FitIntercept() { return fitIntercept; }

  //! Set the model parameters.
  arma::mat& Parameters() { return parameters; }
  //! Get the model parameters.
  const arma::mat& Parameters() const { return parameters; }

  //! Gets the features size of the training data
  size_t FeatureSize() const
  { return fitIntercept ? parameters.n_rows - 1 :
           parameters.n_rows; }

  /**
   * Serialize the LinearSVM model.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(parameters));
    ar(CEREAL_NVP(numClasses));
    ar(CEREAL_NVP(lambda));
    ar(CEREAL_NVP(fitIntercept));
  }

 private:
  //! Parameters after optimization.
  arma::mat parameters;
  //! Number of classes.
  size_t numClasses;
  //! L2-Regularization constant.
  double lambda;
  //! The margin between the correct class and all other classes.
  double delta;
  //! Intercept term flag.
  bool fitIntercept;
};

} // namespace mlpack

// Include implementation.
#include "linear_svm_impl.hpp"

#endif // MLPACK_METHODS_LINEAR_SVM_LINEAR_SVM_HPP
