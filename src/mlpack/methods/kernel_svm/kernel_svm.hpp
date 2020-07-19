/**
 * @file methods/kernel_svm/kernel_svm.hpp
 * @author Himanshu Pathak
 *
 * An implementation of Kernel SVM.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_KERNEL_SVM_KERNEL_SVM_HPP
#define MLPACK_METHODS_KERNEL_SVM_KERNEL_SVM_HPP

#include <mlpack/prereqs.hpp>
#include <ensmallen.hpp>
#include <string>


namespace mlpack {
namespace svm {

/**
 * The KernelSVM class implements an smo algorithm for support vector machine
 * model, and supports training with multiple non-linear and linear kenel.
 * The class supports different observation types via the MatType template
 * parameter; for instance, support vector classification can be performed
 * on sparse datasets by specifying arma::sp_mat as the MatType parameter.
 *
 *

 * @code
 * @article{Microsoft Research,
 *   author    = {John C. Platt},
 *   title     = {Sequential Minimal Optimization:A Fast 
                  Algorithm for Training Support Vector Machines},
 *   journal   = {Technical Report MSR-TR-98-14},
 *   year      = {1998},
 *   url       = {https://www.microsoft.com/en-us/research
                  /wp-content/uploads/2016/02/tr-98-14.pdf},
 * }
 * @endcode
 *
 *
 * An example on how to use the interface is shown below:
 *
 * @code
 * arma::mat train_data; // Training data matrix.
 * arma::Row<size_t> labels; // Labels associated with the data.
 * const size_t inputSize = 1000; // Size of input feature vector.
 *
 * // Train the model using default options.
 * KernelSVM<> svm(train_data, C, kernel_flag, max_iter, tol,
 *     kernel::Gaussian());
 *
 * arma::mat test_data;
 * arma::Row<size_t> predictions;
 * lsvm.Classify(test_data, predictions);
 * @endcode
 *
 * @tparam MatType Type of data matrix.
 * @tparam KernelType Type of kernel used with svm.
 */
template <typename MatType = arma::mat,
          typename KernelType = kernel::LinearKernel>
class KernelSVM
{
 public:
  /**
   * Construct the Kernel SVM class with the provided data and labels.
   *
   * @param data Input training features. Each column associate with one sample
   * @param labels Labels associated with the feature data.
   * @param regularization standard svm regularization parameter.
   * @param fitIntercept true when not using linear kernel.
   * @param max_iter maximum number of iteration for training.
   * @param tol tolerance value.
   * @param kernel kernel type of kernel used with svm.
   */
  KernelSVM(const MatType& data,
            const arma::Row<size_t>& labels,
            const double regularization = 1.0,
            const bool fitIntercept = false,
            const size_t max_iter = 10,
            const double tol = 1e-3,
            const KernelType& kernel = KernelType());

  /**
   * Initialize the Kernel SVM without performing training.  Default  Be sure 
   * to use Train() before calling
   * Classify() or ComputeAccuracy(), otherwise the results may be meaningless.
   *
   * @param inputSize Size of the input feature vector.
   * @param regularization standard svm regularization parameter.
   * @param fitIntercept true when not using linear kernel.
   * @param kernel kernel type of kernel used with svm.
   */
  KernelSVM(const size_t inputSize,
            const double regularization = 1.0,
            const bool fitIntercept = false,
            const KernelType& kernel = KernelType());

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
   * Train the Kernel SVM with the given training data.
   *
   * @tparam OptimizerType Desired optimizer.
   * @param data Input training features. Each column associate with one sample.
   * @param labels Labels associated with the feature data.
   * @param max_iter maximum number of iteration for training.
   * @param tol tolerance value.
   * @return Objective value of the final point.
   */
  double Train(const MatType& data,
               const arma::Row<size_t>& labels,
               const size_t max_iter = 5,
               const double tol = 1e-3);

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
   * Serialize the KernelSVM model.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & BOOST_SERIALIZATION_NVP(parameters);
    ar & BOOST_SERIALIZATION_NVP(fitIntercept);
    ar & BOOST_SERIALIZATION_NVP(trainingData);
    ar & BOOST_SERIALIZATION_NVP(trainLabels);
    ar & BOOST_SERIALIZATION_NVP(alpha);
    ar & BOOST_SERIALIZATION_NVP(intercept);
  }

 private:
  size_t inputSize;
  //! Parameters after optimization.
  arma::mat parameters;
  //! Locally saved standard svm regularization parameter.
  double regularization;
  //! Intercept term flag.
  bool fitIntercept;
  //! Locally saved interce value of kernel.
  double intercept;
  //! Locally saved alpha values.
  arma::vec alpha;
  //! Locally saved KernelType values.
  KernelType kernel;
  //! Locally saved labels of classes to give prediction for non-linear kernel.
  arma::rowvec trainLabels;
  //! Locally saved input data of classes for non-linear kernel.
  arma::mat trainingData;
};

} // namespace svm
} // namespace mlpack

// Include implementation.
#include "kernel_svm_impl.hpp"

#endif // MLPACK_METHODS_KERNEL_SVM_KERNEL_SVM_HPP
