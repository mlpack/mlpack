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
 * The KernelSVM class.
 *
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
   * @param C standard svm regularization parameter.
   * @param fitIntercept true when not using linear kernel.
   * @param max_iter maximum number of iteration for training.
   * @param tol tolerance value.
   * @param kernel kernel type of kernel used with svm.
   */
  KernelSVM(const MatType& data,
            const arma::Row<size_t>& labels,
            const double C = 1.0,
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
   * @param C standard svm regularization parameter.
   * @param fitIntercept true when not using linear kernel.
   * @param kernel kernel type of kernel used with svm.
   */
  KernelSVM(const size_t inputSize,
            const double C = 1.0,
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
   * @param numClasses Number of classes for classification.
   * @param optimizer Desired optimizer.
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
  }

 private:
  size_t inputSize;
  //! Parameters after optimization.
  arma::mat parameters;
  //! Locally saved C parameter.
  double C;
  //! Intercept term flag.
  bool fitIntercept;
  //! Locally saved b parameter.
  double b;
  //! Locally saved alpha values.
  arma::vec alpha;
  //! Locally saved KernelType values.
  const KernelType& kernel;
  //! Locally saved choosen input labels.
  arma::rowvec y;
  //! Locally saved choosen input data.
  arma::mat x;
};

} // namespace svm
} // namespace mlpack

// Include implementation.
#include "kernel_svm_impl.hpp"

#endif // MLPACK_METHODS_KERNEL_SVM_KERNEL_SVM_HPP
