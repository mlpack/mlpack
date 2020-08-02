/**
 * @file methods/kernel_svm/kernel_svm_function.hpp
 * @author Himanshu Pathak
 *
 * An implementation of Kernel SVM.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_KERNEL_SVM_KERNEL_FUNCTION_SVM_HPP
#define MLPACK_METHODS_KERNEL_SVM_KERNEL_FUNCTION_SVM_HPP

#include <mlpack/prereqs.hpp>
#include <ensmallen.hpp>
#include <string>

namespace mlpack {
namespace svm {

/**
 * The KernelSVMFunction class implements an smo algorithm for support vector machine
 * model, and supports training with multiple non-linear and linear kenel.
 * The class supports different observation types via the MatType template
 *
 * @tparam MatType Type of data matrix.
 * @tparam KernelType Type of kernel used with svm.
 */
template <typename MatType = arma::mat,
          typename KernelType = kernel::LinearKernel>
class KernelSVMFunction
{
 public:
  /**
   * Construct the Kernel SVM class with the provided data and labels.
   *
   * @param data Input training features. Each column associate with one sample
   * @param labels Labels associated with the feature data.
   * @param regularization standard svm regularization parameter.
   * @param fitIntercept add intercept term or not.
   * @param firstClasslabel label used for first class.
   * @param secondClasslabel label used for second class.
   * @param max_iter maximum number of iteration for training.
   * @param tol tolerance value.
   * @param kernel kernel type of kernel used with svm.
   */
  KernelSVMFunction(const MatType& data,
                    const arma::Row<size_t>& labels,
                    const double regularization = 1.0,
                    const bool fitIntercept = false,
                    const size_t firstClasslabel = 0,
                    const size_t secondClasslabel = 1,
                    const size_t max_iter = 10,
                    const double tol = 1e-3,
                    const KernelType kernel = KernelType());

  /**
   * Initialize the Kernel SVM without performing training.  Default  Be sure 
   * to use Train() before calling
   * Classify() or ComputeAccuracy(), otherwise the results may be meaningless.
   *
   * @param regularization standard svm regularization parameter.
   * @param fitIntercept add intercept term or not.
   * @param firstClasslabel label used for first class.
   * @param secondClasslabel label used for second class.
   * @param kernel kernel type of kernel used with svm.
   */
  KernelSVMFunction(const double regularization = 1.0,
                    const bool fitIntercept = false,
                    const size_t firstClasslabel = 0,
                    const size_t secondClasslabel = 1,
                    const KernelType kernel = KernelType());

  /**
   * Classify the given points, returning class scores for each point.
   *
   * @param data Matrix of data points to be classified.
   */
  arma::rowvec Classify(const MatType& data) const;

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

  /**
   * Serialize the KernelSVMFunction model.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & BOOST_SERIALIZATION_NVP(fitIntercept);
    ar & BOOST_SERIALIZATION_NVP(trainingData);
    ar & BOOST_SERIALIZATION_NVP(trainCoefficients);
    ar & BOOST_SERIALIZATION_NVP(alpha);
    ar & BOOST_SERIALIZATION_NVP(intercept);
  }

 private:
  //! Locally saved label for first class.
  size_t firstClasslabel;
  //! Locally saved label for second class.
  size_t secondClasslabel;
  //! Locally saved standard svm regularization parameter.
  double regularization;
  //! Intercept term flag.
  bool fitIntercept;
  //! Locally saved intercept value of kernel.
  double intercept;
  //! Locally saved alpha values.
  arma::rowvec alpha;
  //! Locally saved KernelType values.
  KernelType kernel;
    //! Locally saved training labels.
  arma::rowvec trainCoefficients;
  //! Locally saved input data of classes for non-linear kernel.
  arma::mat trainingData;
};

} // namespace svm
} // namespace mlpack

// Include implementation.
#include "kernel_svm_function_impl.hpp"

#endif // MLPACK_METHODS_KERNEL_SVM_KERNEL_SVM_HPP
