/**
 * @file sparse_svm.hpp
 * @author Ayush Chamoli
 *
 * An implementation of Sparse SVM.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_SPARSE_SVM_SPARSE_SVM_HPP
#define MLPACK_METHODS_SPARSE_SVM_SPARSE_SVM_HPP

#include <mlpack/prereqs.hpp>
#include <ensmallen.hpp>

#include "sparse_svm_function.hpp"

namespace mlpack {
namespace svm {

/**
 * The SparseSVM class implements an L2-regularized Support Vector
 * model, and supports training with multiple optimizers and classification.
 * The class supports different observation types via the MatType template
 * parameter; for instance, support vector classification can be performed
 * on sparse datasets by specifying arma::sp_mat as the MatType parameter.
 * More technical details about the model can be found on the following webpage:
 *
 * http://www.jmlr.org/papers/volume3/bi03a/bi03a.pdf
 *
 * @tparam MatType Type of data matrix.
 */
template <typename MatType>
class SparseSVM
{
 public:
  /**
   * Construct the SparseSVM class with the provided data and labels.
   * This will train the model.
   *
   * @tparam OptimizerType Desired differentiable separable optimizer
   * @param data Input training features. Each column associate with one sample
   * @param labels Labels associated with the feature data.
   * @param numClasses Number of classes for classification.
   * @param lambda L2-regularization constant.
   * @param optimizer Desired optimizer.
   */
  template <typename OptimizerType = ens::ParallelSGD<>>
  SparseSVM(const MatType& data,
            const arma::Row<size_t>& labels,
            const size_t numClasses = 2,
            const double lambda = 0.0001,
            OptimizerType optimizer = OptimizerType());

  /**
   * Classify the given points, returning the predicted labels for each point.
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
   * Train the Sparse SVM with the given training data.
   *
   * @tparam OptimizerType Desired optimizer
   * @param data Input training features. Each column associate with one sample
   * @param labels Labels associated with the feature data.
   * @param numClasses Number of classes for classification.
   * @param lambda L2-regularization constant.
   * @param optimizer Desired optimizer.
   * @return Objective value of the final point.
   */
  template <typename OptimizerType = ens::ParallelSGD<>>
  double Train(const MatType& data,
               const arma::Row<size_t>& labels,
               const size_t numClasses = 2,
               const double lambda = 0.0001,
               OptimizerType optimizer = OptimizerType());


  //! Sets the number of classes.
  size_t& NumClasses() { return numClasses; }
  //! Gets the number of classes.
  size_t NumClasses() const { return numClasses; }

  //! Sets the regularization parameter.
  double& Lambda() { return lambda; }
  //! Gets the regularization parameter.
  double Lambda() const { return lambda; }

  //! Set the model parameters.
  arma::mat& Parameters() { return parameters; }
  //! Get the model parameters.
  const arma::mat& Parameters() const { return parameters; }

  //! Gets the features size of the training data
  size_t FeatureSize() const { return parameters.n_cols; }

  /**
   * Serialize the SparseSVM model.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & BOOST_SERIALIZATION_NVP(parameters);
    ar & BOOST_SERIALIZATION_NVP(numClasses);
    ar & BOOST_SERIALIZATION_NVP(lambda);
  }

 private:
  //! Parameters after optimization.
  arma::mat parameters;
  //! Number of classes.
  size_t numClasses;
  //! L2-Regularization constant.
  double lambda;
};

} // namespace svm
} // namespace mlpack

// Include implementation.
#include "sparse_svm_impl.hpp"

#endif
