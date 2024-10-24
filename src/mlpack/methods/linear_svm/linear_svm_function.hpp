/**
 * @file methods/linear_svm/linear_svm_function.hpp
 * @author Shikhar Bhardwaj
 * @author Ayush Chamoli
 *
 * Implementation of the hinge loss function for training a linear SVM with the
 * parallel SGD algorithm.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_LINEAR_SVM_LINEAR_SVM_FUNCTION_HPP
#define MLPACK_METHODS_LINEAR_SVM_LINEAR_SVM_FUNCTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * The hinge loss function for the linear SVM objective function.
 * This is used by various ensmallen optimizers to train the linear
 * SVM model.
 */
template<typename MatType = arma::mat, typename ParametersType = arma::mat>
class LinearSVMFunction
{
 public:
  using ElemType = typename ParametersType::elem_type;
  using DenseMatType = typename GetDenseMatType<ParametersType>::type;
  using SparseMatType = typename GetSparseMatType<ParametersType>::type;
  using DenseColType = typename GetDenseColType<SparseMatType>::type;
  using DenseRowType = typename GetDenseRowType<ParametersType>::type;

  /**
   * Construct the Linear SVM objective function with given parameters.
   *
   * @param dataset Input training data, each column associate with one sample
   * @param labels Labels associated with the feature data.
   * @param numClasses Number of classes for classification.
   * @param lambda L2-regularization constant.
   * @param delta Margin of difference between correct class and other classes.
   * @param fitIntercept Intercept term flag.
   */
  LinearSVMFunction(const MatType& dataset,
                    const arma::Row<size_t>& labels,
                    const size_t numClasses,
                    const double lambda = 0.0001,
                    const double delta = 1.0,
                    const bool fitIntercept = false);

  /**
   * Shuffle the dataset.
   */
  void Shuffle();

  /**
   * Initialize Linear SVM weights (trainable parameters) with the given
   * parameters.
   *
   * @param weights This will be filled with the initialized model weights.
   * @param featureSize The number of features in the training set.
   * @param numClasses Number of classes for classification.
   * @param fitIntercept If true, an intercept is fitted.
   */
  static void InitializeWeights(ParametersType& weights,
                                const size_t featureSize,
                                const size_t numClasses,
                                const bool fitIntercept = false);

  /**
   * Constructs the ground truth label matrix with the passed labels.
   *
   * @param labels Labels associated with the training data.
   * @param groundTruth Reference to sparse matrix to stores the result.
   */
  void GetGroundTruthMatrix(const arma::Row<size_t>& labels,
                            SparseMatType& groundTruth) const;

  /**
   * Evaluate the hinge loss function for all the datapoints
   *
   * @param parameters The parameters of the SVM.
   * @return The value of the loss function for the entire dataset.
   */
  ElemType Evaluate(const ParametersType& parameters) const;

  /**
   * Evaluate the hinge loss function on the specified datapoints.
   *
   * @param parameters The parameters of the SVM.
   * @param firstId Index of the datapoints to use for function
   *      evaluation.
   * @param batchSize Size of batch to process.
   * @return The value of the loss function for the given parameters.
   */
  ElemType Evaluate(const ParametersType& parameters,
                    const size_t firstId,
                    const size_t batchSize = 1) const;

  /**
   * Evaluate the gradient of the hinge loss function following the
   * LinearFunctionType requirements on the Gradient function.
   *
   * @tparam GradType Type of the gradient matrix.
   * @param parameters The parameters of the SVM.
   * @param gradient Linear matrix to output the gradient into.
   */
  template<typename GradType>
  void Gradient(const ParametersType& parameters,
                GradType& gradient) const;

  /**
   * Evaluate the gradient of the hinge loss function, following
   * the LinearFunctionType requirements on the Gradient function.
   *
   * @tparam GradType Type of the gradient matrix.
   * @param parameters The parameters of the SVM.
   * @param firstId Index of the datapoint to use for the gradient evaluation.
   * @param gradient Linear matrix to output the gradient into.
   * @param batchSize Size of the batch to process.
   */
  template<typename GradType>
  void Gradient(const ParametersType& parameters,
                const size_t firstId,
                GradType& gradient,
                const size_t batchSize = 1) const;

  /**
   * Evaluate the gradient of the hinge loss function, following
   * the LinearFunctionType requirements on the Gradient function
   * followed by evaluation of the hinge loss function on all the
   * datapoints
   *
   * @tparam GradType Type of the gradient matrix.
   * @param parameters The parameters of the SVM.
   * @param gradient Linear matrix to output the gradient into.
   * @return The value of the loss function at the given parameters.
   */
  template<typename GradType>
  ElemType EvaluateWithGradient(const ParametersType& parameters,
                                GradType& gradient) const;

  /**
   * Evaluate the gradient of the hinge loss function, following
   * the LinearFunctionType requirements on the Gradient function
   * followed by evaluation of the hinge loss function on the specified
   * datapoints.
   *
   * @tparam GradType Type of the gradient matrix.
   * @param parameters The parameters of the SVM.
   * @param firstId Index of the datapoint to use for the gradient and function
   * evaluation.
   * @param gradient Linear matrix to output the gradient into.
   * @param batchSize Size of the batch to process.
   * @return The value of the loss function at the given parameters.
   */
  template<typename GradType>
  ElemType EvaluateWithGradient(const ParametersType& parameters,
                                const size_t firstId,
                                GradType& gradient,
                                const size_t batchSize = 1) const;

  //! Return the initial point for the optimization.
  const ParametersType& InitialPoint() const { return initialPoint; }
  //! Modify the initial point for the optimization.
  ParametersType& InitialPoint() { return initialPoint; }

  //! Get the dataset.
  const SparseMatType& Dataset() const { return dataset; }
  //! Modify the dataset.
  SparseMatType& Dataset() { return dataset; }

  //! Sets the regularization parameter.
  double& Lambda() { return lambda; }
  //! Gets the regularization parameter.
  double Lambda() const { return lambda; }

  //! Gets the intercept flag.
  bool FitIntercept() const { return fitIntercept; }

  //! Return the number of functions.
  size_t NumFunctions() const;

 private:
  //! The initial point, from which to start the optimization.
  ParametersType initialPoint;

  //! Label matrix for provided data
  SparseMatType groundTruth;

  //! The datapoints for training.  This will be an alias until Shuffle() is
  //! called.
  MatType dataset;

  //! Number of Classes.
  size_t numClasses;

  //! The regularization parameter for L2-regularization.
  double lambda;

  //! The margin between the correct class and all other classes.
  double delta;

  //! Intercept term flag.
  bool fitIntercept;
};

} // namespace mlpack

// Include implementation
#include "linear_svm_function_impl.hpp"

#endif // MLPACK_METHODS_LINEAR_SVM_LINEAR_SVM_FUNCTION_HPP
