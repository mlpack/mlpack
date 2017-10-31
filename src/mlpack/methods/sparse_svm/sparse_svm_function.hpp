/**
 * @file sparse_svm_function.hpp
 * @author Shikhar Bhardwaj
 *
 * Implementation of the hinge loss function for training a sparse SVM with the
 * parallel SGD algorithm.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_SPARSE_SVM_SPARSE_SVM_FUNCTION_HPP
#define MLPACK_METHODS_SPARSE_SVM_SPARSE_SVM_FUNCTION_HPP

#include <mlpack/prereqs.hpp>

class SparseSVMFunction
{
 public:
  //! Nothing to do for the default constructor.
  SparseSVMFunction() {}

  //! Member initialization constructor.
  SparseSVMFunction(const arma::sp_mat& dataset, const arma::vec& labels);

  /**
   * Shuffle the dataset.
   */
  void Shuffle();

  /**
   * Evaluate the hinge loss function on the specified datapoints.
   *
   * @param parameters The parameters of the SVM.
   * @param startId First index of the datapoints to use for function
   *      evaluation.
   * @param batchSize Size of batch to process.
   * @return The value of the loss function at the given parameters.
   */
  double Evaluate(const arma::mat& parameters,
                  const size_t startId,
                  const size_t batchSize = 1);

  /**
   * Evaluate the gradient the gradient of the hinge loss function, following
   * the SparseFunctionType requirements on the Gradient function.
   *
   * @param parameters The parameters of the SVM.
   * @param id Index of the datapoint to use for the gradient evaluation.
   * @param gradient Sparse matrix to output the gradient into.
   */
  template <typename GradType>
  void Gradient(const arma::mat& parameters, size_t id, GradType& gradient);

  //! Return the initial point for the optimization.
  const arma::mat& InitialPoint() const { return initialPoint; }
  //! Modify the initial point for the optimization.
  arma::mat& InitialPoint() { return initialPoint; }

  //! Get the dataset.
  const arma::sp_mat& Dataset() const { return dataset; }
  //! Modify the dataset.
  arma::sp_mat& Dataset() { return dataset; }

  //! Get the labels.
  const arma::vec& Labels() const { return labels; }
  //! Modify the labels.
  arma::vec& Labels() { return labels; }

  //! Return the number of functions.
  size_t NumFunctions();

 private:
  //! The initial point, from which to start the optimization.
  arma::mat initialPoint;

  //! The datapoints for training.
  arma::sp_mat dataset;

  //! The labels, y_i.
  arma::vec labels;
};

// Include implementation
#include "sparse_svm_function_impl.hpp"

#endif // MLPACK_METHODS_SPARSE_SVM_SPARSE_SVM_FUNCTION_HPP
