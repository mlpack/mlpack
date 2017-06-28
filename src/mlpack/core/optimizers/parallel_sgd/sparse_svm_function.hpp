/**
 * @file parallel_sgd_impl.hpp
 * @author Shikhar Bhardwaj
 *
 * Implementation of the hinge loss function for training a sparse SVM with the
 * parallel SGD algorithm
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_PARALLEL_SGD_SPARSE_SVM_HPP
#define MLPACK_CORE_OPTIMIZERS_PARALLEL_SGD_SPARSE_SVM_HPP
#include <mlpack/prereqs.hpp>

class SparseSVMLossFunction{
 public:
  //! Nothing to do for the default constructor.
  SparseSVMLossFunction() = default;

  //! Member initialization constructor.
  SparseSVMLossFunction(arma::sp_mat& dataset, arma::vec& labels);

  //! Evaluate a function.
  double Evaluate(const arma::vec& weights, size_t id);

  //! Evaluate the gradient of a function.
  void Gradient(const arma::vec& weights, size_t id, arma::sp_mat& gradient);

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
  //! The datapoints for training.
  arma::sp_mat dataset;

  //! The labels, y_i.
  arma::vec labels;
};

// Include implementation
#include "sparse_svm_function_impl.hpp"

#endif
