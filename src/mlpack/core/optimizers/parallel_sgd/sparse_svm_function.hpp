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
  SparseSVMLossFunction() = default;
  SparseSVMLossFunction(arma::SpMat<double>& dataset, arma::vec& labels);
  double Evaluate(const arma::vec& weights, size_t id);
  void Gradient(const arma::vec& weights, size_t id, arma::mat& gradient);
  arma::Col<size_t> Components(size_t id);

  const arma::SpMat<double>& Dataset() const { return dataset; }
  arma::SpMat<double>& Dataset() { return dataset; }

  const arma::vec& Labels() const { return labels; }
  arma::vec& Labels() { return labels; }

  size_t NumFunctions();

 private:
  arma::SpMat<double> dataset;
  arma::vec labels;
};

// Include implementation
#include "sparse_svm_function_impl.hpp"

#endif
