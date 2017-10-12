/**
 * @file sparse_svm_function_impl.hpp
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
#ifndef MLPACK_METHODS_SPARSE_SVM_SPARSE_SVM_FUNCTION_IMPL_HPP
#define MLPACK_METHODS_SPARSE_SVM_SPARSE_SVM_FUNCTION_IMPL_HPP

// In case it hasn't been included yet.
#include "sparse_svm_function.hpp"

SparseSVMFunction::SparseSVMFunction(
    const arma::sp_mat& dataset, const arma::vec& labels) :
    dataset(dataset),
    labels(math::MakeAlias(const_cast<arma::vec&>(labels), false))
{ /* Nothing to do */ }

void SparseSVMFunction::Shuffle()
{
  arma::sp_mat newDataset;
  arma::vec newLabels;

  // Shuffle the data.
  math::ShuffleData(dataset, labels, newDataset, newLabels);

  math::ClearAlias(newLabels);

  dataset = std::move(newDataset);
  labels = std::move(newLabels);
}

double SparseSVMFunction::Evaluate(const arma::mat& parameters,
                                   const size_t firstId,
                                   const size_t batchSize)
{
  // The hinge loss function.
  const size_t lastId = firstId + batchSize - 1;
  return arma::accu(arma::max(0.0, 1 - labels.subvec(firstId, lastId) %
      dataset.cols(firstId, lastId) *
      arma::repmat(parameters, 1, batchSize).t()));
}

template <typename GradType>
void SparseSVMFunction::Gradient(
    const arma::mat& parameters,
    const size_t firstId,
    GradType& gradient,
    const size_t batchSize)
{
  // Evaluate the gradient of the hinge loss function.
  const size_t lastId = firstId + batchSize - 1;
  arma::vec dots = 1 - labels.subvec(firstId, lastId) %
      dataset.cols(firstId, lastId) *
      arma::repmat(parameters, 1, batchSize).t();
  gradient = GradType(parameters.n_rows, 1);
  for (size_t i = 0; i < batchSize; ++i)
  {
    if (dots[i] >= 0)
    {
      // Is this correct?
      gradient += -1 * GradType(dataset.col(id) * labels(id));
    }
  }
}

size_t SparseSVMFunction::NumFunctions()
{
  // The number of points in the dataset is the number of functions, as this
  // is a data dependent function.
  return dataset.n_cols;
}

#endif // MLPACK_METHODS_SPARSE_SVM_SPARSE_SVM_FUNCTION_IMPL_HPP
