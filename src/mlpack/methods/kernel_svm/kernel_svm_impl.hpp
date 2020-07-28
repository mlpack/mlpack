/**
 * @file methods/kernel_svm/kernel_svm_impl.hpp
 * @author Himanshu Pathak
 *
 * Implementation of Kernel SVM.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_KERNEL_SVM_KERNEL_SVM_IMPL_HPP
#define MLPACK_METHODS_KERNEL_SVM_KERNEL_SVM_IMPL_HPP

// In case it hasn't been included yet.
#include "kernel_svm.hpp"

namespace mlpack {
namespace svm {

template <typename MatType, typename KernelType>
KernelSVM<MatType, KernelType>::KernelSVM(
    const MatType& data,
    const arma::Row<size_t>& labels,
    const double regularization,
    const bool fitIntercept,
    const size_t numClass,
    const size_t maxIter,
    const double tol) :
    regularization(regularization),
    fitIntercept(fitIntercept),
    numClass(numClass)
{
  numClassifier = double(numClass) * ((double(numClass) - 1) / 2);
  Train(data, labels, maxIter, tol);
}

template <typename MatType, typename KernelType>
KernelSVM<MatType, KernelType>::KernelSVM(
    const double regularization,
    const bool fitIntercept,
    const size_t numClass) :
    regularization(regularization),
    fitIntercept(fitIntercept),
    numClass(numClass)
{
  numClassifier = numClass * (numClass - 1) / 2;
}

template<typename MatType, typename KernelType>
double KernelSVM<MatType, KernelType>::Train(
    const MatType& data,
    const arma::Row<size_t>& labels,
    const size_t maxIter,
    const double tol)
{
  classesClassifier = arma::zeros(numClassifier, 2);

  size_t countClass = 0;
  for (size_t i = 0; i < numClass; i++)
  {
    for (size_t j = i+1; j < numClass; j++)
    {
      classesClassifier(countClass, 0) = i;
      classesClassifier(countClass, 1) = j;
      arma::rowvec tempLables = arma::zeros(1, data.n_cols);
      for (size_t k = 0; k < data.n_cols; k++)
      {
        if (labels(k) == j)
          tempLables(k) = -1;
        if (labels(k) ==  i)
          tempLables(k) = 1;
      }
      KernelSVMFunction<MatType, KernelType> svm(data.cols(arma::find(tempLables == 1 || tempLables == -1)),
                                                          tempLables.cols(arma::find(tempLables == 1 || tempLables == -1)),
                                                          regularization, fitIntercept, maxIter, tol);
      network.push_back(svm);

      countClass++;
    }
  }
}

template <typename MatType, typename KernelType>
void KernelSVM<MatType, KernelType>::Classify(
    const MatType& data,
    arma::Row<size_t>& labels) const
{
  arma::mat scores;
  Classify(data, labels, scores);
}

template <typename MatType, typename KernelType>
void KernelSVM<MatType, KernelType>::Classify(
    const MatType& data,
    arma::Row<size_t>& labels,
    arma::mat& scores) const
{
  Classify(data, scores);
  for (size_t k = 0; k < numClassifier; k++)
  {
    double threshold = arma::as_scalar(arma::mean(scores.row(k), 1));

    for (size_t i = 0; i< data.n_cols; i++)
    {
      if (scores(k, i) >= threshold)
        scores(k, i) = 1;
      if (scores(k, i) < threshold)
        scores(k, i) = 0;
    }
  }

  arma::mat prediction = arma::zeros(numClass, data.n_cols);
  for (size_t i = 0; i< data.n_cols; i++)
  {
    for (size_t k = 0; k < numClassifier; k++)
    {
      if (scores(k, i) == 1)
        prediction(classesClassifier(k, 0), i) += 1;
      if (scores(k, i) == 0)
        prediction(classesClassifier(k, 1), i) += 1;
    } 
  }

  labels.zeros(data.n_cols);
  for (int i = 0; i < data.n_cols; i++)
  {
    labels(i) = prediction.col(i).index_max();
  }

}

template <typename MatType, typename KernelType>
void KernelSVM<MatType, KernelType>::Classify(
    const MatType& data,
    arma::mat& scores) const
{
  // Giving prediction when non-linear kernel is used.
  scores = arma::zeros(numClassifier, data.n_cols);
  for (size_t k = 0; k < numClassifier; k++)
  {
    scores.row(k) = network[k].Classify(data);
  }
}

template <typename MatType, typename KernelType>
template <typename VecType>
size_t KernelSVM<MatType, KernelType>::Classify(const VecType& point) const
{
  arma::Row<size_t> label(1);
  Classify(point, label);
  return size_t(label(0));
}

template <typename MatType, typename KernelType>
double KernelSVM<MatType, KernelType>::ComputeAccuracy(
    const MatType& testData,
    const arma::Row<size_t>& testLabels) const
{
  arma::Row<size_t> labels;

  // Get predictions for the provided data.
  Classify(testData, labels);

  // Increment count for every correctly predicted label.
  size_t count = 0;
  for (size_t i = 0; i < labels.n_elem ; i++)
    if (testLabels(i) == labels(i))
      count++;

  // Return the accuracy.
  return (double) count / labels.n_elem;
}

} // namespace svm
} // namespace mlpack

#endif // MLPACK_METHODS_KERNEL_SVM_KERNEL_SVM_IMPL_HPP
