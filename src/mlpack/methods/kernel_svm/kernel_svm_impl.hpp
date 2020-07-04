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
    const double C,
    const bool fitIntercept,
    const size_t max_iter,
    const double tol,
    const KernelType& kernel) :
    C(C),
    fitIntercept(fitIntercept),
    kernel(kernel)
{
  Train(data, labels, max_iter, tol);
}

template <typename MatType, typename KernelType>
KernelSVM<MatType, KernelType>::KernelSVM(
    const size_t inputSize,
    const double C,
    const bool fitIntercept,
    const KernelType& kernel) :
    inputSize(inputSize),
    C(C),
    fitIntercept(fitIntercept),
    kernel(kernel)
{
  // No training to do here.
}

template<typename MatType, typename KernelType>
double KernelSVM<MatType, KernelType>::Train(
    const MatType& data,
    const arma::Row<size_t>& labels,
    const size_t max_iter,
    const double tol)
{
  arma::Row<int> label(data.n_cols);

  for (size_t i = 0; i< data.n_cols; i++)
  {
    if (labels(i) == 0)
      label(i) = -1;
    else
      label(i) = 1;
  }

  alpha = arma::zeros(data.n_cols);
  size_t count = 0;
  b = 0;
  arma::vec E = arma::zeros(data.n_cols, 1);
  double eta = 0;
  double L = 0;
  double H = 0;
  arma::mat K = arma::mat(data.n_cols, data.n_cols);

  for (size_t i = 0; i < data.n_cols; i++)
  {
    for (size_t j = 0; j < data.n_cols; j++)
    {
      K(i, j) = kernel.Evaluate(data.col(i), data.col(j));
    }
  }
  while (count < max_iter)
  {
    size_t num_changed_alphas = 0;
    for (size_t i = 0; i < data.n_cols; i++)
    {
      E(i) = b + arma::sum(alpha.t() % label % K.col(i).t()) - label(i);
      if ((label(i) * E(i) < -tol && alpha(i) < C) ||
        (label(i) * E(i) > tol && alpha(i) > 0))
      {
        size_t j = rand() % data.n_cols;
        while (j == i)
        {
          j = rand() % data.n_cols;
        }
        E(j) = b + arma::sum(alpha.t() % label % K.col(j).t()) - label(j);

        double alpha_i_old = alpha(i);
        double alpha_j_old = alpha(j);

        if (label(i) == label(j))
        {
          L = std::max(0.0, alpha(j) + alpha(i) - C);
          H = std::min(C, alpha(j) + alpha(i));
        }
        else
        {
          L = std::max(0.0, alpha(j) - alpha(i));
          H = std::min(C, C + alpha(j) - alpha(i));
        }

        if (L == H)
          continue;

        eta = 2 * K(i, j) - K(i, i) - K(j, j);
        if (eta >= 0)
          continue;

        alpha(j) = alpha(j) - (label(j) * (E(i) - E(j))) / eta;

        alpha(j) = std::min(H, alpha(j));
        alpha(j) = std::max(L, alpha(j));

        if (std::abs(alpha(j) - alpha_j_old) < tol)
        {
          alpha(j) = alpha_j_old;
          continue;
        }

        alpha(i) = alpha(i) + label(i) * label(j) * (alpha_j_old - alpha(j));

        double b1 = b - E(i)
                    - label(i) * (alpha(i) - alpha_i_old) *  K(i, j)
                    - label(j) * (alpha(j) - alpha_j_old) *  K(i, j);
        double b2 = b - E(j)
                    - label(i) * (alpha(i) - alpha_i_old) *  K(i, j)
                    - label(j) * (alpha(j) - alpha_j_old) *  K(j, j);

        if (0 < alpha(i) && alpha(i) < C)
          b = b1;
        else if (0 < alpha(j) && alpha(j) < C)
          b = b2;
        else
          b = (b1 + b2) / 2;

        num_changed_alphas = num_changed_alphas + 1;
      }
    }

    if (num_changed_alphas == 0)
      count = count + 1;
    else
      count = 0;
  }
  parameters = (data * (alpha.t() % label).t()).t();
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
  double threshold = arma::as_scalar(arma::mean(scores, 1));

  // Prepare necessary data.
  labels.zeros(data.n_cols);

  for (size_t i = 0; i< scores.n_elem; i++)
  {
    if (scores(i) >= threshold)
      labels(i) = 1;
    if (scores(i) < threshold)
      labels(i) = 0;
  }
}

template <typename MatType, typename KernelType>
void KernelSVM<MatType, KernelType>::Classify(
    const MatType& data,
    arma::mat& scores) const
{
  scores = (data.t() * parameters.t() + b).t();
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
