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
    const size_t numClasses,
    const double delta,
    const double C,
    const std::string& kernelFunction,
    const bool fitIntercept,
    const size_t max_iter,
    const double tol) :
    numClasses(numClasses),
    delta(delta),
    C(C),
    kernelFunction(kernelFunction),
    fitIntercept(fitIntercept),
    max_iter(max_iter)
{
  Train(data, labels, numClasses, max_iter, tol);
}

template <typename MatType, typename KernelType>
KernelSVM<MatType, KernelType>::KernelSVM(
    const size_t inputSize,
    const size_t numClasses,
    const double delta,
    const double C,
    const std::string& kernelFunction,
    const bool fitIntercept) :
    inputSize(inputSize),
    numClasses(numClasses),
    delta(delta),
    C(C),
    kernelFunction(kernelFunction),
    fitIntercept(fitIntercept)
{
  // No training to do here.
}

template<typename MatType, typename KernelType>
double KernelSVM<MatType, KernelType>::Train(
    const MatType& data,
    const arma::Row<size_t>& labels,
    const size_t numClasses,
    const size_t max_iter,
    const double tol)
{
  alpha = arma::zeros(data.n_cols);
  size_t count = 0;
  b = 0;
  arma::vec E = arma::zeros(data.n_cols, 1);
  double eta = 0;
  double L = 0;
  double H = 0;
  arma::mat K;

  if (kernelFunction == "linear")
  {
    K = data.t() * data;
  }

  else if (kernelFunction == "gaussian")
  {
    arma::vec X2 = arma::sum(arma::pow(data, 2), 2);
    K = X2 + (X2.t() - 2 * data * data.t());
  }
  else
  {
    K = arma::zeros(data.n_rows, data.n_cols);
  }

  while(count < max_iter)
  {
    size_t num_changes_alphas = 0;
    for(size_t i = 0; i < data.n_cols; i++)
    {
      E(i) = b + arma::sum(alpha % (labels.t() % K.col(i))) - labels(i);
      if ((labels(i) * E(i) < -tol && alpha(i) < C) 
            || (labels(i) * E(i) > tol && alpha(i) > 0))
      {
        size_t j = rand() % data.n_cols;
        while (j == i)
        {
          j = rand() % data.n_cols;
        }

        E(j) = b + arma::sum(alpha % (labels.t() % K.col(j))) - labels(j);
        double alpha_j_old = alpha(j);
        double alpha_i_old = alpha(i);
        if(labels(i) == labels(j))
        {
          L = std::max(0.0, alpha(i) + alpha(j) - C);
          H = std::min(C, alpha(i) + alpha(j));
        }
        else
        {
          L = std::max(0.0, alpha(j) - alpha(i));
          H = std::min(C, C + alpha(j) - alpha(i)); 
        }

        if(L == H)
          continue;

        eta = 2 * K(i, j) - K(i, i) - K(j, j);

        if(eta >= 0)
          continue;

        alpha(j) = alpha(j) - (labels(i) * (E(i) - E(j))) / eta;

        alpha(j) = std::min(H, alpha(j));
        alpha(j) = std::max(L, alpha(j));

        if(abs(alpha(j) - alpha_j_old) < tol)
        {
          alpha(j) = alpha_j_old;
          continue;
        }

        alpha(i) = alpha(i) + labels(i) * labels(j) * (alpha_j_old - alpha(j));

        double b1 = b - E(i)
                    - labels(i) * (alpha(i) - alpha_i_old) * K(i, j)
                    - labels(j) * (alpha(j) - alpha_i_old) * K(i, j);

        double b2 = b - E(j)
                    - labels(i) * (alpha(i) - alpha_i_old) * K(i, j)
                    - labels(j) * (alpha(j) - alpha_i_old) * K(j, j);

        if (0 < alpha(i) && alpha(j) < C)
        {
          b = b1;
        }
        else if (0 < alpha(j) && alpha(j)< C)
        {
          b = b2;
        }
        else
        {
          b = (b1 + b2) / 2;
        }
        num_changes_alphas = num_changes_alphas + 1;
      }
    }
    if (num_changes_alphas == 0)
      count = count + 1;
    else
      count = 0;
  }
  w = ((alpha.t() % labels) * data.t());
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

  // Prepare necessary data.
  labels.zeros(data.n_cols);

  labels = arma::conv_to<arma::Row<size_t>>::from(
      arma::index_max(scores));
}

template <typename MatType, typename KernelType>
void KernelSVM<MatType, KernelType>::Classify(
    const MatType& data,
    arma::mat& scores) const
{
  if (kernelFunction == "linear")
  {
    scores = data.t() * w.t() + b;
    scores = scores.t();
  }
  else if (kernelFunction == "gaussian")
  {
    MatType X2 = arma::sum(arma::pow(data, 2), 2);
    MatType K = X2 + (X2.t() - 2 * data * data.t());
    K = alpha.t() * K;
    scores = arma::sum(K, 2);
  }
  else
  {}
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
