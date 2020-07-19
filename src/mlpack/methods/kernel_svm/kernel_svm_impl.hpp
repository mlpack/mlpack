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
    const size_t max_iter,
    const double tol,
    const KernelType& kernel) :
    regularization(regularization),
    fitIntercept(fitIntercept),
    kernel(kernel)
{
  intercept = 0;
  Train(data, labels, max_iter, tol);
}

template <typename MatType, typename KernelType>
KernelSVM<MatType, KernelType>::KernelSVM(
    const size_t inputSize,
    const double regularization,
    const bool fitIntercept,
    const KernelType& kernel) :
    inputSize(inputSize),
    regularization(regularization),
    fitIntercept(fitIntercept),
    kernel(kernel)
{
  intercept = 0;
}

template<typename MatType, typename KernelType>
double KernelSVM<MatType, KernelType>::Train(
    const MatType& data,
    const arma::Row<size_t>& labels,
    const size_t max_iter,
    const double tol)
{
 trainLabels = arma::ones(1, data.n_cols);

  // Changing labels values to 1, -1 values provided
  // by user should 0 and 1.
  for (size_t i = 0; i< data.n_cols; i++)
  {
    if (labels(i) == 0)
      trainLabels(i) = -1;
  }

  // Intializing variable to calculate alphas.
  alpha = arma::zeros(data.n_cols);
  size_t count = 0;

  // Vector to store error value.
  arma::vec E = arma::zeros(data.n_cols, 1);
  // Eta value.
  double eta = 0;

  // Variable to store lower bound and higher bound.
  double L = 0;
  double H = 0;

  // Storing kernel fucntion values.
  arma::mat K = arma::mat(data.n_cols, data.n_cols);

  // Pre-compute the Kernel Matrix
  for (size_t i = 0; i < data.n_cols; i++)
  {
    for (size_t j = 0; j < data.n_cols; j++)
    {
      K(i, j) = kernel.Evaluate(data.col(i), data.col(j));
    }
  }

  // Training starts from here.
  while (count < max_iter)
  {
    size_t num_changed_alphas = 0;
    for (size_t i = 0; i < data.n_cols; i++)
    {
      // Calculate Ei = f(x(i)) - y(i) using (2).
      // E(i) = b + sum (X(i, :) * (repmat(alphas.*Y,1,n).*X)') - Y(i);
      E(i) = intercept + arma::sum(alpha.t() % trainLabels % K.col(i).t())
             - trainLabels(i);
      if ((trainLabels(i) * E(i) < -tol && alpha(i) < regularization) ||
        (trainLabels(i) * E(i) > tol && alpha(i) > 0))
      {
        // In practice, there are many ways one can use to select
        // the i and j. In this simplified code, we select them randomly.
        size_t j = rand() % data.n_cols;
        while (j == i)
        {
          j = rand() % data.n_cols;
        }

        // Calculate Ej = f(x(j)) - y(j) using (2).
        E(j) = intercept + arma::sum(alpha.t() % trainLabels % K.col(j).t())
               - trainLabels(j);

        // Saving old alpha values.
        double alpha_i_old = alpha(i);
        double alpha_j_old = alpha(j);

        // Compute L and H to find max and min values of alphas.
        if (trainLabels(i) == trainLabels(j))
        {
          L = std::max(0.0, alpha(j) + alpha(i) - regularization);
          H = std::min(regularization, alpha(j) + alpha(i));
        }
        else
        {
          L = std::max(0.0, alpha(j) - alpha(i));
          H = std::min(regularization, regularization +
                       alpha(j) - alpha(i));
        }

        if (L == H)
          continue;

        // Compute eta by (14).
        eta = 2 * K(i, j) - K(i, i) - K(j, j);
        if (eta >= 0)
          continue;
        // Compute and clip new value for alpha j using (12) and (15).
        alpha(j) = alpha(j) - (trainLabels(j) * (E(i) - E(j))) / eta;

        // Clip.
        alpha(j) = std::min(H, alpha(j));
        alpha(j) = std::max(L, alpha(j));

        // Check if change in alpha is noticeable or not.
        if (std::abs(alpha(j) - alpha_j_old) < tol)
        {
          alpha(j) = alpha_j_old;
          continue;
        }

        // Determine value for alpha i using (16).
        alpha(i) = alpha(i) + trainLabels(i) * trainLabels(j)
                   * (alpha_j_old - alpha(j));

        // Compute b1 and b2 using (17) and (18) respectively.
        double b1 = intercept - E(i)
                    - trainLabels(i) * (alpha(i) - alpha_i_old) *  K(i, j)
                    - trainLabels(j) * (alpha(j) - alpha_j_old) *  K(i, j);
        double b2 = intercept - E(j)
                    - trainLabels(i) * (alpha(i) - alpha_i_old) *  K(i, j)
                    - trainLabels(j) * (alpha(j) - alpha_j_old) *  K(j, j);

        // Compute b by (19).
        if (0 < alpha(i) && alpha(i) < regularization)
          intercept = b1;
        else if (0 < alpha(j) && alpha(j) < regularization)
          intercept = b2;
        else
          intercept = (b1 + b2) / 2;

        num_changed_alphas = num_changed_alphas + 1;
      }
    }

    if (num_changed_alphas == 0)
      count = count + 1;
    else
      count = 0;
  }
  // Calculating paramter values for linear kernel.
  parameters = (data * (alpha.t() % trainLabels).t()).t();

  trainingData = data;
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
  if (fitIntercept)
  {
    // Giving prediction when non-linear kernel is used.
    scores = arma::zeros(1, data.n_cols);
    double threshold = arma::as_scalar(arma::mean(alpha));
    for (size_t i = 0; i < data.n_cols; i++)
    {
      double  prediction = 0;
      for (size_t j = 0; j < trainingData.n_cols; j++)
      {
        if (alpha(j) <= threshold)
          continue;
        prediction = prediction + alpha(j) *
                     trainLabels(j) * kernel.Evaluate(data.col(i),
                     trainingData.col(j));
      }
      scores(i) = prediction + intercept;
    }
  }
  else
  {
    // Giving predictions for linear kernel.
    scores = (data.t() * parameters.t() + intercept).t();
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
