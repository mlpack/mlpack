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
#ifndef MLPACK_METHODS_KERNEL_SVM_KERNEL_SVM_FUNCTION_IMPL_HPP
#define MLPACK_METHODS_KERNEL_SVM_KERNEL_SVM_FUNCTION_IMPL_HPP

// In case it hasn't been included yet.
#include "kernel_svm.hpp"

namespace mlpack {
namespace svm {

template <typename MatType, typename KernelType>
KernelSVMFunction<MatType, KernelType>::KernelSVMFunction(
    const MatType& data,
    const arma::rowvec& labels,
    const double regularization,
    const bool fitIntercept,
    const size_t maxIter,
    const double tol,
    const KernelType kernel) :
    regularization(regularization),
    fitIntercept(fitIntercept),
    kernel(kernel)
{
  intercept = 0;
  Train(data, labels, maxIter, tol);
}

template <typename MatType, typename KernelType>
KernelSVMFunction<MatType, KernelType>::KernelSVMFunction(
    const double regularization,
    const bool fitIntercept,
    const KernelType kernel) :
    regularization(regularization),
    fitIntercept(fitIntercept),
    kernel(kernel)
{
  intercept = 0;
}

template<typename MatType, typename KernelType>
double KernelSVMFunction<MatType, KernelType>::Train(
    const MatType& data,
    const arma::rowvec& labels,
    const size_t maxIter,
    const double tol)
{
  // Saving values of sample data to be used
  // with kernel function.
  trainingData = data;

  // Changing labels values to 1, -1 values provided
  // by user should 0 and 1.
  trainCoefficients = labels;

  // Intializing variable to calculate alphas.
  alpha = arma::zeros(1, data.n_cols);
  size_t count = 0;

  // Vector to store error value.
  arma::vec E = arma::zeros(data.n_cols, 1);
  // Eta value.
  double eta = 0;

  // Variable to store lower bound and higher bound.
  double L = 0;
  double H = 0;

  // Storing kernel function values.
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
  while (count < maxIter)
  {
    size_t changedAlphas = 0;
    for (size_t i = 0; i < data.n_cols; i++)
    {
      // Calculate Ei = f(x(i)) - y(i) using (2).
      // E(i) = b + sum (X(i, :) * (repmat(alphas.*Y,1,n).*X)') - Y(i);
      E(i) = intercept + arma::as_scalar(arma::sum(alpha
             % trainCoefficients % K.col(i).t()))- trainCoefficients(i);

      if ((trainCoefficients(i) * E(i) < -tol && alpha(i) < regularization) ||
        (trainCoefficients(i) * E(i) > tol && alpha(i) > 0))
      {
        // In practice, there are many ways one can use to select
        // the i and j. In this simplified code, we select them randomly.
        size_t j = rand() % data.n_cols;
        while (j == i)
        {
          j = rand() % data.n_cols;
        }
        assert(j < data.n_cols && j >= 0);

        // Calculate Ej = f(x(j)) - y(j) using (2).
        E(j) = intercept + arma::as_scalar(arma::sum(alpha
               % trainCoefficients % K.col(j).t())) - trainCoefficients(j);

        // Saving old alpha values.
        double alphaIold = alpha(i);
        double alphaJold = alpha(j);

        // Compute L and H to find max and min values of alphas.
        if (trainCoefficients(i) == trainCoefficients(j))
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
        alpha(j) = alpha(j) - (trainCoefficients(j) * (E(i) - E(j))) / eta;

        // Clip.
        alpha(j) = std::min(H, alpha(j));
        alpha(j) = std::max(L, alpha(j));

        // Check if change in alpha is noticeable or not.
        if (std::abs(alpha(j) - alphaJold) < tol)
        {
          alpha(j) = alphaJold;
          continue;
        }

        // Determine value for alpha i using (16).
        alpha(i) = alpha(i) + trainCoefficients(i) * trainCoefficients(j)
                   * (alphaJold - alpha(j));

        // Compute b1 and b2 using (17) and (18) respectively.
        double b1 = intercept - E(i)
                    - trainCoefficients(i) * (alpha(i) - alphaIold) *  K(i, j)
                    - trainCoefficients(j) * (alpha(j) - alphaJold) *  K(i, j);
        double b2 = intercept - E(j)
                    - trainCoefficients(i) * (alpha(i) - alphaIold) *  K(i, j)
                    - trainCoefficients(j) * (alpha(j) - alphaJold) *  K(j, j);

        // Compute b by (19).
        if (0 < alpha(i) && alpha(i) < regularization)
          intercept = b1;
        else if (0 < alpha(j) && alpha(j) < regularization)
          intercept = b2;
        else
          intercept = (b1 + b2) / 2;

        changedAlphas++;
      }
    }

    if (changedAlphas == 0)
      count++;
    else
      count = 0;
  }

  return eta;
}

template <typename MatType, typename KernelType>
arma::rowvec KernelSVMFunction<MatType, KernelType>::Classify(
    const MatType& data) const
{
  // Giving prediction when non-linear kernel is used.
  arma::rowvec scores = arma::zeros(1, data.n_cols);
  double threshold = arma::as_scalar(arma::mean(alpha));
  for (size_t i = 0; i < data.n_cols; i++)
  {
    double  prediction = 0;
    for (size_t j = 0; j < trainingData.n_cols; j++)
    {
      if (alpha(j) <= threshold)
        continue;
      prediction = prediction + alpha(j) *
                   trainCoefficients(j) * kernel.Evaluate(data.col(i),
                   trainingData.col(j));
    }
    if (!fitIntercept)
      scores(i) = prediction;
    else
      scores(i) = prediction + intercept;
  }
  return scores;
}

} // namespace svm
} // namespace mlpack

#endif // MLPACK_METHODS_KERNEL_SVM_KERNEL_SVM_IMPL_HPP
