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
template <typename OptimizerType>
KernelSVM<MatType, KernelType>::KernelSVM(
    const MatType& data,
    const arma::Row<size_t>& labels,
    const size_t numClasses,
    const double lambda,
    const double delta,
    const bool fitIntercept,
    OptimizerType optimizer) :
    numClasses(numClasses),
    lambda(lambda),
    delta(delta),
    fitIntercept(fitIntercept)
{
  Train(data, labels, numClasses, optimizer);
}

template <typename MatType, typename KernelType>
KernelSVM<MatType, KernelType>::KernelSVM(
    const size_t inputSize,
    const size_t numClasses,
    const double lambda,
    const double delta,
    const bool fitIntercept) :
    numClasses(numClasses),
    lambda(lambda),
    delta(delta),
    fitIntercept(fitIntercept)
{
}

template <typename MatType, typename KernelType>
KernelSVM<MatType, KernelType>::KernelSVM(
    const size_t numClasses,
    const double lambda,
    const double delta,
    const bool fitIntercept) :
    numClasses(numClasses),
    lambda(lambda),
    delta(delta),
    fitIntercept(fitIntercept)
{
  // No training to do here.
}

template <typename MatType, typename KernelType>
template <typename OptimizerType, typename... CallbackTypes>
double KernelSVM<MatType, KernelType>::Train(
    const MatType& data,
    const arma::Row<size_t>& labels,
    const size_t numClasses,
    OptimizerType optimizer,
    CallbackTypes&&... callbacks)
{
  arma::vec alpha = arma::zeros(data.n_cols);
  size_t count = 0;

  while(true)
  {
    arma::vec alpha_prev = arma::vec(alpha);

    for(size_t j = 0; j < data.n_cols; j++)
    {
      size_t i = rand()%((data.n_cols - 1) + 1);
      double k_ij = KernelType::Evaluate(data.col(i), data.col(i)) +
                    KernelType::Evaluate(data.col(j), data.col(j)) -
                    2 * KernelType::Evaluate(data.col(i), data.col(j));
      if (k_ij == 0)
        continue;
      double alpha_prime_j = alpha(j);
      double alpha_prime_i = alpha(i);
      double L = ComputeL(alpha_prime_j, alpha_prime_i, labels(j), labels(i));
      double H = ComputeH(alpha_prime_j, alpha_prime_i, labels(j), labels(i));
      // Compute model parameters
      arma::mat w = calc_w(alpha, y, X)
      arma::mat b = calc_b(X, y, self.w)

      // Compute E_i, E_j
      E_i = E(x_i, y_i, self.w, self.b)
      E_j = E(x_j, y_j, self.w, self.b)

      // Set new alpha values
      alpha(j) = alpha_prime_j + float(labels(j) * (E_i - E_j))/k_ij;
      alpha(j) = max(alpha(j), L);
      alpha(j) = min(alpha(j), H);

      alpha(i) = alpha_prime_i + labels(i)*labels(j) * (alpha_prime_j - alpha(j));
    }

    // Check convergence
    arma::mat diff = arma::norm(alpha - alpha_prev);
    if (diff < epsilon)
      break;

    if (count >= max_iter)
    {
      std::LOG<<"Iteration number exceeded the max iterations" <<max_iter;
      return;
    }
  // Compute final model parameters
  b = calc_b(data, labels, w)
  if (kernel_type == 'linear')
    w = calc_w(alpha, labels, data);
  // Get support vectors
  arma::mat alpha_idx = where(alpha > 0)[0];
  arma::mat support_vectors;
  return count
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
  if (data.n_rows != FeatureSize())
  {
    std::ostringstream oss;
    oss << "LinearSVM::Classify(): dataset has " << data.n_rows
        << " dimensions, but model has " << FeatureSize() << " dimensions!";
    throw std::invalid_argument(oss.str());
  }

  if (fitIntercept)
  {
    scores = parameters.rows(0, parameters.n_rows - 2).t() * data
        + arma::repmat(parameters.row(parameters.n_rows - 1).t(), 1,
        data.n_cols);
  }
  else
  {
    scores = parameters.t() * data;
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
