/**
 * @file nca_softmax_impl.h
 * @author Ryan Curtin
 *
 * Implementation of the Softmax error function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_NCA_NCA_SOFTMAX_ERROR_FUNCTCLIN_IMPL_H
#define MLPACK_METHODS_NCA_NCA_SOFTMAX_ERROR_FUNCTCLIN_IMPL_H

// In case it hasn't been included already.
#include "nca_softmax_error_function.hpp"

namespace mlpack {
namespace nca {

// Initialize with the given kernel.
template<typename MetricType>
SoftmaxErrorFunction<MetricType>::SoftmaxErrorFunction(
    const arma::mat& dataset,
    const arma::Row<size_t>& labels,
    MetricType metric) :
    dataset(dataset),
    labels(labels),
    metric(metric),
    precalculated(false)
{ /* nothing to do */ }

//! The non-separable implementation, which uses Precalculate() to save time.
template<typename MetricType>
double SoftmaxErrorFunction<MetricType>::Evaluate(const arma::mat& coordinates)
{
  // Calculate the denominators and numerators, if necessary.
  Precalculate(coordinates);

  return -accu(p); // Sum of p_i for all i.  We negate because our solver
                   // minimizes, not maximizes.
};

//! The separated objective function, which does not use Precalculate().
template<typename MetricType>
double SoftmaxErrorFunction<MetricType>::Evaluate(const arma::mat& coordinates,
                                                  const size_t i)
{
  // Unfortunately each evaluation will take O(N) time because it requires a
  // scan over all points in the dataset.  Our objective is to compute p_i.
  double denominator = 0;
  double numerator = 0;

  // It's quicker to do this now than one point at a time later.
  stretchedDataset = coordinates * dataset;

  for (size_t k = 0; k < dataset.n_cols; ++k)
  {
    // Don't consider the case where the points are the same.
    if (k == i)
      continue;

    // We want to evaluate exp(-D(A x_i, A x_k)).
    double eval = std::exp(-metric.Evaluate(stretchedDataset.unsafe_col(i),
                                            stretchedDataset.unsafe_col(k)));

    // If they are in the same class, update the numerator.
    if (labels[i] == labels[k])
      numerator += eval;

    denominator += eval;
  }

  // Now the result is just a simple division, but we have to be sure that the
  // denominator is not 0.
  if (denominator == 0.0)
  {
    Log::Warn << "Denominator of p_" << i << " is 0!" << std::endl;
    return 0;
  }

  return -(numerator / denominator); // Negate because the optimizer is a
                                     // minimizer.
}

//! The non-separable implementation, where Precalculate() is used.
template<typename MetricType>
void SoftmaxErrorFunction<MetricType>::Gradient(const arma::mat& coordinates,
                                                arma::mat& gradient)
{
  // Calculate the denominators and numerators, if necessary.
  Precalculate(coordinates);

  // Now, we handle the summation over i:
  //   sum_i (p_i sum_k (p_ik x_ik x_ik^T) -
  //       sum_{j in class of i} (p_ij x_ij x_ij^T)
  // We can algebraically manipulate the whole thing to produce a more
  // memory-friendly way to calculate this.  Looping over each i and k (again
  // O((n * (n + 1)) / 2) as with the last step, we can add the following to the
  // sum:
  //
  //   if class of i is the same as the class of k, add
  //     (((p_i - (1 / p_i)) p_ik) + ((p_k - (1 / p_k)) p_ki)) x_ik x_ik^T
  //   otherwise, add
  //     (p_i p_ik + p_k p_ki) x_ik x_ik^T
  arma::mat sum;
  sum.zeros(stretchedDataset.n_rows, stretchedDataset.n_rows);
  for (size_t i = 0; i < stretchedDataset.n_cols; i++)
  {
    for (size_t k = (i + 1); k < stretchedDataset.n_cols; k++)
    {
      // Calculate p_ik and p_ki first.
      double eval = exp(-metric.Evaluate(stretchedDataset.unsafe_col(i),
                                         stretchedDataset.unsafe_col(k)));
      double p_ik = 0, p_ki = 0;
      p_ik = eval / denominators(i);
      p_ki = eval / denominators(k);

      // Subtract x_i from x_k.  We are not using stretched points here.
      arma::vec x_ik = dataset.col(i) - dataset.col(k);
      arma::mat secondTerm = (x_ik * trans(x_ik));

      if (labels[i] == labels[k])
        sum += ((p[i] - 1) * p_ik + (p[k] - 1) * p_ki) * secondTerm;
      else
        sum += (p[i] * p_ik + p[k] * p_ki) * secondTerm;
    }
  }

  // Assemble the final gradient.
  gradient = -2 * coordinates * sum;
}

//! The separable implementation.
template<typename MetricType>
void SoftmaxErrorFunction<MetricType>::Gradient(const arma::mat& coordinates,
                                                const size_t i,
                                                arma::mat& gradient)
{
  // We will need to calculate p_i before this evaluation is done, so these two
  // variables will hold the information necessary for that.
  double numerator = 0;
  double denominator = 0;

  // The gradient involves two matrix terms which are eventually combined into
  // one.
  arma::mat firstTerm;
  arma::mat secondTerm;

  firstTerm.zeros(coordinates.n_rows, coordinates.n_cols);
  secondTerm.zeros(coordinates.n_rows, coordinates.n_cols);

  // Compute the stretched dataset.
  stretchedDataset = coordinates * dataset;

  for (size_t k = 0; k < dataset.n_cols; ++k)
  {
    // Don't consider the case where the points are the same.
    if (i == k)
      continue;

    // Calculate the numerator of p_ik.
    double eval = exp(-metric.Evaluate(stretchedDataset.unsafe_col(i),
                                       stretchedDataset.unsafe_col(k)));

    // If the points are in the same class, we must add to the second term of
    // the gradient as well as the numerator of p_i.  We will divide by the
    // denominator of p_ik later.  For x_ik we are not using stretched points.
    arma::vec x_ik = dataset.col(i) - dataset.col(k);
    if (labels[i] == labels[k])
    {
      numerator += eval;
      secondTerm += eval * x_ik * trans(x_ik);
    }

    // We always have to add to the denominator of p_i and the first term of the
    // gradient computation.  We will divide by the denominator of p_ik later.
    denominator += eval;
    firstTerm += eval * x_ik * trans(x_ik);
  }

  // Calculate p_i.
  double p = 0;
  if (denominator == 0)
  {
    Log::Warn << "Denominator of p_" << i << " is 0!" << std::endl;
    // If the denominator is zero, then all p_ik should be zero and there is
    // no gradient contribution from this point.
    gradient.zeros(coordinates.n_rows, coordinates.n_rows);
    return;
  }
  else
  {
    p = numerator / denominator;
    firstTerm /= denominator;
    secondTerm /= denominator;
  }

  // Now multiply the first term by p_i, and add the two together and multiply
  // all by 2 * A.  We negate it though, because our optimizer is a minimizer.
  gradient = -2 * coordinates * (p * firstTerm - secondTerm);
}

template<typename MetricType>
const arma::mat SoftmaxErrorFunction<MetricType>::GetInitialPoint() const
{
  return arma::eye<arma::mat>(dataset.n_rows, dataset.n_rows);
}

template<typename MetricType>
void SoftmaxErrorFunction<MetricType>::Precalculate(
    const arma::mat& coordinates)
{
  // Ensure it is the right size.
  lastCoordinates.set_size(coordinates.n_rows, coordinates.n_cols);

  // Make sure the calculation is necessary.
  if ((accu(coordinates == lastCoordinates) == coordinates.n_elem) &&
      precalculated)
    return; // No need to calculate; we already have this stuff saved.

  // Coordinates are different; save the new ones, and stretch the dataset.
  lastCoordinates = coordinates;
  stretchedDataset = coordinates * dataset;

  // For each point i, we must evaluate the softmax function:
  //   p_ij = exp( -K(x_i, x_j) ) / ( sum_{k != i} ( exp( -K(x_i, x_k) )))
  //   p_i = sum_{j in class of i} p_ij
  // We will do this by keeping track of the denominators for each i as well as
  // the numerators (the sum for all j in class of i).  This will be on the
  // order of O((n * (n + 1)) / 2), which really isn't all that great.
  p.zeros(stretchedDataset.n_cols);
  denominators.zeros(stretchedDataset.n_cols);
  for (size_t i = 0; i < stretchedDataset.n_cols; i++)
  {
    for (size_t j = (i + 1); j < stretchedDataset.n_cols; j++)
    {
      // Evaluate exp(-d(x_i, x_j)).
      double eval = exp(-metric.Evaluate(stretchedDataset.unsafe_col(i),
                                         stretchedDataset.unsafe_col(j)));

      // Add this to the denominators of both p_i and p_j: K(i, j) = K(j, i).
      denominators[i] += eval;
      denominators[j] += eval;

      // If i and j are the same class, add to numerator of both.
      if (labels[i] == labels[j])
      {
        p[i] += eval;
        p[j] += eval;
      }
    }
  }

  // Divide p_i by their denominators.
  p /= denominators;

  // Clean up any bad values.
  for (size_t i = 0; i < stretchedDataset.n_cols; i++)
  {
    if (denominators[i] == 0.0)
    {
      Log::Debug << "Denominator of p_{" << i << ", j} is 0." << std::endl;

      // Set to usable values.
      denominators[i] = std::numeric_limits<double>::infinity();
      p[i] = 0;
    }
  }

  // We've done a precalculation.  Mark it as done.
  precalculated = true;
}

} // namespace nca
} // namespace mlpack

#endif
