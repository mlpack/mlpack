/**
 * @file boostmetric_impl.hpp
 * @author Manish Kumar
 *
 * Implementation of BoostMetric class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_BOOSTMETRIC_BOOSTMETRIC_IMPL_HPP
#define MLPACK_METHODS_BOOSTMETRIC_BOOSTMETRIC_IMPL_HPP

// In case it was not already included.
#include "boostmetric.hpp"

namespace mlpack {
namespace boostmetric {

/**
 * Takes in a reference to the dataset. Copies the data, initializes
 * all of the member variables and constraint object.
 */
template<typename MetricType>
BOOSTMETRIC<MetricType>::BOOSTMETRIC(const arma::mat& dataset,
                                     const arma::Row<size_t>& labels,
                                     const size_t k) :
    dataset(dataset),
    labels(labels),
    k(k),
    maxIter(500),
    tolerance(1e-7),
    wTolerance(1e-5),
    constraint(dataset, labels, k)
{ /* nothing to do */ }

// Calculate inner product of two matrices.
inline double innerProduct(arma::mat& Ar, arma::mat& Z)
{
  double sum = 0.0;

  for (size_t i = 0; i < Z.n_elem; i++)
    sum += Ar(i) * Z(i);

  return sum;
}

template<typename MetricType>
void BOOSTMETRIC<MetricType>::LearnDistance(arma::mat& outputMatrix)
{
  Timer::Start("boostmetric");

  // See if we were passed an initialized matrix.
  if ((outputMatrix.n_rows != dataset.n_rows) ||
      (outputMatrix.n_cols != dataset.n_rows))
  {
    outputMatrix.eye(dataset.n_rows, dataset.n_rows);
  }

  // Compute triplets.
  arma::Mat<size_t> triplets;
  constraint.Triplets(triplets, dataset, labels);

  size_t N = triplets.n_cols;
  size_t dim = dataset.n_rows;

  // Initialize u.
  arma::vec u;
  u.ones(N);

  // Normalize u such that sum(u_i) = 1
  u = u / N;

  arma::cube A(dim, dim, N);

  // Initialize A (coefficient matrix of difference between distance
  // of impostors & target neighbors).
  for (size_t i = 0; i < N; i++)
  {
    A.slice(i) = (dataset.col(triplets(0, i)) - dataset.col(triplets(2, i))) *
        arma::trans(dataset.col(triplets(0, i)) - dataset.col(triplets(2, i)))
        -
        (dataset.col(triplets(0, i)) - dataset.col(triplets(1, i))) *
        arma::trans(dataset.col(triplets(0, i)) - dataset.col(triplets(1, i)));
  }

  // Declare weighted sum of Ar matrices.
  arma::mat Acap;

  // Now iterate!
  for (size_t i = 0; i < maxIter; i++)
  {
    // Initialize Acap.
    Acap.zeros(dim, dim);
    for (size_t j = 0; j < N; j++)
      Acap += u(j) * A.slice(j);

    arma::vec eigval;
    arma::mat eigvec;
    arma::eig_sym(eigval, eigvec, (Acap + trans(Acap) / 2));

    // Get maximum eigvalue.
    double maxEig = arma::max(eigval);

    if (maxEig < tolerance)
    {
      Log::Info << "BoostMetric: minimized within tolerance " << tolerance
            << "; " << "terminating optimization." << std::endl;
      break;
    }

    // Output current maximum eigen value.
    Log::Info << "BoostMetric: iteration " << i << ", maximum eigen value "
        <<  maxEig << "." << std::endl;

    // Get index of maximum element.
    arma::uvec maxIndex = arma::find(eigval == maxEig);

    arma::mat Z = eigvec.col(maxIndex(0)) * arma::trans(eigvec.col(maxIndex(0)));

    // Initialize H vector by computing inner product. Hr = <Ar, Z>
    arma::vec H(N);
    for (size_t j = 0; j < N; j++)
      H(j) = innerProduct(A.slice(j), Z);

    // Binary Search for weight calculation.
    double w;
    while (true)
    {
      // Calculate mid value.
      w = (wHigh + wLow) * 0.5;

      double lhs = 0;
      for (size_t j = 0; j < N; j++)
        lhs += (H(j) - tolerance) * std::exp(-w * H(j)) * u(j);

      // Update search direction.
      if (lhs > 0)
        wLow = w;
      else
        wHigh = w;

      // Terminate search.
      if (wHigh - wLow < wTolerance || std::abs(lhs) < wTolerance)
        break;
    }

    // Update u.
    for (size_t j = 0; j < N; j++)
      u(j) = u(j) * std::exp(-H(j) * w);

    // Normalize u so that sum(u_i) = 1.
    u /= arma::sum(u);

    // Update p.s.d matrix.
    outputMatrix += w * Z;
  }

  // Generate distance from p.s.d matrix.
  arma::vec eigval;
  arma::mat eigvec;

  // Generate eigen value & corresponding eigen vectors.
  arma::eig_sym(eigval, eigvec, (outputMatrix + arma::trans(outputMatrix)) / 2);

  // Discard negative eigen values, if any.
  eigval.transform( [](double val) { return (val > 0) ? std::sqrt(val) : double(0); } );

  outputMatrix = arma::trans(eigvec * diagmat(eigval));

  Timer::Stop("boostmetric");
}


} // namespace boostmetric
} // namespace mlpack

#endif
