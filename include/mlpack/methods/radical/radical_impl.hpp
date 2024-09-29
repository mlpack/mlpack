/**
 * @file methods/radical/radical_impl.hpp
 * @author Nishant Mehta
 *
 * Implementation of Radical class
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RADICAL_RADICAL_IMPL_HPP
#define MLPACK_METHODS_RADICAL_RADICAL_IMPL_HPP

#include "radical.hpp"

namespace mlpack {

// Set the parameters to RADICAL.
inline Radical::Radical(
    const double noiseStdDev,
    const size_t replicates,
    const size_t angles,
    const size_t sweeps,
    const size_t m) :
    noiseStdDev(noiseStdDev),
    replicates(replicates),
    angles(angles),
    sweeps(sweeps),
    m(m)
{
  // Nothing to do here.
}

inline void Radical::CopyAndPerturb(arma::mat& xNew,
                                    const arma::mat& x) const
{
  xNew = repmat(x, replicates, 1) + noiseStdDev * arma::randn(
      replicates * x.n_rows, x.n_cols);
}


inline double Radical::Vasicek(arma::vec& z) const
{
  z = sort(z);

  // Apparently slower.
  /*
  vec logs = log(z.subvec(m, z.n_elem - 1) - z.subvec(0, z.n_elem - 1 - m));
  //vec val = sum(log(z.subvec(m, nPoints - 1) - z.subvec(0, nPoints - 1 - m)));
  return (double) sum(logs);
  */

  // Apparently faster.
  double sum = 0;
  arma::uword range = z.n_elem - m;
  for (arma::uword i = 0; i < range; ++i)
  {
    sum += std::log(std::max(z(i + m) - z(i), DBL_MIN));
  }

  return sum;
}


inline double Radical::DoRadical2D(const arma::mat& matX, 
                                   util::Timers& timers)
{
  timers.Start("radical_copy_and_perturb");
  CopyAndPerturb(perturbed, matX);
  timers.Stop("radical_copy_and_perturb");

  arma::mat::fixed<2, 2> matJacobi;

  arma::vec values(angles);

  for (size_t i = 0; i < angles; ++i)
  {
    const double theta = (i / (double) angles) * M_PI / 2.0;
    const double cosTheta = cos(theta);
    const double sinTheta = sin(theta);

    matJacobi(0, 0) = cosTheta;
    matJacobi(1, 0) = -sinTheta;
    matJacobi(0, 1) = sinTheta;
    matJacobi(1, 1) = cosTheta;

    candidate = perturbed * matJacobi;
    arma::vec candidateY1 = candidate.unsafe_col(0);
    arma::vec candidateY2 = candidate.unsafe_col(1);

    values(i) = Vasicek(candidateY1) + Vasicek(candidateY2);
  }

  arma::uword indOpt = 0;
  values.min(indOpt); // we ignore the return value; we don't care about it
  return (indOpt / (double) angles) * M_PI / 2.0;
}


inline void Radical::DoRadical(const arma::mat& matXT,
                               arma::mat& matY,
                               arma::mat& matW,
                               util::Timers& timers)
{
  // matX is nPoints by nDims (although less intuitive than columns being
  // points, and although this is the transpose of the ICA literature, this
  // choice is for computational efficiency when repeatedly generating
  // two-dimensional coordinate projections for Radical2D).
  timers.Start("radical_transpose_data");
  arma::mat matX = trans(matXT);
  timers.Stop("radical_transpose_data");

  // If m was not specified, initialize m as recommended in
  // (Learned-Miller and Fisher, 2003).
  if (m < 1)
    m = floor(std::sqrt((double) matX.n_rows));

  const size_t nDims = matX.n_cols;
  const size_t nPoints = matX.n_rows;

  timers.Start("radical_whiten_data");
  arma::mat matXWhitened;
  arma::mat matWhitening;
  WhitenFeatureMajorMatrix(matX, matY, matWhitening);
  timers.Stop("radical_whiten_data");
  // matY is now the whitened form of matX.

  // In the RADICAL code, they do not copy and perturb initially, although the
  // paper does.  We follow the code as it should match their reported results
  // and likely does a better job bouncing out of local optima.
  // GeneratePerturbedX(X, X);

  // Initialize the unmixing matrix to the whitening matrix.
  timers.Start("radical_do_radical");
  matW = matWhitening;

  arma::mat matYSubspace(nPoints, 2);

  arma::mat matJ = arma::eye(nDims, nDims);

  for (size_t sweepNum = 0; sweepNum < sweeps; sweepNum++)
  {
    Log::Info << "RADICAL: sweep " << sweepNum << "." << std::endl;

    for (size_t i = 0; i < nDims - 1; ++i)
    {
      for (size_t j = i + 1; j < nDims; ++j)
      {
        Log::Debug << "RADICAL 2D on dimensions " << i << " and " << j << "."
            << std::endl;

        matYSubspace.col(0) = matY.col(i);
        matYSubspace.col(1) = matY.col(j);

        const double thetaOpt = DoRadical2D(matYSubspace, timers);

        const double cosThetaOpt = cos(thetaOpt);
        const double sinThetaOpt = sin(thetaOpt);

        // Set elements of transformation matrix.
        matJ(i, i) = cosThetaOpt;
        matJ(j, i) = -sinThetaOpt;
        matJ(i, j) = sinThetaOpt;
        matJ(j, j) = cosThetaOpt;

        matY *= matJ;

        // Unset elements of transformation matrix, so matJ = eye() again.
        matJ(i, i) = 1;
        matJ(j, i) = 0;
        matJ(i, j) = 0;
        matJ(j, j) = 1;
      }
    }
  }
  timers.Stop("radical_do_radical");

  // The final transposes provide W and Y in the typical form from the ICA
  // literature.
  timers.Start("radical_transpose_data");
  matW = trans(matW);
  matY = trans(matY);
  timers.Stop("radical_transpose_data");
}

inline void WhitenFeatureMajorMatrix(const arma::mat& matX,
                                     arma::mat& matXWhitened,
                                     arma::mat& matWhitening)
{
  arma::mat matU, matV;
  arma::vec s;
  arma::svd(matU, s, matV, cov(matX));
  matWhitening = matU * diagmat(1 / sqrt(s)) * trans(matV);
  matXWhitened = matX * matWhitening;
}

} // namespace mlpack

#endif
