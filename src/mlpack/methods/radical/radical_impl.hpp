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

template<typename MatType>
inline void Radical::CopyAndPerturb(MatType& xNew,
                                    const MatType& x) const
{
  xNew = repmat(x, replicates, 1) + noiseStdDev * arma::randn<MatType>(
      replicates * x.n_rows, x.n_cols);
}

template<typename VecType>
inline typename VecType::elem_type Radical::Vasicek(
    VecType& z,
    const size_t m) const
{
  using ElemType = typename VecType::elem_type;

  z = sort(z);

  // Apparently slower.
  /*
  vec logs = log(z.subvec(m, z.n_elem - 1) - z.subvec(0, z.n_elem - 1 - m));
  //vec val = sum(log(z.subvec(m, nPoints - 1) - z.subvec(0, nPoints - 1 - m)));
  return (double) sum(logs);
  */

  // Apparently faster.
  ElemType sum = 0;
  arma::uword range = z.n_elem - m;
  for (arma::uword i = 0; i < range; ++i)
  {
    sum += std::log(std::max(z(i + m) - z(i),
                             std::numeric_limits<ElemType>::min()));
  }

  return sum;
}

template<typename MatType>
inline typename MatType::elem_type Radical::Apply2D(const MatType& matX,
                                                    const size_t m,
                                                    MatType& perturbed,
                                                    MatType& candidate,
                                                    util::Timers& timers)
{
  using VecType = typename GetColType<MatType>::type;
  using ElemType = typename MatType::elem_type;

  timers.Start("radical_copy_and_perturb");
  CopyAndPerturb(perturbed, matX);
  timers.Stop("radical_copy_and_perturb");

  typename MatType::template fixed<2, 2> matJacobi;
  VecType values(angles);

  for (size_t i = 0; i < angles; ++i)
  {
    const ElemType theta = (i / (ElemType) angles) * M_PI / 2.0;
    const ElemType cosTheta = cos(theta);
    const ElemType sinTheta = sin(theta);

    matJacobi(0, 0) = cosTheta;
    matJacobi(1, 0) = -sinTheta;
    matJacobi(0, 1) = sinTheta;
    matJacobi(1, 1) = cosTheta;

    candidate = perturbed * matJacobi;
    VecType candidateY1 = candidate.unsafe_col(0);
    VecType candidateY2 = candidate.unsafe_col(1);

    values(i) = Vasicek(candidateY1, m) + Vasicek(candidateY2, m);
  }

  arma::uword indOpt = values.index_min();
  return (indOpt / (ElemType) angles) * M_PI / 2.0;
}

template<typename MatType>
inline void Radical::DoRadical(const MatType& matXT,
                               MatType& matY,
                               MatType& matW,
                               util::Timers& timers)
{
  Apply(matXT, matY, matW, timers);
}

template<typename MatType>
inline void Radical::Apply(const MatType& matXT,
                           MatType& matY,
                           MatType& matW,
                           util::Timers& timers)
{
  using ElemType = typename MatType::elem_type;

  // matX is nPoints by nDims (although less intuitive than columns being
  // points, and although this is the transpose of the ICA literature, this
  // choice is for computational efficiency when repeatedly generating
  // two-dimensional coordinate projections for Radical2D).
  timers.Start("radical_transpose_data");
  MatType matX = trans(matXT);
  timers.Stop("radical_transpose_data");

  // If m was not specified, initialize m as recommended in
  // (Learned-Miller and Fisher, 2003).
  size_t localM = m;
  if (localM < 1)
    localM = floor(std::sqrt((ElemType) matX.n_rows));

  // If the number of sweeps was not specified, perform one for each dimension.
  size_t localSweeps = sweeps;
  if (localSweeps < 1)
    localSweeps = matX.n_rows - 1;

  const size_t nDims = matX.n_cols;
  const size_t nPoints = matX.n_rows;

  timers.Start("radical_whiten_data");
  MatType matXWhitened;
  MatType matWhitening;
  WhitenFeatureMajorMatrix(matX, matY, matWhitening);
  timers.Stop("radical_whiten_data");
  // matY is now the whitened form of matX.

  // These two matrices will be used repeatedly by Radical2D().  We create them
  // here to avoid repeated allocations.
  MatType perturbed, candidate;

  // In the RADICAL code, they do not copy and perturb initially, although the
  // paper does.  We follow the code as it should match their reported results
  // and likely does a better job bouncing out of local optima.
  // GeneratePerturbedX(X, X);

  // Initialize the unmixing matrix to the whitening matrix.
  timers.Start("radical_do_radical");
  matW = matWhitening;

  MatType matYSubspace(nPoints, 2);

  MatType matJ = arma::eye<MatType>(nDims, nDims);

  for (size_t sweepNum = 0; sweepNum < localSweeps; sweepNum++)
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

        const ElemType thetaOpt = Apply2D(matYSubspace, localM, perturbed,
            candidate, timers);

        const ElemType cosThetaOpt = cos(thetaOpt);
        const ElemType sinThetaOpt = sin(thetaOpt);

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

template<typename Archive>
void Radical::serialize(Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(noiseStdDev));
  ar(CEREAL_NVP(replicates));
  ar(CEREAL_NVP(angles));
  ar(CEREAL_NVP(sweeps));
  ar(CEREAL_NVP(m));
}

template<typename MatType>
inline void WhitenFeatureMajorMatrix(const MatType& matX,
                                     MatType& matXWhitened,
                                     MatType& matWhitening)
{
  using VecType = typename GetColType<MatType>::type;

  MatType matU, matV;
  VecType s;
  arma::svd(matU, s, matV, cov(matX));
  matWhitening = matU * diagmat(1 / sqrt(s)) * trans(matV);
  matXWhitened = matX * matWhitening;
}

} // namespace mlpack

#endif
