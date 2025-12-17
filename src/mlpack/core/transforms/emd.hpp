/**
 * @file core/transforms/emd.hpp
 * @author Mohammad Mundiwala
 *
 * Implementation of the EMD feature extractor (empirical mode decomposition).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TRANSFORMS_EMD_HPP
#define MLPACK_CORE_TRANSFORMS_EMD_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/transforms/spline_envelope.hpp>

namespace mlpack
{

template<typename eT>
inline eT LinearExtrapolate(const eT x0,
                            const eT y0,
                            const eT x1,
                            const eT y1,
                            const eT x)
{
  return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
}


template<typename ColType>
inline void FindExtrema(const ColType& h,
                        arma::uvec& maxIdx,
                        arma::uvec& minIdx)
{
  using eT = typename ColType::elem_type;
  const size_t N = h.n_elem;

  maxIdx.reset();
  minIdx.reset();

  if (N == 0)
    return;

  if (N == 1)
  {
    maxIdx = arma::uvec{0};
    minIdx = arma::uvec{0};
    return;
  }

  if (N == 2)
  {
    maxIdx = arma::uvec{0, 1};
    minIdx = arma::uvec{0, 1};
    return;
  }

  std::vector<arma::uword> maxTemp;
  std::vector<arma::uword> minTemp;

  for (size_t i = 1; i + 1 < N; ++i)
  {
    const eT him1 = h[i - 1];
    const eT hi   = h[i];
    const eT hip1 = h[i + 1];

    if ((hi > him1) && (hi > hip1))
      maxTemp.push_back((arma::uword) i);

    if ((hi < him1) && (hi < hip1))
      minTemp.push_back((arma::uword) i);
  }

  // Always include endpoints to allow envelope construction on monotone data.
  if (maxTemp.empty() || maxTemp.front() != 0)
    maxTemp.insert(maxTemp.begin(), arma::uword(0));
  if (maxTemp.back() != (arma::uword) (N - 1))
    maxTemp.push_back((arma::uword) (N - 1));

  if (minTemp.empty() || minTemp.front() != 0)
    minTemp.insert(minTemp.begin(), arma::uword(0));
  if (minTemp.back() != (arma::uword) (N - 1))
    minTemp.push_back((arma::uword) (N - 1));

  maxIdx = arma::conv_to<arma::uvec>::from(maxTemp);
  minIdx = arma::conv_to<arma::uvec>::from(minTemp);
}

template<typename ColType>
inline double L2NormCpuCopy(const ColType& x)
{
  using eT = typename ColType::elem_type;
  const arma::Col<eT> xc(x);
  return arma::norm(xc, 2);
}

template<typename ColType>
inline void SiftingStep(const ColType& h,
                        ColType& hNext)
{
  using eT = typename ColType::elem_type;
  const size_t N = h.n_elem;

  if (N == 0)
  {
    hNext.reset();
    return;
  }

  arma::uvec maxIdx, minIdx;
  ColType upper(N), lower(N);

  FindExtrema(h, maxIdx, minIdx);

  using mlpack::emd::BuildSplineEnvelope;
  BuildSplineEnvelope(h, maxIdx, upper);
  BuildSplineEnvelope(h, minIdx, lower);

  hNext.set_size(N);
  hNext = h - eT(0.5) * (upper + lower);
}

template<typename ColType>
inline void FirstImf(const ColType& signal,
                     ColType& imf,
                     const size_t maxSiftIter = 10,
                     const double tol = 1e-3)
{
  ColType h = signal;
  ColType hNew(signal.n_elem);

  for (size_t iter = 0; iter < maxSiftIter; ++iter)
  {
    SiftingStep(h, hNew);

    // convergence based on relative L2 change (computed on CPU copy)
    const arma::Col<typename ColType::elem_type> diff(hNew - h);
    const double num = arma::norm(diff, 2);
    const double den = L2NormCpuCopy(h);
    const double relChange = (den > 0.0) ? (num / den) : num;

    h.swap(hNew);

    if (relChange < tol)
      break;
  }

  imf = h;
}

} // namespace mlpack

namespace mlpack
{

/**
 * Empirical Mode Decomposition (EMD) on a 1D signal
 *
 * @tparam ColType  Armadillo compatible column vector type.
 * @tparam MatType  Armadillo compatible matrix type.
 * @param signal Input 1D signal (length N).
 * @param imfs Output matrix of extracted IMFs, size N x K.
 * @param residue Output final residue, size N x 1.
 * @param maxImfs Maximum number of IMFs to extract.
 * @param maxSiftIter Maximum number of sifting iterations per IMF.
 * @param tol Stopping tolerance used by the sifting procedure.
 */
template<typename ColType,
         typename MatType,
         typename = std::enable_if_t<
             std::is_same_v<typename ColType::elem_type,
                            typename MatType::elem_type>>>
inline void EMD(const ColType& signal,
                MatType& imfs,
                ColType& residue,
                const size_t maxImfs = 10,
                const size_t maxSiftIter = 10,
                const double tol = 1e-3)
{
  using eT = typename ColType::elem_type;

  const size_t N = signal.n_elem;
  imfs.reset();
  residue = signal;

  if (N == 0)
    return;

  std::vector<ColType> imfList;
  imfList.reserve(maxImfs);

  const double signalNorm = arma::norm(signal, 2);

  for (size_t k = 0; k < maxImfs; ++k)
  {
    // Stop if residue is monotone (fewer than 2 extrema).
    arma::uvec maxIdx, minIdx;
    FindExtrema(residue, maxIdx, minIdx); // make this accept ColType&
    if (maxIdx.n_elem + minIdx.n_elem < 2)
      break;

    ColType imf;
    FirstImf(residue, imf, maxSiftIter, tol); // make this accept ColType&

    const double imfNorm = arma::norm(imf, 2);
    if (imfNorm < std::numeric_limits<double>::epsilon() * signalNorm)
      break;

    imfList.push_back(imf);
    residue -= imf;
  }

  if (!imfList.empty())
  {
    imfs.set_size(N, imfList.size());
    for (size_t k = 0; k < imfList.size(); ++k)
      imfs.col(k) = imfList[k];
  }
  else
  {
    imfs.set_size(N, 0);
  }
}

} // namespace mlpack
#endif // MLPACK_METHODS_EMD_HPP
