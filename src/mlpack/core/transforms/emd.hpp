/**
 * @file core/transforms/emd.hpp
 * @author Mohammad Mundiwala
 *
 * Implementation of the EMD feature extractor (empirical mode decomposition)
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

namespace mlpack {

template<typename ColType>
inline void FindExtrema(const ColType& h,
                        arma::uvec& maxIdx,
                        arma::uvec& minIdx)
{
  // Identify indices of strict local maxima and minima (discrete neighbors) and
  // always include endpoints. This determines whether the residue is monotone
  // and supplies knots for spline envelopes in sifting.
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

  arma::uvec maxIdx = find(
    h.subvec(0, h.n_elem - 3) < h.subvec(1, h.n_elem - 2) %
    h.subvec(1, h.n_elem - 2) > h.subvec(2, h.n_elem - 1));
  arma::uvec minIdx = find(
    h.subvec(0, h.n_elem - 3) > h.subvec(1, h.n_elem - 2) %
    h.subvec(1, h.n_elem - 2) < h.subvec(2, h.n_elem - 1));

  // Always include endpoints to allow envelope construction on monotone data.
  if (minIdx.empty() || minIdx.front() != 0)
    minIdx.insert(minIdx.begin(), arma::uword(0));
  if (minIdx.back() != (arma::uword) (N - 1))
    minIdx.push_back((arma::uword) (N - 1));
}

//helper to count zero crossings as part of IMF stopping criteria
template<typename ColType>
inline size_t CountZeroCrossings(const ColType& h)
{
  using eT = typename ColType::elem_type;
  const size_t N = h.n_elem;
  if (N < 2) return 0;

  auto sgn = [](eT v) -> int { return (v > eT(0)) - (v < eT(0)); };

  int prev = 0;
  size_t zc = 0;

  for (size_t i = 0; i < N; ++i)
  {
    const int cur = sgn(h[i]);
    if (cur == 0) continue;
    if (prev != 0 && cur != prev) ++zc;
    prev = cur;
  }
  return zc;
}

//faster norm computation for arma types
template<typename ColType>
inline double L2Norm(const ColType& x)
{
  if constexpr (arma::is_arma_type<ColType>::value)
    return arma::norm(x, 2);
  else
    return L2NormCpuCopy(x);
}

template<typename ColType>
inline double L2NormCpuCopy(const ColType& x)
{
  // Compute L2 norm on arma::Col copy for portability across types
  // would not be needed if data is always loaded into arma types.
  using eT = typename ColType::elem_type;
  const arma::Col<eT> xc(x);
  return arma::norm(xc, 2);
}

// sifting step extracts mean envelope and produces next h
template<typename ColType>
inline void SiftingStep(ColType& h,
                        ColType& meanEnvOut,
                        size_t& numExtremaOut)
{
  using eT = typename ColType::elem_type;
  const size_t N = h.n_elem;

  if (N == 0) { meanEnvOut.reset(); numExtremaOut = 0; return; }

  arma::uvec maxIdx, minIdx;
  ColType upper(N), lower(N);

  FindExtrema(h, maxIdx, minIdx);

  using mlpack::BuildSplineEnvelope;
  BuildSplineEnvelope(h, maxIdx, upper);
  BuildSplineEnvelope(h, minIdx, lower);

  meanEnvOut = (upper + lower)/2;

  h -= meanEnvOut; //in-place update to next h
  
  // count the number of extrema 
  numExtremaOut = 
    (maxIdx.n_elem >= 2 ? maxIdx.n_elem - 2 : 0) +
    (minIdx.n_elem >= 2 ? minIdx.n_elem - 2 : 0);
}

//extract first IMF via sifting, using EMD stopping criteria

template<typename ColType>
inline bool FirstImf(const ColType& signal,
                     ColType& imf,
                     const size_t maxSiftIter = 50,
                     const double tolMean = 1e-3)
{
  using eT = typename ColType::elem_type;

  ColType h = signal;
  ColType hNew(signal.n_elem);
  ColType meanEnv(signal.n_elem);

  for (size_t iter = 0; iter < maxSiftIter; ++iter)
  {
    size_t ext = 0;
    SiftingStep(h, meanEnv, ext);

    // If the current signal is monotone, EMD is NA
    // ext = # extrema points
    if (iter == 0 && ext <= 1)
      return false;
    // mean-envelope criterion
    // mean envelope should be close to zero
    const double meanRatio = (L2Norm(h) > 0.0) ? (L2Norm(meanEnv) / L2Norm(h))
                                              : L2Norm(meanEnv);

    // IMF criterion: extrema vs zero-crossings
    // number of extrema and zero-crossings must differ at most by one
    const size_t zc  = CountZeroCrossings(hNew);
    const bool imfShapeOk = (std::max(ext, zc) - std::min(ext, zc) <= 1);

    h.swap(hNew);

    if (imfShapeOk && (meanRatio < tolMean))
      break;
  }

  imf = h;
  return true;
}

} // namespace mlpack

namespace mlpack
{

/**
 * Empirical Mode Decomposition on a 1D signal.
 *
 * Algorithm outline: repeatedly extract IMFs via sifting (cubic spline
 * envelopes of extrema) until the residue is monotone or a limit is reached.
 * Supports any Armadillo-compatible column/matrix types with matching element
 * types (enforced via SFINAE).
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
    ColType imf;
    if (!FirstImf(residue, imf, maxSiftIter, tol)) // Get next IMF via sifting
      break;

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
#endif // MLPACK_CORE_TRANSFORMS_EMD_HPP
