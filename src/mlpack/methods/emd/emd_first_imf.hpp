// emd_first_imf.hpp
#ifndef MLPACK_METHODS_EMD_FIRST_IMF_HPP
#define MLPACK_METHODS_EMD_FIRST_IMF_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/emd/spline_envelope.hpp>

namespace mlpack {
namespace emd {

template<typename eT>
inline eT LinearExtrapolate(const eT x0,
                            const eT y0,
                            const eT x1,
                            const eT y1,
                            const eT x)
{
  // Assume caller ensures x1 != x0.
  return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
}

template<typename eT>
inline void FindExtrema(const arma::Col<eT>& h,
                        arma::uvec& maxIdx,
                        arma::uvec& minIdx)
{
  const arma::uword N = h.n_elem;
  maxIdx.reset();
  minIdx.reset();

  if (N < 3)
    return;

  std::vector<arma::uword> maxTemp;
  std::vector<arma::uword> minTemp;

  for (arma::uword i = 1; i < N - 1; ++i)
  {
    if ((h[i] > h[i - 1]) && (h[i] > h[i + 1]))
      maxTemp.push_back(i);

    if ((h[i] < h[i - 1]) && (h[i] < h[i + 1]))
      minTemp.push_back(i);
  }

  // end points are extrema (simplified from libeemd for now)
  if (!maxTemp.empty())
  {
    if (maxTemp.front() != 0)
      maxTemp.insert(maxTemp.begin(), arma::uword(0));
    if (maxTemp.back() != N - 1)
      maxTemp.push_back(N - 1);
  }

  if (!minTemp.empty())
  {
    if (minTemp.front() != 0)
      minTemp.insert(minTemp.begin(), arma::uword(0));
    if (minTemp.back() != N - 1)
      minTemp.push_back(N - 1);
  }

  maxIdx = arma::conv_to<arma::uvec>::from(maxTemp);
  minIdx = arma::conv_to<arma::uvec>::from(minTemp);
}

// One sifting step: hNext = h - 0.5*(upper + lower).
template<typename eT>
inline void SiftingStep(const arma::Col<eT>& h,
                        arma::Col<eT>& hNext)
{
  const arma::uword N = h.n_elem;
  if (N == 0)
  {
    hNext.reset();
    return;
  }

  arma::uvec maxIdx, minIdx;
  arma::Col<eT> upper(N), lower(N);

  FindExtrema(h, maxIdx, minIdx);
  BuildSplineEnvelope(h, maxIdx, upper); //cubic spline 
  BuildSplineEnvelope(h, minIdx, lower); // implementation seperate
  hNext.set_size(N);
  for (arma::uword i = 0; i < N; ++i)
  {
    const eT m = eT(0.5) * (upper[i] + lower[i]);
    hNext[i] = h[i] - m;
  }
}
// no enforcment of # zero crossings  = # extrema
// and local mean = 0  
// more rigorous conditons to be added once this works. 

// repeat sifting
template<typename eT>
inline void FirstImf(const arma::Col<eT>& signal,
                     arma::Col<eT>& imf,
                     const size_t maxSiftIter = 10,
                     const double tol = 1e-3)
{
  arma::Col<eT> h = signal;
  arma::Col<eT> hNew(signal.n_elem);

  for (size_t iter = 0; iter < maxSiftIter; ++iter)
  {
    SiftingStep(h, hNew);

    // convergence based on relative L2 change.
    const double num = arma::norm(hNew - h, 2);
    const double den = arma::norm(h, 2);
    const double relChange = (den > 0.0) ? (num / den) : num;

    h.swap(hNew);

    if (relChange < tol)
      break;
  }

  imf = h;
}

} // namespace emd
} // namespace mlpack

#endif
