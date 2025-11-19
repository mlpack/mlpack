#ifndef MLPACK_METHODS_EMD_EMD_HPP
#define MLPACK_METHODS_EMD_EMD_HPP

#include <mlpack/prereqs.hpp>
#include <vector>

#include <mlpack/methods/emd/emd_first_imf.hpp>

namespace mlpack {
namespace emd {

template<typename eT>
inline void Emd(const arma::Col<eT>& signal,
                arma::Mat<eT>& imfs,
                arma::Col<eT>& residue,
                const size_t maxImfs = 10,
                const size_t maxSiftIter = 10,
                const double tol = 1e-3)
{
  const arma::uword N = signal.n_elem;
  imfs.reset();
  residue = signal;

  if (N == 0)
    return;

  std::vector< arma::Col<eT> > imfList;
  imfList.reserve(maxImfs);

  const double signalNorm = arma::norm(signal, 2);

  for (size_t k = 0; k < maxImfs; ++k)
  {
    // stop if residue once monotone
    arma::uvec maxIdx, minIdx;
    FindExtrema(residue, maxIdx, minIdx);
    if (maxIdx.n_elem + minIdx.n_elem < 2)
      break;

    arma::Col<eT> imf;
    FirstImf(residue, imf, maxSiftIter, tol);

    // stop when the IMF is negligible
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

} // namespace emd
} // namespace mlpack

#endif
