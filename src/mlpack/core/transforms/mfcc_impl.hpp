/**
 * @file core/data/mfcc_impl.hpp
 * @author Omar Shrit
 *
 * Implementation file for MFCC (Mel-Frequency Cepstral Coefficients) filters.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_MFCC_IMPL_HPP
#define MLPACK_CORE_DATA_MFCC_IMPL_HPP

#include "mfcc.hpp"

namespace mlpack {

template<typename eT>
inline arma::Mat<eT> DCTMatrix(size_t numCoeffs, size_t numFilters)
{
  // n = [0, 1, ..., numCoeffs-1] as a column.
  arma::Col<eT> n = arma::linspace<arma::Col<eT>>(0, numCoeffs - 1,
      numCoeffs);
  // k + 0.5 = [0.5, 1.5, ..., numFilters-0.5] as a row.
  arma::Row<eT> kHalf = arma::linspace<arma::Row<eT>>(0.5, numFilters - 0.5,
      numFilters);

  return arma::cos((M_PI / numFilters) * n * kHalf);
}

template<typename eT>
inline void MFCC(const arma::Mat<eT>& inputSignal,
                 arma::Mat<eT>& mfcc,
                 size_t sampleRate,
                 size_t numCoeffs,
                 size_t numMelFilters,
                 double windowLength,
                 double windowStep,
                 size_t nFFT,
                 double lowFreq,
                 double highFreq,
                 const typename std::enable_if_t<
                    std::is_floating_point<eT>::value>*)
{
  arma::Mat<eT> mfe;
  MFE(inputSignal, mfe, sampleRate, numMelFilters, windowLength, windowStep,
      nFFT, lowFreq, highFreq);

  mfcc = DCTMatrix<eT>(numCoeffs, numMelFilters) * mfe;
}

template<typename eT>
inline void MFCC(const arma::Mat<eT>& inputSignal,
                 arma::Mat<eT>& mfcc,
                 size_t sampleRate,
                 size_t numCoeffs,
                 size_t numMelFilters,
                 double windowLength,
                 double windowStep,
                 size_t nFFT,
                 double lowFreq,
                 double highFreq,
                 const typename std::enable_if_t<
                    !std::is_floating_point<eT>::value>*)
{
  static_assert(std::is_floating_point<eT>::value,
      "MFCC(): input matrix must have a floating-point element type "
      "(float or double). Integer matrices are not supported because "
      "FFT, log, and other operations require floating-point "
      "arithmetic.");
}

} // namespace mlpack

#endif
