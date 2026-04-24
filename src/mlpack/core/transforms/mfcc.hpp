/**
 * @file core/data/mfcc.hpp
 * @author Omar Shrit
 *
 * MFCC (Mel-Frequency Cepstral Coefficients) feature extraction from raw PCM
 * audio data.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_MFCC_HPP
#define MLPACK_CORE_DATA_MFCC_HPP

#include <mlpack/prereqs.hpp>

#include "mfe.hpp"

namespace mlpack {

/**
 * Build a DCT-II matrix of shape (numCoeffs x numFilters).
 *
 *     D[n][k] = cos(π · n · (k + 0.5) / M)
 *
 * Multiplying this matrix by a column of log-mel energies yields the
 * cepstral coefficients.  Constructed using vectorised outer-product-style
 * operations.
 */
template<typename eT>
inline arma::Mat<eT> DCTMatrix(size_t numCoeffs, size_t numFilters);

/**
 * Extract Mel-Frequency Cepstral Coefficients (MFCC) from one or more audio
 * signals.
 *
 * The input matrix is column-major: each column is a separate audio signal.
 * MFCC first computes the MFE and then applies a DCT matrix to
 * decorrelate the log-mel energies and retain only the first numCoeffs
 * coefficients.
 *
 * When the input has a single column, the output has shape
 * (numCoeffs x numWindows).  When the input has multiple columns, the
 * frames from all signals are concatenated horizontally into a single output
 * matrix of shape (numCoeffs x totalWindows).
 *
 * numMelFilters must be >= numCoeffs; the DCT compresses numMelFilters
 * log-mel energies down to numCoeffs cepstral coefficients.
 *
 * @param inputSignal   Matrix where each column is a raw PCM signal.
 * @param mfcc          Result matrix of shape (numCoeffs x totalWindows).
 * @param sampleRate    Sample rate in Hz.
 * @param numCoeffs     Number of cepstral coefficients.
 * @param numMelFilters Number of mel bands (must be > than numCoeffs).
 * @param windowLength  Window length in milliseconds.
 * @param windowStep    Window hop in milliseconds.
 * @param nFFT          Number of FFT points.
 * @param lowFreq       Low frequency bound in Hz.
 * @param highFreq      High frequency bound in Hz.
 */
template<typename eT>
inline void MFCC(const arma::Mat<eT>& inputSignal,
                 arma::Mat<eT>& mfcc,
                 size_t sampleRate,
                 size_t numCoeffs = 13,
                 size_t numMelFilters = 40,
                 double windowLength = 25.0,
                 double windowStep = 10.0,
                 size_t nFFT = 0,
                 double lowFreq = 0.0,
                 double highFreq = 0.0,
                 const typename std::enable_if_t<
                    std::is_floating_point<eT>::value>* = 0);

/**
 * Another overload to prevent using integral types with MFCC.
 * This will throw a compile time error.
 */
template<typename eT>
inline void MFCC(const arma::Mat<eT>& inputSignal,
                 arma::Mat<eT>& mfcc,
                 size_t sampleRate,
                 size_t numCoeffs = 13,
                 size_t numMelFilters = 40,
                 double windowLength = 25.0,
                 double windowStep = 10.0,
                 size_t nFFT = 0,
                 double lowFreq = 0.0,
                 double highFreq = 0.0,
                 const typename std::enable_if_t<
                    !std::is_floating_point<eT>::value>* = 0);

} // namespace mlpack

#include "mfcc_impl.hpp"

#endif
