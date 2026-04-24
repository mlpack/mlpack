/**
 * @file core/data/mfe.hpp
 * @author Omar Shrit
 *
 * MFE (Mel-Frequency Energy) feature extraction from raw PCM audio data.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_MFE_HPP
#define MLPACK_CORE_DATA_MFE_HPP

#include <mlpack/prereqs.hpp>

/**
* ========================================
*          Theoretical Concept.
* ========================================
*
* General MFCC algorithm concept as follows:
*
* 1. Take an audio signal.
* 
* 2. Cut this signal into a set of overlapped subsignals using a sliding window
* function.
* 
* 3. Multiply (element wise), each subsignals by Hamming window function to
* make the sides very close to zero, prevent spectrum leakage.
*
* 4. Compute the Power Spectrum for each one of these new subsignals.
* 
* 5. Apply MelFilter banks to the each one of the power spectrum. 
* 
* 6. Compte the log for each one of subsignals resulted from MelFilter banks.
*
*  -> So far we have MFE
*
* 7. Compute the FFT of the previous log by applying DCT function. Now we have
* MFCC
*
*  References:
*
*  https://www.youtube.com/watch?v=hF72sY70_IQ
*  https://www.youtube.com/watch?v=SJo7vPgRlBQ&t=223s&pp=ygULTUZDQyBmaWx0ZXI%3D
*/

namespace mlpack {

/**
 * Compute a Hamming window of the given length.
 *
 * The Hamming window is defined as:
 *
 *     w[n] = 0.54 - 0.46 * cos(2 * pi * n / (N - 1))     n = 0, ..., N-1
 *
 * The artificial window cuts the edges of each signal window to be close to
 * zero while keeping the center close to 1.0. This reduces spectral leakage,
 * which means energy that appears into neighbours frequency bin that does
 * not originally exist in time domain.
 *
 * @param len Window length in samples.
 * @return vector contains hamming window.
 */
template<typename eT>
inline arma::Col<eT> HammingWindow(size_t len);

/*
 * Convert Hz frequencies into Mel scale.
 */
template<typename eT>
inline eT HzToMel(eT hz);

/**
 * Build a mel-scaled triangular filterbank matrix.
 *
 * Places (numFilters + 2) points uniformly on the mel scale between lowFreq
 * and highFreq, converts to Hz, maps to FFT bin indices, and constructs
 * overlapping triangular filters.  Adjacent filters overlap at their half-power
 * points so that the entire frequency range is covered without gaps.
 *
 * @return Matrix of that contains the mel filters.
 */
template<typename eT>
inline arma::Mat<eT> MelFilterbank(size_t numFilters,
                                   size_t nFFT,
                                   size_t sampleRate,
                                   double lowFreq,
                                   double highFreq);

/**
 * Extract log-mel filterbank energies (MFE) from one or more audio signals.
 *
 * The input matrix is column-major: each column is a separate audio signal
 * (e.g. loaded from a different file).  Each signal is processed separately
 * through mel filterbanks.
 *
 * @param inputSignal   Matrix where each column is a raw PCM signal.
 * @param mfe           MFE matrix of shape (numMelFilters x totalWindows).
 * @param sampleRate    Sample rate in Hz.
 * @param numMelFilters Number of mel bands.
 * @param windowLength  Window length in milliseconds.
 * @param windowStep    Window hop in milliseconds.
 * @param nFFT          Number of FFT points.
 * @param lowFreq       Low frequency bound in Hz.
 * @param highFreq      High frequency bound in Hz.
 */
template<typename eT>
inline void MFE(const arma::Mat<eT>& inputSignal,
                arma::Mat<eT>& mfe,
                size_t sampleRate,
                size_t numMelFilters = 40,
                double windowLength = 25.0,
                double windowStep = 10.0,
                size_t nFFT = 0,
                double lowFreq = 0.0,
                double highFreq = 0.0,
                const typename std::enable_if_t<
                    std::is_floating_point<eT>::value>* = 0);

/**
 * Another overload to prevent using integral types with MFE.
 * This will throw a compile time error.
 */
template<typename eT>
inline void MFE(const arma::Mat<eT>& inputSignal,
                arma::Mat<eT>& mfe,
                size_t sampleRate,
                size_t numMelFilters = 40,
                double windowLength = 25.0,
                double windowStep = 10.0,
                size_t nFFT = 0,
                double lowFreq = 0.0,
                double highFreq = 0.0,
                const typename std::enable_if_t<
                    !std::is_floating_point<eT>::value>* = 0);

/**
 * It would be faster for FFT to use a window size that is pair and dividable
 * by 2. Since the use might input a different windows size from 25 ms. This
 * will produce a different samples sizes with different sampling frequencies,
 * and we might ends up with arbitrary non easily dividable window size.
 *
 * Therefore the objective of this function is to multiply by 2 using bitwise
 * operation as long as we are < window size.
 *
 * @param n window size, number of samples generated from a specific window.
 */
inline size_t NextPowerOf2(size_t n);

/**
 * Compute the first part power spectrum of each window using arma::fft().
 *
 * Each column of `windows` is zero-padded to `nFFT` length, transformed via
 * arma::fft(), and the energy represented by squared magnitude of the first bins
 * is kept.
 */
template<typename eT>
inline void PowerSpectrum(const arma::Mat<eT>& windows, arma::Mat<eT>& power,
    size_t nFFT);

/**
 * Split the signal into a set of overlapping windows.
 *
 * The input signal is sliced into numWindows columns of length windowLength,
 * each offset by windowStep samples from the previous.  If the signal is
 * shorter than one window it is zero-padded.
 *
 * The signal is divided into a set of windows with identical lengths. These
 * windows have an overlap that is defined by a step. Zero padding is
 * applied in the case when the window is size is less than the 
 * signal.
 *
 * All of these windows are conglomerated into one windows matrix
 *
 * @param signal Input audio samples as a column vectors
 * @param windows Output audio signal converted into a set of sliding windows.
 * @param windowLength Number of samples per window.
 * @param windowStep Number of samples between the start of consecutive windows.
 */
template<typename MatType, typename eT>
inline void SlidingWindow(const MatType& signal,
                          arma::Mat<eT>& windows,
                          size_t windowLength,
                          size_t windowStep);

} // namespace mlpack

#include "mfe_impl.hpp"

#endif
