/**
 * @file core/data/mfcc.hpp
 * @author Omar Shrit
 *
 * MFE (Mel-Frequency Energy) and MFCC (Mel-Frequency Cepstral Coefficients)
 * feature extraction from raw PCM audio data.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_MFCC_HPP
#define MLPACK_CORE_DATA_MFCC_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

template<typename eT>
inline eT HzToMel(eT hz)
{
  return static_cast<eT>(2595.0) * std::log10(static_cast<eT>(1.0) +
      hz / static_cast<eT>(700.0));
}

template<typename eT>
inline eT MelToHz(eT mel)
{
  return static_cast<eT>(700.0) * (std::pow(static_cast<eT>(10.0),
      mel / static_cast<eT>(2595.0)) - static_cast<eT>(1.0));
}

inline size_t NextPowerOf2(size_t n)
{
  size_t p = 1;
  while (p < n)
    p <<= 1;
  return p;
}

/**
 * Apply a Finit Impulse Response filter (aka, pre-emphasis filter) to a
 * single signal column.
 *
 *     y[n] = x[n] - coeff * x[n-1]
 *
 * The filter boosts high frequencies to compensate for the natural spectral.
 *
 * @param signal arma column signal, usually one audio file.
 * @param coeff Set to 0 to disable.
 */
template<typename eT>
inline void FinitImpulseResponseFilter(arma::Col<eT>& signal, float coeff)
{
  if (signal.n_elem < 2 || coeff == 0.0f)
      return;

  for (size_t i = 1; i < signal.n_elem; ++i)
    signal[i] = signal[i] - (coeff) * signal[i - 1];
}

/**
 * Split a single-column signal into overlapping windows.
 *
 * The input signal is sliced into numWindows columns of length frameLength,
 * each offset by frameStep samples from the previous.  If the signal is
 * shorter than one frame it is zero-padded.
 *
 * @return Matrix of shape (frameLength x numFrames), one frame per column.
 */
template<typename eT>
inline arma::Mat<eT> Framize(const arma::Col<eT>& signal,
                             size_t frameLength,
                             size_t frameStep)
{
  if (signal.n_elem < frameLength)
  {
    arma::Mat<eT> frames(frameLength, 1, arma::fill::zeros);
    frames.col(0).subvec(0, signal.n_elem - 1) = signal;
    return frames;
  }

  const size_t numFrames = (signal.n_elem - frameLength) / frameStep + 1;
  arma::Mat<eT> frames(frameLength, numFrames);

  for (size_t i = 0; i < numFrames; ++i)
  {
    const size_t start = i * frameStep;
    frames.col(i) = signal.subvec(start, start + frameLength - 1);
  }

  return frames;
}

/**
 * Split a single-column signal into overlapping segments using a sliding
 * window.
 *
 * The input signal is sliced into numWindows columns of length windowLength,
 * each offset by hopSize samples from the previous.  If the signal is shorter
 * than one window it is zero-padded.
 *
 * @param signal       Input audio samples as a column vector.
 * @param windowLength Number of samples per window.
 * @param hopSize      Number of samples between the start of consecutive
 *                     windows.
 * @return Matrix of shape (windowLength x numWindows), one window per column.
 */
template<typename eT>
inline arma::Mat<eT> SlidingWindow(const arma::Col<eT>& signal,
                                   size_t windowLength,
                                   size_t hopSize)
{
  if (signal.n_elem < windowLength)
  {
    arma::Mat<eT> windows(windowLength, 1, arma::fill::zeros);
    windows.col(0).subvec(0, signal.n_elem - 1) = signal;
    return windows;
  }

  size_t numWindows = (signal.n_elem - windowLength) / hopSize + 1;
  arma::Mat<eT> windows(windowLength, numWindows);

  for (size_t i = 0; i < numWindows; ++i)
  {
    size_t start = i * hopSize;
    windows.col(i) = signal.subvec(start, start + windowLength - 1);
  }

  return windows;
}




} // namespace mlpack

#endif
