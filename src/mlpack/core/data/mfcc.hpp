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

/**
 * ========================================
 *          Theorectical Concept.
 * ========================================
 *
 * General MFCC algorithm concept as follows:
 *
 * 1. Take an audio signal.
 * 1.1 Improve the signal by applying an Finit Impulse high pass filter.
 * 2. Cut this signal into a set of overlapped subsignals using a sliding window
 * function.
 * 3. Applying Hamming window function to make the sides equal to zero.
 * 4. Compute the Power Spectrum for each one of these subsignals.
 * 5. Apply MelFilter banks to the 
 * 6. Compte the log for each one of these sub signals.
 *  -> So far we have MFE
 * 7. Compute the FFT of the previous log by applying DCT function. Now we have
 * MFCC
 *
 * Explains:
 *
 * 1. 
 *
 * 2. Step number two is required in order to preserve both time and frequency
 * domain when we apply FFT, basically we are applying STFT (short time fourier
 * transorm)
 *
 * 3. We must apply hamming window to make the signal goes to zeros on the
 * sides to make it suitable for FFT and avoid spectral leakage.
 *
 * 4. Apply power spectrum, to extract the amplitude.
 *
 *
 *  References:
 *
 *  https://www.youtube.com/watch?v=hF72sY70_IQ
 *  https://www.youtube.com/watch?v=SJo7vPgRlBQ&t=223s&pp=ygULTUZDQyBmaWx0ZXI%3D
 *
 * Matrix structure of the above implementation
 *
 * 1. Audio signal represented as an armadillo Col.
 * 2. Sub signals will result into an arma matrix, the cols are the sub signals
 * and the rows are their number.
 * 3. 
 *
 */

template<typename eT>
inline eT HzToMel(eT hz)
{
  return 2595 * std::log10(1.0 + hz / 700.0);
}

template<typename eT>
inline eT MelToHz(eT mel)
{
  return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
}

/**
 * It would be faster for FFT to use a window size that is pair and dividable
 * by 2. Since we have a window size of 25 ms. This will produce a different
 * samples sizes with different sampling frequencies, and we might ends up with
 * arbitrary non easily dividable window size.
 *
 * Therefore the objective of this function is to multiply by 2 using bitwise
 * operation as long as we are < n.
 *
 * @param n window size, number of samples generated from a specific window
 * (default 25ms).
 *
 */
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
 * @param window Output audio signal converted into a set of sliding windows.
 * @param windowLength Number of samples per window.
 * @param windowStep Number of samples between the start of consecutive windows.
 */
template<typename eT>
inline void SlidingWindow(const arma::Col<eT>& signal,
                          arma::Mat<eT>& windows,
                          size_t windowLength,
                          size_t windowStep)
{
  if (signal.n_elem < windowLength)
  {
    windows.set_size(windowLength, 1);
    windows.col(0).subvec(0, signal.n_elem - 1) = signal;
  }
  else
  {
    size_t numWindows = (signal.n_elem - windowLength) / windowStep + 1;
    windows.set_size(windowLength, numWindows);

    for (size_t i = 0; i < numWindows; ++i)
    {
      size_t start = i * windowStep;
      windows.col(i) = signal.subvec(start, start + windowLength - 1);
    }
  }
}

/**
 * Compute a Hamming window of the given length.
 *
 * The Hamming window is defined as:
 *
 *     w[n] = 0.54 - 0.46 * cos(2π * n / (N - 1))     n = 0, ..., N-1
 *
 * The window tapers the edges of each frame toward zero while keeping the
 * center close to 1.0.  Compared to the rectangular window (no windowing),
 * this greatly reduces spectral leakage — energy that bleeds from one
 * frequency bin into its neighbours due to the abrupt truncation at frame
 * boundaries.
 *
 * @param length Window length in samples (must be >= 2).
 * @return Column vector of length `length`.
 */
template<typename eT>
inline arma::Col<eT> HammingWindow(size_t len)
{
  return (0.54 - (0.46 * arma::cos(2.0 * M_PI *
     arma::linspace<arma::Col<eT>>(0, len - 1, len) / len - 1)));
}


/**
 * Compute the first part power spectrum of each window using arma::fft().
 *
 * Each column of `windows` is zero-padded to `nFFT` length, transformed via
 * arma::fft(), and the energy represented by squared magnitude of the first bins
 * is kept.
 */
template<typename eT>
inline void PowerSpectrum(const arma::Mat<eT>& windows, arma::Mat<eT>& power,
    size_t nFFT)
{
  size_t numWindows = windows.n_cols;
  size_t numBins = nFFT / 2 + 1;
  power.set_size(numBins, numWindows);

  for (size_t i = 0; i < numWindows; ++i)
  {
    // @rcurtin, is there any more efficient way to avoid the copy ? (without move?
    // This zero padding is required by FFT to keep it fast, since the window
    // size might be arbitrary (depending on the sampling frequency).
    arma::Col<eT> padded(nFFT);
    padded.subvec(0, windows.n_rows - 1) = windows.col(i);

    arma::Col<std::complex<eT>> spectrum = arma::fft(padded);

    // Get the power by doing element wise multiplication, note that the
    // spectrum is mirrored so we are getting the first part.
    arma::Col<std::complex<eT>> firstPart = spectrum.subvec(0, numBins - 1);
    power.col(i) = arma::real(firstPart % arma::conj(firstPart));
  }
}

/**
 * Build a mel-scaled triangular filterbank matrix.
 *
 * Places (numFilters + 2) points uniformly on the mel scale between lowFreq
 * and highFreq, converts to Hz, maps to FFT bin indices, and constructs
 * overlapping triangular filters.  Adjacent filters overlap at their half-power
 * points so that the entire frequency range is covered without gaps.
 *
 * @return Matrix of shape (numFilters x (nFFT/2 + 1)).
 */
template<typename eT>
inline arma::Mat<eT> MelFilterbank(size_t numFilters,
                                   size_t nFFT,
                                   size_t sampleRate,
                                   float lowFreq,
                                   float highFreq)
{
  size_t numBins = nFFT / 2 + 1;
  eT melLow = HzToMel(lowFreq);
  eT melHigh = HzToMel(highFreq);

  size_t numPoints = numFilters + 2;
  arma::Col<eT> melPoints = arma::linspace<arma::Col<eT>>(melLow, melHigh,
      numPoints);

  // Convert mel points → Hz → FFT bin index.
  arma::Col<size_t> binIndices(numPoints);
  for (size_t i = 0; i < numPoints; ++i)
  {
    eT hz = MelToHz(melPoints[i]);
    binIndices[i] = static_cast<size_t>(std::floor((nFFT + 1) * hz / sampleRate));
  }

  // Build overlapping triangular filters.
  arma::Mat<eT> filterbank(numFilters, numBins);

  for (size_t m = 0; m < numFilters; ++m)
  {
    size_t left   = binIndices[m];
    size_t center = binIndices[m + 1];
    size_t right  = binIndices[m + 2];

    // Rising slope: left → center.
    if (center > left)
    {
      for (size_t k = left; k <= center && k < numBins; ++k)
      {
        filterbank(m, k) = (k - left) / (center - left);
      }
    }

    // Falling slope: center → right.
    if (right > center)
    {
      for (size_t k = center; k <= right && k < numBins; ++k)
      {
        filterbank(m, k) = static_cast<eT>(right - k) /
                           static_cast<eT>(right - center);
      }
    }
  }

  return filterbank;
}



} // namespace mlpack

#endif
