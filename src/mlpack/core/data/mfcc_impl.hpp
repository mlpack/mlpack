/**
 * @file core/data/mfcc_impl.hpp
 * @author Omar Shrit
 *
 * Implementation file for MFE (Mel-Frequency Energy) and MFCC
 * (Mel-Frequency Cepstral Coefficients) filters
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
inline arma::Col<eT> HammingWindow(size_t len)
{
  return (0.54 - (0.46 * arma::cos(2.0 * M_PI *
     arma::linspace<arma::Col<eT>>(0, len - 1, len) / (len - 1))));
}

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

  // The points are shared between all filters, only the last filter needs 2
  // additional points, one for the center of the triangle, and the other is
  // for the right part of the triangle, I made a mistake intially by having
  // the numPoints = numFilters * 3.
  size_t numPoints = numFilters + 2;
  arma::Col<eT> melPoints = arma::linspace<arma::Col<eT>>(melLow, melHigh,
      numPoints);

  // We need to convert mel points back to Hz so we can get the FFT bin index.
  arma::Col<size_t> binIndices(numPoints);
  for (size_t i = 0; i < numPoints; ++i)
  {
    eT hz = MelToHz(melPoints[i]);
    binIndices[i] = static_cast<size_t>(std::floor((nFFT + 1) * hz / sampleRate));
  }

  arma::Mat<eT> melFilterbank(numFilters, numBins);

  arma::Col<eT> XI = arma::linspace<arma::Col<eT>>(0, numBins - 1, numBins);

  for (size_t i = 0; i < numFilters; ++i)
  {
    eT left   = binIndices[i];
    eT center = binIndices[i + 1];
    eT right  = binIndices[i + 2];

    if (left == center || center == right)
    {
      melFilterbank.row(i).zeros();
      continue;
    }
    // Triangle vertices: these are basically coordinates for the triangles
    // that rises from 0 at left to 1 at center, falls to 0
    // at right.  Bins outside [left, right] extrapolate to 0
    arma::Col<eT> X = { left, center, right };
    arma::Col<eT> Y = { 0, 1, 0};

    arma::Col<eT> YI;
    arma::interp1(X, Y, XI, YI, "*linear", 0);

    // @rcurtin, this is probably much faster since we are going to do matrix
    // multiplication later in MFE (filterbanks x Power).
    // Otherwise I need to assign it to .col and then transpose .t()
    melFilterbank.row(i) = YI.t();
  }

  return melFilterbank;
}

template<typename eT>
inline void MFE(const arma::Mat<eT>& inputSignal,
                arma::Mat<eT>& mfe,
                size_t sampleRate,
                size_t numMelFilters,
                float windowLength,
                float windowStep,
                size_t nFFT,
                float lowFreq,
                float highFreq,
                const typename std::enable_if_t<
                    std::is_floating_point<eT>::value>*)
{
  if (highFreq == 0.0f)
    highFreq = sampleRate / 2.0f;

  size_t lengthInSamples = static_cast<size_t>(windowLength * sampleRate
      / 1000.0f);
  size_t stepsInSamples = static_cast<size_t>(windowStep * sampleRate
      / 1000.0f);

  if (nFFT == 0)
    nFFT = NextPowerOf2(lengthInSamples);

  arma::Mat<eT> filterBanks = MelFilterbank<eT>(numMelFilters, nFFT, sampleRate,
      lowFreq, highFreq);
  
  // Process each column (signal) independently and concatenate results.
  std::vector<arma::Mat<eT>> results(inputSignal.n_cols);

  for (size_t i = 0; i < inputSignal.n_cols; ++i)
  {
    arma::Mat<eT> slidingWindows;
    arma::Mat<eT> power;
    arma::Col<eT> audioSignal = inputSignal.col(i);

    SlidingWindow(audioSignal, slidingWindows, lengthInSamples,
        stepsInSamples);

    // Probably it is more efficient to compute HammingWindow once before the
    // loop @rucrtin, what do you think ?
    slidingWindows.each_col() %= HammingWindow<eT>(lengthInSamples);

    PowerSpectrum(slidingWindows, power, nFFT);

    // Adding a small value 1e-10 so we do not take the log of 0, in case the
    // multiplication results in zero.
    results[i] = arma::log((filterBanks * power) + 1e-10);
  }

  mfe = results[0];
  for (size_t i = 1; i < inputSignal.n_cols; ++i)
    mfe = arma::join_horiz(mfe, results[i]);
}

template<typename eT>
inline void MFE(const arma::Mat<eT>& inputSignal,
                arma::Mat<eT>& mfe,
                size_t sampleRate,
                size_t numMelFilters,
                float windowLength,
                float windowStep,
                size_t nFFT,
                float lowFreq,
                float highFreq,
                const typename std::enable_if_t<
                    !std::is_floating_point<eT>::value>*)
{
  static_assert(std::is_floating_point<eT>::value,
      "MFE(): input matrix must have a floating-point element type "
      "(float or double). Integer matrices are not supported because "
      "FFT, log, and other operations require floating-point "
      "arithmetic.");
}

inline size_t NextPowerOf2(size_t n)
{
  size_t p = 1;
  while (p < n)
    p <<= 1;
  return p;
}

template<typename eT>
inline void PowerSpectrum(const arma::Mat<eT>& windows, arma::Mat<eT>& power,
    size_t nFFT)
{
  size_t numWindows = windows.n_cols;
  size_t numBins = nFFT / 2 + 1;
  power.set_size(numBins, numWindows);

  for (size_t i = 0; i < numWindows; ++i)
  {
    // @rcurtin, is there any more efficient way to avoid the copy ? (without
    // using std::move? This zero padding is required by FFT to keep it fast,
    // since the window size might be arbitrary (depending on the sampling frequency).
    arma::Col<eT> padded(nFFT);
    padded.subvec(0, windows.n_rows - 1) = windows.col(i);

    arma::Col<std::complex<eT>> spectrum = arma::fft(padded);

    // Get the power by doing element wise multiplication, note that the
    // spectrum is mirrored so we must getting the first part only.
    arma::Col<std::complex<eT>> firstPart = spectrum.subvec(0, numBins - 1);
    power.col(i) = arma::real(firstPart % arma::conj(firstPart));
  }
}

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

} // namespace mlpack

#endif
