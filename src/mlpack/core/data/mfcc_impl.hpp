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
inline arma::Col<eT> HammingWindow(size_t len)
{
  return (0.54 - (0.46 * arma::cos(2.0 * M_PI * arma::linspace<arma::Col<eT>>(0,
    len - 1, len) / (len - 1))));
}

template<typename eT>
inline eT HzToMel(eT hz)
{
  return 2595 * std::log10(1.0 + hz / 700.0);
}

template<typename eT>
inline arma::Mat<eT> MelFilterbank(size_t numFilters,
                                   size_t nFFT,
                                   size_t sampleRate,
                                   double lowFreq,
                                   double highFreq)
{
  size_t numBins = nFFT / 2 + 1;
  eT melLow = HzToMel(static_cast<eT>(lowFreq));
  eT melHigh = HzToMel(static_cast<eT>(highFreq));

  size_t numPoints = numFilters + 2;
  arma::Col<eT> melPoints = arma::linspace<arma::Col<eT>>(melLow, melHigh,
      numPoints);

  arma::Col<eT> hzPoints = 700.0 *
      (arma::pow(10.0 * arma::ones<arma::Col<eT>>(numPoints),
      melPoints / 2595.0) - 1.0);

  arma::Col<eT> binFreqHz = arma::regspace<arma::Col<eT>>(0, numBins - 1)
      * sampleRate / nFFT;

  arma::Mat<eT> melFilterbank(numFilters, numBins);

  for (size_t i = 0; i < numFilters; ++i)
  {
    eT left   = hzPoints(i);
    eT center = hzPoints(i + 1);
    eT right  = hzPoints(i + 2);

    if (center <= left || right <= center)
      continue;

    arma::Col<eT> goUp = (binFreqHz - left) / (center - left);
    arma::Col<eT> goDown = (right - binFreqHz) / (right - center);

    melFilterbank.row(i) = arma::clamp(arma::min(goUp, goDown), 0, 1).t();
  }

  return melFilterbank;
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

template<typename eT>
inline void MFE(const arma::Mat<eT>& inputSignal,
                arma::Mat<eT>& mfe,
                size_t sampleRate,
                size_t numMelFilters,
                double windowLength,
                double windowStep,
                size_t nFFT,
                double lowFreq,
                double highFreq,
                const typename std::enable_if_t<
                    std::is_floating_point<eT>::value>*)
{
  std::chrono::time_point<std::chrono::high_resolution_clock> t0, t1, t2, t3, t4;

  if (highFreq == 0.0f)
    highFreq = sampleRate / 2.0f;

  size_t lengthInSamples = static_cast<size_t>(windowLength * sampleRate
      / 1000.0f);
  size_t stepsInSamples = static_cast<size_t>(windowStep * sampleRate
      / 1000.0f);

  // This is slowing the performance of FFT by 4 seconds.
  //if (nFFT == 0)
  //  nFFT = NextPowerOf2(lengthInSamples);
  //lengthInSamples = nFFT //(Deduced from NextPowerOf2)

  // 2 seconds speed up. Elapsed: 34402 ms 
  nFFT = lengthInSamples;

  size_t numBins = nFFT / 2 + 1;

  t0 = std::chrono::high_resolution_clock::now();
  arma::Mat<eT> filterBanks = MelFilterbank<eT>(numMelFilters, nFFT, sampleRate,
      lowFreq, highFreq);
  
  t1 = std::chrono::high_resolution_clock::now();

  size_t totalWindows = 0;
  for (size_t i = 0; i < inputSignal.n_cols; ++i)
  {
    totalWindows += (inputSignal.n_rows - lengthInSamples)
        / stepsInSamples + 1;
  }

  mfe.set_size(numMelFilters, totalWindows);

  size_t colOffset = 0;
  // Benchmarking on one signal for now.
  for (size_t i = 0; i < inputSignal.n_cols; ++i)
  {
    arma::Mat<eT> slidingWindows;
    arma::Mat<eT> power;

    SlidingWindow(inputSignal.col(i), slidingWindows, lengthInSamples,
        stepsInSamples);

    t2 = std::chrono::high_resolution_clock::now();
    PowerSpectrum(slidingWindows, power, nFFT);

    t3 = std::chrono::high_resolution_clock::now();
    // Adding a small value 1e-10 so we do not take the log of 0, in case the
    // multiplication results in zero.
    mfe.cols(colOffset, colOffset + power.n_cols - 1) =
        arma::log((filterBanks * power) + 1e-10);

    colOffset += power.n_cols;

    t4 = std::chrono::high_resolution_clock::now();
  }

  std::cout << "MelFilterBanks: " << std::chrono::duration<double, std::milli>(t1-t0).count() << std::endl;
  std::cout << "SlidingWindow: "  << std::chrono::duration<double, std::milli>(t2-t1).count() << std::endl;
  std::cout << "FFT:           "  << std::chrono::duration<double, std::milli>(t3-t2).count() << std::endl;
  std::cout << "Log:           "  << std::chrono::duration<double, std::milli>(t4-t3).count() << std::endl;
}

template<typename eT>
inline void MFE(const arma::Mat<eT>& inputSignal,
                arma::Mat<eT>& mfe,
                size_t sampleRate,
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
      "MFE(): input matrix must have a floating-point element type "
      "(float or double). Integer matrices are not supported because "
      "FFT, log, and other operations require floating-point "
      "arithmetic.");
}

inline size_t NextPowerOf2(size_t n)
{
  size_t p = 1;
  while (p < n && p < (SIZE_MAX / 2))
    p <<= 1;
  return p;
}

template<typename eT>
inline void PowerSpectrum(const arma::Mat<eT>& windows, arma::Mat<eT>& power,
    size_t nFFT)
{
  size_t numBins = nFFT / 2 + 1;
  // Removing the loop allows to gain 7 seconds speed up in execution time 
  // when running FFT on an 1 hour signal.
  // spectrum is (nFFT x numWindows).
  arma::Mat<std::complex<eT>> spectrum = arma::fft(windows, nFFT);

  // Keep only the first part of the spectrum since it is mirrored.
  // Using this expression, provides 2 seconds speed up as well.
  power = square(abs(spectrum.rows(0, numBins - 1)));
}

template<typename MatType, typename eT>
inline void SlidingWindow(const MatType& signal,
                          arma::Mat<eT>& windows,
                          size_t windowLength,
                          size_t windowStep)
{
  //typedef typename MatType::elem_type eT;

  // Elapsed: 32770 ms Best so far
  arma::Col<eT> hWindow = HammingWindow<eT>(windowLength);

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
      windows.col(i) = signal.subvec(start, start + windowLength - 1) % hWindow;
    }
  }
}

} // namespace mlpack

#endif
