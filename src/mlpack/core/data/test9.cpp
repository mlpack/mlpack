/**
 * Benchmark: arma::fft() vs KissFFT vs PocketFFT
 *
 * Setup:
 *   git clone https://github.com/mborgerding/kissfft.git
 *   git clone -b cpp https://github.com/mreineck/pocketfft.git
 *
 * Compile for float KissFFT:
 *   g++ -O2 -o bench_f test9.cpp \
 *       -I/meta/mlpack/src -Ikissfft -Ipocketfft \
 *       kissfft/kiss_fft.c kissfft/tools/kiss_fftr.c \
 *       -larmadillo
 *
 * Compile for double KissFFT:
 *   g++ -O2 -Dkiss_fft_scalar=double -o bench_d test9.cpp \
 *       -I/meta/mlpack/src -Ikissfft -Ipocketfft \
 *       kissfft/kiss_fft.c kissfft/tools/kiss_fftr.c \
 *       -larmadillo
 */
#include <chrono>
#include <iostream>
#include <complex>
#include <mlpack.hpp>

#include <kiss_fft.h>
#include <kiss_fftr.h>

#include "pocketfft_hdronly.h"

using namespace mlpack;

template<typename eT>
void BenchArmaFFT(const arma::Mat<eT>& windows, size_t nFFT, size_t numBins)
{
  arma::Mat<std::complex<eT>> spectrum = arma::fft(windows, nFFT);
  arma::Mat<eT> power = arma::square(arma::abs(spectrum.rows(0, numBins - 1)));
}

template<typename eT>
void BenchKissFFTC2C(const arma::Col<eT>& signal, size_t windowLength,
    size_t windowStep, size_t nFFT, size_t numBins, size_t numWindows,
    arma::Mat<eT>& power)
{
  kiss_fft_cfg cfg = kiss_fft_alloc(nFFT, 0, NULL, NULL);

  kiss_fft_cpx* fftIn = new kiss_fft_cpx[nFFT]();
  kiss_fft_cpx* fftOut = new kiss_fft_cpx[nFFT];

  eT* powerPtr = power.memptr();

  for (size_t i = 0; i < numWindows; ++i)
  {
    const eT* sigPtr = signal.memptr() + i * windowStep;

    for (size_t j = 0; j < windowLength; ++j)
    {
      fftIn[j].r = sigPtr[j];
      fftIn[j].i = 0;
    }

    kiss_fft(cfg, fftIn, fftOut);

    eT* colOut = powerPtr + i * numBins;
    for (size_t k = 0; k < numBins; ++k)
      colOut[k] = fftOut[k].r * fftOut[k].r + fftOut[k].i * fftOut[k].i;
  }

  delete[] fftIn;
  delete[] fftOut;
  kiss_fft_free(cfg);
}

template<typename eT>
void BenchKissFFTR2C(const arma::Col<eT>& signal, size_t windowLength,
    size_t windowStep, size_t nFFT, size_t numBins, size_t numWindows,
    arma::Mat<eT>& power)
{
  kiss_fftr_cfg cfg = kiss_fftr_alloc(nFFT, 0, NULL, NULL);

  kiss_fft_scalar* fftIn = new kiss_fft_scalar[nFFT]();
  kiss_fft_cpx* fftOut = new kiss_fft_cpx[numBins];

  eT* powerPtr = power.memptr();

  for (size_t i = 0; i < numWindows; ++i)
  {
    const eT* sigPtr = signal.memptr() + i * windowStep;

    for (size_t j = 0; j < windowLength; ++j)
      fftIn[j] = sigPtr[j];

    kiss_fftr(cfg, fftIn, fftOut);

    eT* colOut = powerPtr + i * numBins;
    for (size_t k = 0; k < numBins; ++k)
      colOut[k] = fftOut[k].r * fftOut[k].r + fftOut[k].i * fftOut[k].i;
  }

  delete[] fftIn;
  delete[] fftOut;
  kiss_fftr_free(cfg);
}

template<typename eT>
void BenchPocketFFTC2C(const arma::Col<eT>& signal, size_t windowLength,
    size_t windowStep, size_t nFFT, size_t numBins, size_t numWindows,
    arma::Mat<eT>& power)
{
  pocketfft::shape_t shape = { nFFT };
  pocketfft::stride_t stride = { sizeof(std::complex<eT>) };
  pocketfft::shape_t axes = { 0 };

  arma::Col<std::complex<eT>> fftIn(nFFT, arma::fill::zeros);
  arma::Col<std::complex<eT>> fftOut(nFFT);

  eT* powerPtr = power.memptr();

  for (size_t i = 0; i < numWindows; ++i)
  {
    const eT* sigPtr = signal.memptr() + i * windowStep;
    std::complex<eT>* inPtr = fftIn.memptr();

    for (size_t j = 0; j < windowLength; ++j)
      inPtr[j] = std::complex<eT>(sigPtr[j], 0);

    pocketfft::c2c(shape, stride, stride, axes, pocketfft::FORWARD,
        fftIn.memptr(), fftOut.memptr(), static_cast<eT>(1));

    eT* colOut = powerPtr + i * numBins;
    const std::complex<eT>* outPtr = fftOut.memptr();
    for (size_t k = 0; k < numBins; ++k)
    {
      eT re = outPtr[k].real();
      eT im = outPtr[k].imag();
      colOut[k] = re * re + im * im;
    }
  }
}

template<typename eT>
void BenchPocketFFTR2C(const arma::Col<eT>& signal, size_t windowLength,
    size_t windowStep, size_t nFFT, size_t numBins, size_t numWindows,
    arma::Mat<eT>& power)
{
  pocketfft::shape_t shape = { nFFT };
  pocketfft::stride_t strideIn = { sizeof(eT) };
  pocketfft::stride_t strideOut = { sizeof(std::complex<eT>) };

  arma::Col<eT> fftIn(nFFT, arma::fill::zeros);
  arma::Col<std::complex<eT>> fftOut(numBins);

  eT* powerPtr = power.memptr();

  for (size_t i = 0; i < numWindows; ++i)
  {
    const eT* sigPtr = signal.memptr() + i * windowStep;
    eT* inPtr = fftIn.memptr();

    // Direct copy — same type, no conversion.
    std::memcpy(inPtr, sigPtr, windowLength * sizeof(eT));

    pocketfft::r2c(shape, strideIn, strideOut, 0, pocketfft::FORWARD,
        fftIn.memptr(), fftOut.memptr(), static_cast<eT>(1));

    eT* colOut = powerPtr + i * numBins;
    const std::complex<eT>* outPtr = fftOut.memptr();
    for (size_t k = 0; k < numBins; ++k)
    {
      eT re = outPtr[k].real();
      eT im = outPtr[k].imag();
      colOut[k] = re * re + im * im;
    }
  }
}

// ===================================================================
// Run all benchmarks for a given type.
// ===================================================================
template<typename eT>
void RunBenchmarks(const char* typeName, size_t sampleRate, size_t duration,
    size_t windowLength, size_t windowStep, size_t nFFT, int numTrials)
{
  size_t signalLength = sampleRate * duration;
  size_t numBins = nFFT / 2 + 1;
  size_t numWindows = (signalLength - windowLength) / windowStep + 1;

  std::cout << "\n===== " << typeName << " =====" << std::endl;

  // Generate signal.
  arma::Col<eT> signal = arma::conv_to<arma::Col<eT>>::from(
      arma::sin(2.0 * M_PI * 440.0 *
      arma::linspace<arma::vec>(0, signalLength - 1, signalLength) /
      sampleRate));

  // Build sliding window matrix for Armadillo benchmark only.
  arma::Mat<eT> windows(windowLength, numWindows);
  for (size_t i = 0; i < numWindows; ++i)
  {
    size_t start = i * windowStep;
    windows.col(i) = signal.subvec(start, start + windowLength - 1);
  }

  arma::Mat<eT> power(numBins, numWindows);

  //// [1] arma::fft batch
  //{
    //double bestMs = 1e9;
    //for (int t = 0; t < numTrials; ++t)
    //{
      //auto t0 = std::chrono::high_resolution_clock::now();
      //BenchArmaFFT(windows, nFFT, numBins);
      //auto t1 = std::chrono::high_resolution_clock::now();
      //bestMs = std::min(bestMs,
          //std::chrono::duration<double, std::milli>(t1 - t0).count());
    //}
    //std::cout << "[1] arma::fft batch c2c:        " << bestMs << " ms"
        //<< std::endl;
  //}

  //// [2] KissFFT c2c
  //{
    //double bestMs = 1e9;
    //for (int t = 0; t < numTrials; ++t)
    //{
      //auto t0 = std::chrono::high_resolution_clock::now();
      //BenchKissFFTC2C(signal, windowLength, windowStep, nFFT,
          //numBins, numWindows, power);
      //auto t1 = std::chrono::high_resolution_clock::now();
      //bestMs = std::min(bestMs,
          //std::chrono::duration<double, std::milli>(t1 - t0).count());
    //}
    //std::cout << "[2] KissFFT c2c:                " << bestMs << " ms"
        //<< std::endl;
  //}

  //// [3] KissFFT r2c
  //{
    //double bestMs = 1e9;
    //for (int t = 0; t < numTrials; ++t)
    //{
      //auto t0 = std::chrono::high_resolution_clock::now();
      //BenchKissFFTR2C(signal, windowLength, windowStep, nFFT,
          //numBins, numWindows, power);
      //auto t1 = std::chrono::high_resolution_clock::now();
      //bestMs = std::min(bestMs,
          //std::chrono::duration<double, std::milli>(t1 - t0).count());
    //}
    //std::cout << "[3] KissFFT r2c:                " << bestMs << " ms"
        //<< std::endl;
//  }

  // [4] PocketFFT c2c
  {
    double bestMs = 1e9;
    for (int t = 0; t < numTrials; ++t)
    {
      auto t0 = std::chrono::high_resolution_clock::now();
      BenchPocketFFTC2C(signal, windowLength, windowStep, nFFT,
          numBins, numWindows, power);
      auto t1 = std::chrono::high_resolution_clock::now();
      bestMs = std::min(bestMs,
          std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::cout << "[4] PocketFFT c2c:              " << bestMs << " ms"
        << std::endl;
  }

  // [5] PocketFFT r2c
  {
    double bestMs = 1e9;
    for (int t = 0; t < numTrials; ++t)
    {
      auto t0 = std::chrono::high_resolution_clock::now();
      BenchPocketFFTR2C(signal, windowLength, windowStep, nFFT,
          numBins, numWindows, power);
      auto t1 = std::chrono::high_resolution_clock::now();
      bestMs = std::min(bestMs,
          std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::cout << "[5] PocketFFT r2c:              " << bestMs << " ms"
        << std::endl;
  }
}

int main()
{
  size_t sampleRate = 16000;
  size_t duration = 3600;
  size_t windowLength = 401;
  size_t windowStep = 160;
//  size_t nFFT = 512;
  size_t nFFT = windowLength;
  int numTrials = 3;

  size_t signalLength = sampleRate * duration;
  size_t numWindows = (signalLength - windowLength) / windowStep + 1;

  std::cout << "kiss_fft_scalar: "
      << (sizeof(kiss_fft_scalar) == 4 ? "float" : "double") << std::endl;
  std::cout << "Signal: " << duration << "s, " << signalLength
      << " samples" << std::endl;
  std::cout << "Windows: " << numWindows << ", nFFT: " << nFFT << std::endl;
  std::cout << "Trials: " << numTrials << " (reporting minimum)" << std::endl;

  RunBenchmarks<double>("double", sampleRate, duration, windowLength,
      windowStep, nFFT, numTrials);

  RunBenchmarks<float>("float", sampleRate, duration, windowLength,
      windowStep, nFFT, numTrials);

  return 0;
}
