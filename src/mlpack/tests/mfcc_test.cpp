/**
 * @file tests/mfcc_test.cpp
 * @author Omar Shrit 
 *
 * Tests for MFE() and MFCC(), and internal functions.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>
#include "catch.hpp"
#include "test_catch_tools.hpp"

using namespace mlpack;
using namespace std;

/*
 * MFCC and MFE should run on embedded systems.
 * Therefore all of the following are labeled [tiny]
 */
TEST_CASE("HzToMel", "[MFCC][tiny]")
{
  REQUIRE(HzToMel(0) == 0);
  REQUIRE(HzToMel(1000) == Approx(999).epsilon(1e-1));
  }

TEMPLATE_TEST_CASE("FFTSineWave", "[MFCC][tiny]", float, double)
{
  typedef TestType eT;
  // We basically generate a sine wave and try to get the power of the highest
  // bin, we should be able to see that the frequency of this bin is close to
  // 440 Hz
  size_t sampleRate = 16000;
  float frequency = 440.0f;
  float duration = 0.032f;
  size_t nSamples = static_cast<size_t>(sampleRate * duration);
  size_t nFFT = 512;

  arma::Col<eT> t = arma::linspace<arma::Col<eT>>(0, duration, nSamples);
  arma::Col<eT> signal = arma::sin(2.0f * M_PI * frequency * t);

  arma::Mat<eT> power;

  PowerSpectrum(signal, power, nFFT);

  size_t peakBin = power.index_max();
  float peakPower = power.max();
  float peakFreq = peakBin * sampleRate / nFFT;
  float binWidth = sampleRate / nFFT;

  REQUIRE(peakBin == 14);
  REQUIRE(peakPower == Approx(62655.3).epsilon(1));
  REQUIRE(peakFreq == Approx(437.0).epsilon(1));
  REQUIRE(binWidth == Approx(31.25).epsilon(1));
}

TEMPLATE_TEST_CASE("FilterBanks", "[MFCC][tiny]", float, double)
{
  typedef TestType eT;
  size_t numFilters = 40;
  size_t nFFT = 1024;
  size_t sampleRate = 16000;

  arma::Mat<eT> filterBanks = MelFilterbank<eT>(numFilters, nFFT, sampleRate,
      0.0f, 8000.0f);

  // Shape: 40 filters x 257 bins.
  REQUIRE(filterBanks.n_rows == 40);
  REQUIRE(filterBanks.n_cols == 513);

  REQUIRE(filterBanks.min() >= 0.0);
  REQUIRE(filterBanks.max() <= 1.0);

  // The peak positions should be monotonically increasing (each filter
  // covers a higher frequency than the previous).
  size_t prevPeak = 0;
  for (size_t m = 0; m < numFilters; ++m)
  {
    size_t peak = filterBanks.row(m).index_max();
    REQUIRE(peak > prevPeak);
    prevPeak = peak;
  }

  // We need to be sure that the first filter is covering the bins with low
  // frequencies.
  size_t firstPeak = filterBanks.row(0).index_max();
  REQUIRE(firstPeak <= 5);

  // Similar as the above assert, but for high frequencies.
  size_t lastPeak = filterBanks.row(numFilters - 1).index_max();
  REQUIRE(lastPeak >= 400);
}

/*
 * Signal all zeros is basically silence.
 * FFT is also going to be equal to zeros. no energy.
 */
TEMPLATE_TEST_CASE("MFESilence", "[MFCC][tiny]", float, double)
{
  typedef TestType eT;
  size_t sampleRate = 16000;
  arma::Mat<eT> input(sampleRate, 1);

  arma::Mat<eT> mfe;
  MFE(input, mfe, sampleRate);

  REQUIRE(mfe.n_rows == 40);
  REQUIRE(mfe.n_cols == 97);

  // All values should be approximately log(1e-10) ~ -23.03.
  float logEps = std::log(1e-10f);
  REQUIRE(mfe.min() <= logEps);
  REQUIRE(mfe.max() <= logEps);
}

TEMPLATE_TEST_CASE("MFEDC", "[MFCC][tiny]", float, double)
{
  typedef TestType eT;

  size_t sampleRate = 16000;
  arma::Mat<eT> input(sampleRate, 1);
  input.fill(0.5f);

  arma::Mat<eT> mfe;
  MFE(input, mfe, sampleRate);

  REQUIRE(mfe.n_rows == 40);
  REQUIRE(mfe.n_cols == 97);

  arma::Col<eT> meanMFE = arma::mean(mfe, 1);

  // Filter 0 should have the highest energy (DC is at 0 Hz).
  size_t peakBin = meanMFE.index_max();
  REQUIRE(peakBin == 0);

  // Filter 0 should be well above the epsilon floor.
  float logEps = std::log(1e-10f);
  REQUIRE(meanMFE(0) > logEps + 5.0f);

  size_t lowEnergyCount = 0;
  for (size_t i = 1; i < meanMFE.n_elem; ++i)
  {
    if (meanMFE(i) < meanMFE(0) - 5.0)
      ++lowEnergyCount;
  }
  REQUIRE(lowEnergyCount >= 35);

  // All frames should be nearly identical (stationary signal).
  float meanFrameVar = arma::mean(arma::var(mfe, 0, 1));
  REQUIRE(meanFrameVar < 1.0f);
}

TEMPLATE_TEST_CASE("MFEPureSine440", "[MFCC][tiny]", float, double)
{
  typedef TestType eT;

  size_t sampleRate = 16000;
  arma::Col<eT> t = arma::linspace<arma::Col<eT>>(0, 1.0f, sampleRate);
  arma::Mat<eT> input = arma::sin(2.0f * (float) M_PI * 440.0f * t);

  arma::Mat<eT> mfe;
  MFE(input, mfe, sampleRate);

  REQUIRE(mfe.n_rows == 40);
  REQUIRE(mfe.n_cols == 97);

  arma::Col<eT> meanMFE = arma::mean(mfe, 1);

  // The peak mel bin should be in the low range (440 Hz),
  // which maps to roughly filter 5–6 out of 40).
  size_t peakBin = meanMFE.index_max();
  REQUIRE(peakBin <= 10);

  // Since we have a pure Sine 440 Hz signal, most of the other bins should be
  // close to zero. However, it seems that even with Hamming Window, we still
  // some spectrum leakage which is fine. To pass this test we are checking
  // that other bins have 8 nepers less (approx 70 db) than peak bin.
  double peakVal = meanMFE.max();
  size_t lowEnergyCount = 0;
  for (size_t i = 0; i < meanMFE.n_elem; ++i)
  {
    if (meanMFE[i] < peakVal - 8.0)
      ++lowEnergyCount;
  }
  REQUIRE(lowEnergyCount >= 30);
}

