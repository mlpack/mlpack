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

TEST_CASE("MelToHZ", "[MFCC][tiny]")
{
  REQUIRE(MelToHz(0) == 0);
  REQUIRE(MelToHz(1200) == Approx(1271.28).epsilon(1e-4));
  REQUIRE(MelToHz(1400) == Approx(1632.09).epsilon(1e-4));
  REQUIRE(MelToHz(1600) == Approx(2057.46).epsilon(1e-4));
  REQUIRE(MelToHz(1800) == Approx(2559.64).epsilon(1e-4));
  REQUIRE(MelToHz(2000) == Approx(3153.38).epsilon(1e-4));
  REQUIRE(MelToHz(2200) == Approx(3856.43).epsilon(1e-4));
  REQUIRE(MelToHz(2400) == Approx(4690.03).epsilon(1e-4));
  REQUIRE(MelToHz(2600) == Approx(5679.52).epsilon(1e-4));
  REQUIRE(MelToHz(2800) == Approx(6855.07).epsilon(1e-4));
  REQUIRE(MelToHz(3000) == Approx(8252.34).epsilon(1e-4));
}

TEST_CASE("HzToMel", "[MFCC][tiny]")
{
  REQUIRE(HzToMel(0) == 0);
  REQUIRE(HzToMel(1000) == Approx(999.999).epsilon(1e-4));
  REQUIRE(HzToMel(1500) == Approx(1264.49).epsilon(1e-4));
  REQUIRE(HzToMel(2000) == Approx(1479.92).epsilon(1e-4));
  REQUIRE(HzToMel(3000) == Approx(1819.86).epsilon(1e-4));
  REQUIRE(HzToMel(4000) == Approx(2096.63).epsilon(1e-4));
  REQUIRE(HzToMel(5000) == Approx(2329.63).epsilon(1e-4));
  REQUIRE(HzToMel(6000) == Approx(2530.84).epsilon(1e-4));
  REQUIRE(HzToMel(7000) == Approx(2707.36).epsilon(1e-4));
  REQUIRE(HzToMel(8000) == Approx(2863.47).epsilon(1e-4));
}

TEST_CASE("HammingWindow", "[MFCC][tiny]")
{

}

TEST_CASE("SlidingWindow", "[MFCC][tiny]")
{
  arma::Mat<size_t> windows;
  arma::Col<size_t> linearSignal =
      arma::linspace<arma::Col<size_t>>(0, 999, 1000);

  SlidingWindow(linearSignal, windows, 10, 5);

  REQUIRE(windows.n_cols == 199);
  REQUIRE(windows.n_rows == 10);

  // Check for the signal interleaved.
  REQUIRE(windows.at(0, 0) == 0);
  REQUIRE(windows.at(0, 1) == 5);
  REQUIRE(windows.at(0, 2) == 15);
  REQUIRE(windows.at(0, 198) == 990);
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

TEST_CASE("FilterBanks", "[MFCC][tiny]")
{

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
  REQUIRE(mfe.n_cols == 98);

  // All values should be approximately log(1e-10) ≈ -23.03.
  float logEps = std::log(1e-10f);
  REQUIRE(mfe.min() == Approx(logEps).margin(0.1));
  REQUIRE(mfe.max() == Approx(logEps).margin(0.1));
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
  REQUIRE(mfe.n_cols == 98);

  arma::Col<eT> meanMFE = arma::mean(mfe, 1);

  // Filter 0 should have the highest energy (DC is at 0 Hz).
  size_t peakBin = meanMFE.index_max();
  REQUIRE(peakBin == 0);

  // Filter 0 should be well above the epsilon floor.
  float logEps = std::log(1e-10f);
  REQUIRE(meanMFE(0) > logEps + 5.0f);

  // All other filters should be near the epsilon floor.
  size_t nearFloorCount = 0;
  for (size_t i = 1; i < meanMFE.n_elem; ++i)
  {
    if (std::abs(meanMFE(i) - logEps) < 2.0f)
      ++nearFloorCount;
  }
  REQUIRE(nearFloorCount >= 35);

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
  REQUIRE(mfe.n_cols == 98);

  arma::Col<eT> meanMFE = arma::mean(mfe, 1);

  // The peak mel bin should be in the low range (440 Hz),
  // which maps to roughly filter 5–6 out of 40).
  size_t peakBin = meanMFE.index_max();
  REQUIRE(peakBin <= 10);

  // Since we have a pure Sine 440 Hz signal, most of the other bins are close
  // to zero. (Check FFT sine wave test). Therefore, the log should be very
  // small (-23)
  float delta = std::log(1e-10f);
  size_t nearFloorCount = 0;
  for (size_t i = 0; i < meanMFE.n_elem; ++i)
  {
    if (std::abs(meanMFE[i] - delta) < 1.0f)
      ++nearFloorCount;
  }
  REQUIRE(nearFloorCount >= 30);
}
