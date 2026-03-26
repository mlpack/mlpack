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

TEST_CASE("FFTSineWav", "[MFCC][tiny]")
{
  // We basically generate a sine wave and try to get the power of the highest
  // bin, we should be able to see that the frequency of this bin is close to
  // 440 Hz
  size_t sampleRate = 16000;
  float frequency = 440.0f;
  float duration = 0.032f;
  size_t nSamples = static_cast<size_t>(sampleRate * duration);
  size_t nFFT = 512;
  size_t numBins = nFFT / 2 + 1;

  arma::fvec t = arma::linspace<arma::fvec>(0, duration, nSamples);
  arma::fvec signal = arma::sin(2.0f * M_PI * frequency * t);

  arma::fmat power;

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

TEST_CASE("MFE", "[MFCC][tiny]")
{

}
