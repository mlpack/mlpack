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

}

TEST_CASE("HzToMel", "[MFCC][tiny]")
{

}

TEST_CASE("HammingWindow", "[MFCC][tiny]")
{

}

TEST_CASE("FFT", "[MFCC][tiny]")
{

}

TEST_CASE("FilterBanks", "[MFCC][tiny]")
{

}

TEST_CASE("MFE", "[MFCC][tiny]")
{

}
