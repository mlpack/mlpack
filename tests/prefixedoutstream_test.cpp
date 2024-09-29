/**
 * @file tests/prefixedoutstream_test.cpp
 * @author Matthew Amidon, Ryan Curtin
 *
 * Tests for the PrefixedOutStream class
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <iostream>
#include <sstream>

#include <mlpack/core.hpp>

#include "catch.hpp"

#define BASH_RED "\033[0;31m"
#define BASH_GREEN "\033[0;32m"
#define BASH_YELLOW "\033[0;33m"
#define BASH_CYAN "\033[0;36m"
#define BASH_CLEAR "\033[0m"

using namespace mlpack;
using namespace mlpack::util;

/**
 * Test the output of IO using PrefixedOutStream.  We will pass bogus
 * input to a stringstream so that none of it gets to the screen.
 */
TEST_CASE("TestPrefixedOutStreamBasic", "[PrefixedOutStreamTest]")
{
  std::stringstream ss;
  PrefixedOutStream pss(ss, BASH_GREEN "[INFO ] " BASH_CLEAR);

  pss << "This shouldn't break anything" << std::endl;
  REQUIRE(ss.str() ==
      BASH_GREEN "[INFO ] " BASH_CLEAR "This shouldn't break anything\n");

  ss.str("");
  pss << "Test the new lines...";
  pss << "shouldn't get 'Info' here." << std::endl;
  REQUIRE(ss.str() ==
      BASH_GREEN "[INFO ] " BASH_CLEAR
      "Test the new lines...shouldn't get 'Info' here.\n");

  pss << "But now I should." << std::endl << std::endl;
  pss << "";
  REQUIRE(ss.str() ==
      BASH_GREEN "[INFO ] " BASH_CLEAR
      "Test the new lines...shouldn't get 'Info' here.\n"
      BASH_GREEN "[INFO ] " BASH_CLEAR "But now I should.\n"
      BASH_GREEN "[INFO ] " BASH_CLEAR "\n"
      BASH_GREEN "[INFO ] " BASH_CLEAR "");
}

/**
 * Test that we can correctly output Armadillo objects to PrefixedOutStream
 * objects.
 */
TEST_CASE("TestArmadilloPrefixedOutStream", "[PrefixedOutStreamTest]")
{
  // We will test this with both a vector and a matrix.
  arma::vec test("1.0 1.5 2.0 2.5 3.0 3.5 4.0");

  std::stringstream ss;
  PrefixedOutStream pss(ss, BASH_GREEN "[INFO ] " BASH_CLEAR);

  pss << test;
  // This should result in nothing being on the current line (since it clears
  // it).
  REQUIRE(ss.str() == BASH_GREEN "[INFO ] " BASH_CLEAR "   1.0000\n"
      BASH_GREEN "[INFO ] " BASH_CLEAR "   1.5000\n"
      BASH_GREEN "[INFO ] " BASH_CLEAR "   2.0000\n"
      BASH_GREEN "[INFO ] " BASH_CLEAR "   2.5000\n"
      BASH_GREEN "[INFO ] " BASH_CLEAR "   3.0000\n"
      BASH_GREEN "[INFO ] " BASH_CLEAR "   3.5000\n"
      BASH_GREEN "[INFO ] " BASH_CLEAR "   4.0000\n");

  ss.str("");
  pss << trans(test);

  // This should result in there being stuff on the line.
  REQUIRE(ss.str() == BASH_GREEN "[INFO ] " BASH_CLEAR
      "   1.0000   1.5000   2.0000   2.5000   3.0000   3.5000   4.0000\n");

  arma::mat test2("1.0 1.5 2.0; 2.5 3.0 3.5; 4.0 4.5 4.99999");
  ss.str("");
  pss << test2;
  REQUIRE(ss.str() ==
      BASH_GREEN "[INFO ] " BASH_CLEAR "   1.0000   1.5000   2.0000\n"
      BASH_GREEN "[INFO ] " BASH_CLEAR "   2.5000   3.0000   3.5000\n"
      BASH_GREEN "[INFO ] " BASH_CLEAR "   4.0000   4.5000   5.0000\n");

  // Try and throw a curveball by not clearing the line before outputting
  // something else.  The PrefixedOutStream should not force Armadillo objects
  // onto their own lines.
  ss.str("");
  pss << "hello" << test2;
  REQUIRE(ss.str() ==
      BASH_GREEN "[INFO ] " BASH_CLEAR "hello   1.0000   1.5000   2.0000\n"
      BASH_GREEN "[INFO ] " BASH_CLEAR "   2.5000   3.0000   3.5000\n"
      BASH_GREEN "[INFO ] " BASH_CLEAR "   4.0000   4.5000   5.0000\n");
}

/**
 * Test that we can correctly output things in general.
 */
TEST_CASE("TestPrefixedOutStream", "[PrefixedOutStreamTest]")
{
  std::stringstream ss;
  PrefixedOutStream pss(ss, BASH_GREEN "[INFO ] " BASH_CLEAR);

  pss << "hello world I am ";
  pss << 7;

  REQUIRE(ss.str() ==
      BASH_GREEN "[INFO ] " BASH_CLEAR "hello world I am 7");

  pss << std::endl;
  REQUIRE(ss.str() ==
      BASH_GREEN "[INFO ] " BASH_CLEAR "hello world I am 7\n");

  ss.str("");
  pss << std::endl;
  REQUIRE(ss.str() ==
      BASH_GREEN "[INFO ] " BASH_CLEAR "\n");
}

/**
 * Test format modifiers.
 */
TEST_CASE("TestPrefixedOutStreamModifiers", "[PrefixedOutStreamTest]")
{
  std::stringstream ss;
  PrefixedOutStream pss(ss, BASH_GREEN "[INFO ] " BASH_CLEAR);

  pss << "I have a precise number which is ";
  pss << std::setw(6) << std::setfill('0') << (int)156;

  REQUIRE(ss.str() ==
      BASH_GREEN "[INFO ] " BASH_CLEAR
      "I have a precise number which is 000156");
}

/**
 * Test formatted floating-point output.
 */
TEST_CASE("TestFormattedOutput", "[PrefixedOutStreamTest]")
{
  std::stringstream ss;
  PrefixedOutStream pss(ss, BASH_GREEN "[INFO ]" BASH_CLEAR);

  const double pi = std::acos(-1.0);
  pss << std::setprecision(10) << pi;

  REQUIRE(ss.str() ==
      BASH_GREEN "[INFO ]" BASH_CLEAR "3.141592654");
}

/**
 * Test custom precision output of arma objects.
 */
TEST_CASE("TestArmaCustomPrecision", "[PrefixedOutStreamTest]")
{
  std::stringstream ss;
  PrefixedOutStream pss(ss, BASH_GREEN "[INFO ] " BASH_CLEAR);
  // The vector to be tested.
  arma::vec test("1.0 1.5 2.0 2.5 3.0 3.5 4.0");
  // The matrix to be tested.
  arma::mat test2("1.0 1.5 2.0; 2.5 3.0 3.5; 4.0 4.5 4.99999");

  // Try to print armadillo objects with custom precision.
  ss << std::fixed;
  ss << std::setprecision(6);
  ss.str("");

  pss << test;

  REQUIRE(ss.str() ==
      BASH_GREEN "[INFO ] " BASH_CLEAR "   1.000000\n"
      BASH_GREEN "[INFO ] " BASH_CLEAR "   1.500000\n"
      BASH_GREEN "[INFO ] " BASH_CLEAR "   2.000000\n"
      BASH_GREEN "[INFO ] " BASH_CLEAR "   2.500000\n"
      BASH_GREEN "[INFO ] " BASH_CLEAR "   3.000000\n"
      BASH_GREEN "[INFO ] " BASH_CLEAR "   3.500000\n"
      BASH_GREEN "[INFO ] " BASH_CLEAR "   4.000000\n");

  ss.str("");

  pss << trans(test);

  REQUIRE(ss.str() ==
      BASH_GREEN "[INFO ] " BASH_CLEAR
      "   1.000000   1.500000   2.000000   2.500000"
      "   3.000000   3.500000   4.000000\n");

  // Try printing a matrix, with higher precision.
  ss << std::setprecision(8);
  ss.str("");

  pss << test2;

  REQUIRE(ss.str() ==
      BASH_GREEN "[INFO ] " BASH_CLEAR
      "   1.00000000   1.50000000   2.00000000\n"
      BASH_GREEN "[INFO ] " BASH_CLEAR
      "   2.50000000   3.00000000   3.50000000\n"
      BASH_GREEN "[INFO ] " BASH_CLEAR
      "   4.00000000   4.50000000   4.99999000\n");

  // Try alignment with larger values.
  test2.at(2) = 40;
  ss.str("");
  pss << trans(test2);

  REQUIRE(ss.str() ==
      BASH_GREEN "[INFO ] " BASH_CLEAR
      "    1.00000000    2.50000000   40.00000000\n"
      BASH_GREEN "[INFO ] " BASH_CLEAR
      "    1.50000000    3.00000000    4.50000000\n"
      BASH_GREEN "[INFO ] " BASH_CLEAR
      "    2.00000000    3.50000000    4.99999000\n");

  // Test stream after reset.
  test2.at(2) = 4;
  ss << std::setprecision(6);
  ss.unsetf(std::ios::floatfield);
  ss.str("");

  pss << test2;
  REQUIRE(ss.str() ==
      BASH_GREEN "[INFO ] " BASH_CLEAR "   1.0000   1.5000   2.0000\n"
      BASH_GREEN "[INFO ] " BASH_CLEAR "   2.5000   3.0000   3.5000\n"
      BASH_GREEN "[INFO ] " BASH_CLEAR "   4.0000   4.5000   5.0000\n");
}
