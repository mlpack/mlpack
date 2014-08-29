/**
 * @file cli_test.cpp
 * @author Matthew Amidon, Ryan Curtin
 *
 * Test for the CLI input parameter system.
 *
 * This file is part of MLPACK 1.0.10.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <sstream>
#ifndef _WIN32
  #include <sys/time.h>
#endif

// For Sleep().
#ifdef _WIN32
  #include <Windows.h>
#endif

#include <mlpack/core.hpp>

#define DEFAULT_INT 42

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

#define BASH_RED "\033[0;31m"
#define BASH_GREEN "\033[0;32m"
#define BASH_YELLOW "\033[0;33m"
#define BASH_CYAN "\033[0;36m"
#define BASH_CLEAR "\033[0m"

using namespace mlpack;
using namespace mlpack::util;

BOOST_AUTO_TEST_SUITE(CLITest);

/**
 * Tests that CLI works as intended, namely that CLI::Add propagates
 * successfully.
 */
BOOST_AUTO_TEST_CASE(TestCLIAdd)
{
  // Check that the CLI::HasParam returns false if no value has been specified
  // on the commandline and ignores any programmatical assignments.
  CLI::Add<bool>("global/bool", "True or False", "alias/bool");

  // CLI::HasParam should return false here.
  BOOST_REQUIRE(!CLI::HasParam("global/bool"));

  // Check the description of our variable.
  BOOST_REQUIRE_EQUAL(CLI::GetDescription("global/bool").compare(
      std::string("True or False")) , 0);

  // Check that our aliasing works.
  BOOST_REQUIRE_EQUAL(CLI::HasParam("global/bool"),
      CLI::HasParam("alias/bool"));
  BOOST_REQUIRE_EQUAL(CLI::GetDescription("global/bool").compare(
      CLI::GetDescription("alias/bool")), 0);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<bool>("global/bool"),
      CLI::GetParam<bool>("alias/bool"));
}

/**
 * Test the output of CLI.  We will pass bogus input to a stringstream so that
 * none of it gets to the screen.
 */
BOOST_AUTO_TEST_CASE(TestPrefixedOutStreamBasic)
{
  std::stringstream ss;
  PrefixedOutStream pss(ss, BASH_GREEN "[INFO ] " BASH_CLEAR);

  pss << "This shouldn't break anything" << std::endl;
  BOOST_REQUIRE_EQUAL(ss.str(),
      BASH_GREEN "[INFO ] " BASH_CLEAR "This shouldn't break anything\n");

  ss.str("");
  pss << "Test the new lines...";
  pss << "shouldn't get 'Info' here." << std::endl;
  BOOST_REQUIRE_EQUAL(ss.str(),
      BASH_GREEN "[INFO ] " BASH_CLEAR
      "Test the new lines...shouldn't get 'Info' here.\n");

  pss << "But now I should." << std::endl << std::endl;
  pss << "";
  BOOST_REQUIRE_EQUAL(ss.str(),
      BASH_GREEN "[INFO ] " BASH_CLEAR
      "Test the new lines...shouldn't get 'Info' here.\n"
      BASH_GREEN "[INFO ] " BASH_CLEAR "But now I should.\n"
      BASH_GREEN "[INFO ] " BASH_CLEAR "\n"
      BASH_GREEN "[INFO ] " BASH_CLEAR "");
}

/**
 * Tests that the various PARAM_* macros work properly.
 */
BOOST_AUTO_TEST_CASE(TestOption)
{
  // This test will involve creating an option, and making sure CLI reflects
  // this.
  PARAM(int, "test_parent/test", "test desc", "", DEFAULT_INT, false);

  BOOST_REQUIRE_EQUAL(CLI::GetDescription("test_parent/test"), "test desc");
  BOOST_REQUIRE_EQUAL(CLI::GetParam<int>("test_parent/test"), DEFAULT_INT);
}

/**
 * Ensure that a Boolean option which we define is set correctly.
 */
BOOST_AUTO_TEST_CASE(TestBooleanOption)
{
  PARAM_FLAG("flag_test", "flag test description", "");

  BOOST_REQUIRE_EQUAL(CLI::HasParam("flag_test"), false);

  BOOST_REQUIRE_EQUAL(CLI::GetDescription("flag_test"),
      "flag test description");

  // Now check that CLI reflects that it is false by default.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<bool>("flag_test"), false);

  // Now, if we specify this flag, it should be true.
  int argc = 2;
  char* argv[2];
  argv[0] = strcpy(new char[strlen("programname")], "programname");
  argv[1] = strcpy(new char[strlen("--flag_test")], "--flag_test");

  CLI::ParseCommandLine(argc, argv);

  BOOST_REQUIRE_EQUAL(CLI::GetParam<bool>("flag_test"), true);
  BOOST_REQUIRE_EQUAL(CLI::HasParam("flag_test"), true);

  delete[] argv[0];
  delete[] argv[1];
}

/**
 * Test that we can correctly output Armadillo objects to PrefixedOutStream
 * objects.
 */
BOOST_AUTO_TEST_CASE(TestArmadilloPrefixedOutStream)
{
  // We will test this with both a vector and a matrix.
  arma::vec test("1.0 1.5 2.0 2.5 3.0 3.5 4.0");

  std::stringstream ss;
  PrefixedOutStream pss(ss, BASH_GREEN "[INFO ] " BASH_CLEAR);

  pss << test;
  // This should result in nothing being on the current line (since it clears
  // it).
  BOOST_REQUIRE_EQUAL(ss.str(), BASH_GREEN "[INFO ] " BASH_CLEAR "   1.0000\n"
      BASH_GREEN "[INFO ] " BASH_CLEAR "   1.5000\n"
      BASH_GREEN "[INFO ] " BASH_CLEAR "   2.0000\n"
      BASH_GREEN "[INFO ] " BASH_CLEAR "   2.5000\n"
      BASH_GREEN "[INFO ] " BASH_CLEAR "   3.0000\n"
      BASH_GREEN "[INFO ] " BASH_CLEAR "   3.5000\n"
      BASH_GREEN "[INFO ] " BASH_CLEAR "   4.0000\n");

  ss.str("");
  pss << trans(test);
  // This should result in there being stuff on the line.
  BOOST_REQUIRE_EQUAL(ss.str(), BASH_GREEN "[INFO ] " BASH_CLEAR
      "   1.0000   1.5000   2.0000   2.5000   3.0000   3.5000   4.0000\n");

  arma::mat test2("1.0 1.5 2.0; 2.5 3.0 3.5; 4.0 4.5 4.99999");
  ss.str("");
  pss << test2;
  BOOST_REQUIRE_EQUAL(ss.str(),
      BASH_GREEN "[INFO ] " BASH_CLEAR "   1.0000   1.5000   2.0000\n"
      BASH_GREEN "[INFO ] " BASH_CLEAR "   2.5000   3.0000   3.5000\n"
      BASH_GREEN "[INFO ] " BASH_CLEAR "   4.0000   4.5000   5.0000\n");

  // Try and throw a curveball by not clearing the line before outputting
  // something else.  The PrefixedOutStream should not force Armadillo objects
  // onto their own lines.
  ss.str("");
  pss << "hello" << test2;
  BOOST_REQUIRE_EQUAL(ss.str(),
      BASH_GREEN "[INFO ] " BASH_CLEAR "hello   1.0000   1.5000   2.0000\n"
      BASH_GREEN "[INFO ] " BASH_CLEAR "   2.5000   3.0000   3.5000\n"
      BASH_GREEN "[INFO ] " BASH_CLEAR "   4.0000   4.5000   5.0000\n");
}

/**
 * Test that we can correctly output things in general.
 */
BOOST_AUTO_TEST_CASE(TestPrefixedOutStream)
{
  std::stringstream ss;
  PrefixedOutStream pss(ss, BASH_GREEN "[INFO ] " BASH_CLEAR);

  pss << "hello world I am ";
  pss << 7;

  BOOST_REQUIRE_EQUAL(ss.str(),
      BASH_GREEN "[INFO ] " BASH_CLEAR "hello world I am 7");

  pss << std::endl;
  BOOST_REQUIRE_EQUAL(ss.str(),
      BASH_GREEN "[INFO ] " BASH_CLEAR "hello world I am 7\n");

  ss.str("");
  pss << std::endl;
  BOOST_REQUIRE_EQUAL(ss.str(),
      BASH_GREEN "[INFO ] " BASH_CLEAR "\n");
}

/**
 * Test format modifiers.
 */
BOOST_AUTO_TEST_CASE(TestPrefixedOutStreamModifiers)
{
  std::stringstream ss;
  PrefixedOutStream pss(ss, BASH_GREEN "[INFO ] " BASH_CLEAR);

  pss << "I have a precise number which is ";
  pss << std::setw(6) << std::setfill('0') << (int)156;

  BOOST_REQUIRE_EQUAL(ss.str(),
      BASH_GREEN "[INFO ] " BASH_CLEAR
      "I have a precise number which is 000156");
}

/**
 * We should be able to start and then stop a timer multiple times and it should
 * save the value.
 */
BOOST_AUTO_TEST_CASE(MultiRunTimerTest)
{
  Timer::Start("test_timer");

  // On Windows (or, at least, in Windows not using VS2010) we cannot use
  // usleep() because it is not provided.  Instead we will use Sleep() for a
  // number of milliseconds.
  #ifdef _WIN32
  Sleep(10);
  #else
  usleep(10000);
  #endif

  Timer::Stop("test_timer");

  BOOST_REQUIRE_GE(Timer::Get("test_timer").tv_usec, 10000);

  // Restart it.
  Timer::Start("test_timer");

  #ifdef _WIN32
  Sleep(10);
  #else
  usleep(10000);
  #endif

  Timer::Stop("test_timer");

  BOOST_REQUIRE_GE(Timer::Get("test_timer").tv_usec, 20000);

  // Just one more time, for good measure...
  Timer::Start("test_timer");

  #ifdef _WIN32
  Sleep(20);
  #else
  usleep(20000);
  #endif

  Timer::Stop("test_timer");

  BOOST_REQUIRE_GE(Timer::Get("test_timer").tv_usec, 40000);
}

BOOST_AUTO_TEST_SUITE_END();
