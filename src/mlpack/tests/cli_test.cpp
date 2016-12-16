/**
 * @file cli_test.cpp
 * @author Matthew Amidon, Ryan Curtin
 *
 * Test for the CLI input parameter system.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>

#define DEFAULT_INT 42

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

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
 * Tests that the various PARAM_* macros work properly.
 */
BOOST_AUTO_TEST_CASE(TestOption)
{
  // This test will involve creating an option, and making sure CLI reflects
  // this.
  PARAM_IN(int, "test_parent/test", "test desc", "", DEFAULT_INT, false);

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
  argv[0] = strcpy(new char[strlen("programname") + 1], "programname");
  argv[1] = strcpy(new char[strlen("--flag_test") + 1], "--flag_test");

  CLI::ParseCommandLine(argc, argv);

  BOOST_REQUIRE_EQUAL(CLI::GetParam<bool>("flag_test"), true);
  BOOST_REQUIRE_EQUAL(CLI::HasParam("flag_test"), true);

  delete[] argv[0];
  delete[] argv[1];
}

/**
 * Test that a vector option works correctly.
 */
BOOST_AUTO_TEST_CASE(TestVectorOption)
{
  PARAM_VECTOR_IN(size_t, "test_vec", "test description", "t");

  int argc = 5;
  const char* argv[5];
  argv[0] = "./test";
  argv[1] = "--test_vec";
  argv[2] = "1";
  argv[3] = "2";
  argv[4] = "4";

  Log::Fatal.ignoreInput = true;
  CLI::ParseCommandLine(argc, const_cast<char**>(argv));
  Log::Fatal.ignoreInput = false;

  BOOST_REQUIRE(CLI::HasParam("test_vec"));

  std::vector<size_t> v = CLI::GetParam<std::vector<size_t>>("test_vec");

  BOOST_REQUIRE_EQUAL(v.size(), 3);
  BOOST_REQUIRE_EQUAL(v[0], 1);
  BOOST_REQUIRE_EQUAL(v[1], 2);
  BOOST_REQUIRE_EQUAL(v[2], 4);
}

/**
 * Test that we can use a vector option by specifying it many times.
 */
BOOST_AUTO_TEST_CASE(TestVectorOption2)
{
  PARAM_VECTOR_IN(size_t, "test2_vec", "test description", "T");

  int argc = 7;
  const char* argv[7];
  argv[0] = "./test";
  argv[1] = "--test2_vec";
  argv[2] = "1";
  argv[3] = "--test2_vec";
  argv[4] = "2";
  argv[5] = "--test2_vec";
  argv[6] = "4";

  Log::Fatal.ignoreInput = true;
  CLI::ParseCommandLine(argc, const_cast<char**>(argv));
  Log::Fatal.ignoreInput = false;

  BOOST_REQUIRE(CLI::HasParam("test_vec"));

  std::vector<size_t> v = CLI::GetParam<std::vector<size_t>>("test_vec");

  BOOST_REQUIRE_EQUAL(v.size(), 3);
  BOOST_REQUIRE_EQUAL(v[0], 1);
  BOOST_REQUIRE_EQUAL(v[1], 2);
  BOOST_REQUIRE_EQUAL(v[2], 4);

}

BOOST_AUTO_TEST_SUITE_END();
