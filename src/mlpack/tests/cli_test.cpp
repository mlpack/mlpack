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

#include <iostream>
#include <sstream>

#ifndef _WIN32
  #include <sys/time.h>
#endif

// For Sleep().
#ifdef _WIN32
  #include <windows.h>
#endif

#include <mlpack/core.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

#define BASH_RED "\033[0;31m"
#define BASH_GREEN "\033[0;32m"
#define BASH_YELLOW "\033[0;33m"
#define BASH_CYAN "\033[0;36m"
#define BASH_CLEAR "\033[0m"

using namespace mlpack;
using namespace mlpack::util;

// When we run these tests, we have to nuke the existing CLI object that's
// created by default.
struct CLITestDestroyer
{
  CLITestDestroyer() { CLI::Destroy(); }
};

BOOST_FIXTURE_TEST_SUITE(CLITest, CLITestDestroyer);

/**
 * Before running a test that uses the CLI options, we have to add the default
 * options that are required for CLI to function, since it will be destroyed at
 * the end of every test that uses CLI in this test suite.
 */
void AddRequiredCLIOptions()
{
  CLI::Add<bool>(false, "help", "Default help info.", 'h');
  CLI::Add<std::string>("", "info", "Get help on a specific module or option.");
  CLI::Add<bool>(false, "verbose", "Display informational messages and the full"
      " list of parameters and timers at the end of execution.", 'v');
  CLI::Add<bool>(false, "version", "Display the version of mlpack.", 'V');
}

/**
 * Tests that CLI works as intended, namely that CLI::Add propagates
 * successfully.
 */
BOOST_AUTO_TEST_CASE(TestCLIAdd)
{
  AddRequiredCLIOptions();

  // Check that the CLI::HasParam returns false if no value has been specified
  // on the commandline and ignores any programmatical assignments.
  CLI::Add<bool>(false, "global/bool", "True or False", 'a');

  // CLI::HasParam should return false here.
  BOOST_REQUIRE(!CLI::HasParam("global/bool"));

  // Check that our aliasing works.
  BOOST_REQUIRE_EQUAL(CLI::HasParam("global/bool"),
      CLI::HasParam("a"));
  BOOST_REQUIRE_EQUAL(CLI::GetParam<bool>("global/bool"),
      CLI::GetParam<bool>("a"));

  CLI::Destroy();
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
  AddRequiredCLIOptions();

  // This test will involve creating an option, and making sure CLI reflects
  // this.
  PARAM_IN(int, "test_parent/test", "test desc", "", 42, false);

  BOOST_REQUIRE_EQUAL(CLI::GetParam<int>("test_parent/test"), 42);

  CLI::Destroy();
}

/**
 * Test that duplicate flags are filtered out correctly.
 */
BOOST_AUTO_TEST_CASE(TestDuplicateFlag)
{
  AddRequiredCLIOptions();

  PARAM_FLAG("test", "test", "t");

  int argc = 3;
  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "--test";
  argv[2] = "--test";

  // This should not throw an exception.
  CLI::ParseCommandLine(argc, const_cast<char**>(argv));
}

/**
 * Test that duplicate options throw an exception.
 */
BOOST_AUTO_TEST_CASE(TestDuplicateParam)
{
  AddRequiredCLIOptions();

  int argc = 5;
  const char* argv[5];
  argv[0] = "./test";
  argv[1] = "--info";
  argv[2] = "test1";
  argv[3] = "--info";
  argv[4] = "test2";

  // This should throw an exception.
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(CLI::ParseCommandLine(argc, const_cast<char**>(argv)),
      std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensure that a Boolean option which we define is set correctly.
 */
BOOST_AUTO_TEST_CASE(TestBooleanOption)
{
  AddRequiredCLIOptions();

  PARAM_FLAG("flag_test", "flag test description", "");

  BOOST_REQUIRE_EQUAL(CLI::HasParam("flag_test"), false);

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

  CLI::Destroy();
}

/**
 * Test that a vector option works correctly.
 */
BOOST_AUTO_TEST_CASE(TestVectorOption)
{
  AddRequiredCLIOptions();

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
  AddRequiredCLIOptions();

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

//  Log::Fatal.ignoreInput = true;
  CLI::ParseCommandLine(argc, const_cast<char**>(argv));
//  Log::Fatal.ignoreInput = false;

  BOOST_REQUIRE(CLI::HasParam("test2_vec"));

  std::vector<size_t> v = CLI::GetParam<std::vector<size_t>>("test2_vec");

  BOOST_REQUIRE_EQUAL(v.size(), 3);
  BOOST_REQUIRE_EQUAL(v[0], 1);
  BOOST_REQUIRE_EQUAL(v[1], 2);
  BOOST_REQUIRE_EQUAL(v[2], 4);

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
  AddRequiredCLIOptions();

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

  BOOST_REQUIRE_GE(Timer::Get("test_timer").count(), 10000);

  // Restart it.
  Timer::Start("test_timer");

  #ifdef _WIN32
  Sleep(10);
  #else
  usleep(10000);
  #endif

  Timer::Stop("test_timer");

  BOOST_REQUIRE_GE(Timer::Get("test_timer").count(), 20000);

  // Just one more time, for good measure...
  Timer::Start("test_timer");

  #ifdef _WIN32
  Sleep(20);
  #else
  usleep(20000);
  #endif

  Timer::Stop("test_timer");

  BOOST_REQUIRE_GE(Timer::Get("test_timer").count(), 40000);

  CLI::Destroy();
}

BOOST_AUTO_TEST_CASE(TwiceStartTimerTest)
{
  AddRequiredCLIOptions();

  Timer::Start("test_timer");

  BOOST_REQUIRE_THROW(Timer::Start("test_timer"), std::runtime_error);

  CLI::Destroy();
}

BOOST_AUTO_TEST_CASE(TwiceStopTimerTest)
{
  AddRequiredCLIOptions();

  Timer::Start("test_timer");
  Timer::Stop("test_timer");

  BOOST_REQUIRE_THROW(Timer::Stop("test_timer"), std::runtime_error);

  CLI::Destroy();
}

BOOST_AUTO_TEST_CASE(InputMatrixParamTest)
{
  AddRequiredCLIOptions();

  // --matrix is an input parameter; it won't be transposed.
  CLI::Add<arma::mat>(arma::mat(), "matrix", "Test matrix", 'm', false, true,
      false);

  // Set some fake arguments.
  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "-m";
  argv[2] = "test_data_3_1000.csv";

  int argc = 3;

  // The const-cast is a little hacky but should be fine...
  Log::Fatal.ignoreInput = true;
  CLI::ParseCommandLine(argc, const_cast<char**>(argv));
  Log::Fatal.ignoreInput = false;

  // The --matrix parameter should exist.
  BOOST_REQUIRE(CLI::HasParam("matrix"));
  // The --matrix_file parameter should not exist (it should be transparent from
  // inside the program).
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(CLI::HasParam("matrix_file"), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  arma::mat dataset = CLI::GetParam<arma::mat>("matrix");
  arma::mat dataset2 = CLI::GetParam<arma::mat>("matrix");

  BOOST_REQUIRE_EQUAL(dataset.n_rows, 3);
  BOOST_REQUIRE_EQUAL(dataset.n_cols, 1000);
  BOOST_REQUIRE_EQUAL(dataset2.n_rows, 3);
  BOOST_REQUIRE_EQUAL(dataset2.n_cols, 1000);

  for (size_t i = 0; i < dataset.n_elem; ++i)
    BOOST_REQUIRE_CLOSE(dataset[i], dataset2[i], 1e-10);

  // Clean it up.
  CLI::Destroy();
}

BOOST_AUTO_TEST_CASE(InputMatrixNoTransposeParamTest)
{
  AddRequiredCLIOptions();

  // --matrix is a non-transposed input parameter.
  CLI::Add<arma::mat>(arma::mat(), "matrix", "Test matrix", 'm', false, true,
      true);

  // Set some fake arguments.
  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "--matrix_file";
  argv[2] = "test_data_3_1000.csv";

  int argc = 3;

  // The const-cast is a little hacky but should be fine...
  CLI::ParseCommandLine(argc, const_cast<char**>(argv));

  // The --matrix parameter should exist.
  BOOST_REQUIRE(CLI::HasParam("matrix"));
  // The --matrix_file parameter should not exist (it should be transparent from
  // inside the program).
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(CLI::HasParam("matrix_file"), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  arma::mat dataset = CLI::GetParam<arma::mat>("matrix");
  arma::mat dataset2 = CLI::GetParam<arma::mat>("matrix");

  BOOST_REQUIRE_EQUAL(dataset.n_rows, 1000);
  BOOST_REQUIRE_EQUAL(dataset.n_cols, 3);
  BOOST_REQUIRE_EQUAL(dataset2.n_rows, 1000);
  BOOST_REQUIRE_EQUAL(dataset2.n_cols, 3);

  for (size_t i = 0; i < dataset.n_elem; ++i)
    BOOST_REQUIRE_CLOSE(dataset[i], dataset2[i], 1e-10);

  // Clean it up.
  CLI::Destroy();
}

BOOST_AUTO_TEST_CASE(OutputMatrixParamTest)
{
  AddRequiredCLIOptions();

  // --matrix is an output parameter.
  CLI::Add<arma::mat>(arma::mat(), "matrix", "Test matrix", 'm', false, false,
      false);

  // Set some fake arguments.
  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "-m";
  argv[2] = "test.csv";

  int argc = 3;

  CLI::ParseCommandLine(argc, const_cast<char**>(argv));

  // The --matrix parameter should exist.
  BOOST_REQUIRE(CLI::HasParam("matrix"));
  // The --matrix_file parameter should not exist (it should be transparent from
  // inside the program).
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(CLI::HasParam("matrix_file"), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  // Since it's an output parameter, we don't need any input and don't need to
  // call ParseCommandLine().
  arma::mat dataset = arma::randu<arma::mat>(3, 100);
  CLI::GetParam<arma::mat>("matrix") = dataset;

  // Write the file.
  CLI::Destroy();
  AddRequiredCLIOptions();

  // Now load the matrix back and make sure it was saved correctly.
  arma::mat dataset2;
  data::Load("test.csv", dataset2);

  BOOST_REQUIRE_EQUAL(dataset.n_cols, dataset2.n_cols);
  BOOST_REQUIRE_EQUAL(dataset.n_rows, dataset2.n_rows);
  for (size_t i = 0; i < dataset.n_elem; ++i)
    BOOST_REQUIRE_CLOSE(dataset[i], dataset2[i], 1e-10);

  // Remove the file.
  remove("test.csv");
  CLI::Destroy();
}

BOOST_AUTO_TEST_CASE(OutputMatrixNoTransposeParamTest)
{
  AddRequiredCLIOptions();

  // --matrix is an output parameter.
  CLI::Add<arma::mat>(arma::mat(), "matrix", "Test matrix", 'm', false, false,
      true);

  // Set some fake arguments.
  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "-m";
  argv[2] = "test.csv";

  int argc = 3;

  CLI::ParseCommandLine(argc, const_cast<char**>(argv));

  // The --matrix parameter should exist.
  BOOST_REQUIRE(CLI::HasParam("matrix"));
  // The --matrix_file parameter should not exist (it should be transparent from
  // inside the program).
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(CLI::HasParam("matrix_file"), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  // Since it's an output parameter, we don't need any input and don't need to
  // call ParseCommandLine().
  arma::mat dataset = arma::randu<arma::mat>(3, 100);
  CLI::GetParam<arma::mat>("matrix") = dataset;

  // Write the file.
  CLI::Destroy();
  AddRequiredCLIOptions();

  // Now load the matrix back and make sure it was saved correctly.
  arma::mat dataset2;
  data::Load("test.csv", dataset2, true, false);

  BOOST_REQUIRE_EQUAL(dataset.n_cols, dataset2.n_cols);
  BOOST_REQUIRE_EQUAL(dataset.n_rows, dataset2.n_rows);
  for (size_t i = 0; i < dataset.n_elem; ++i)
    BOOST_REQUIRE_CLOSE(dataset[i], dataset2[i], 1e-10);

  // Remove the file.
  remove("test.csv");
  CLI::Destroy();
}

BOOST_AUTO_TEST_CASE(IntParamTest)
{
  AddRequiredCLIOptions();

  CLI::Add<int>(0, "int", "Test int", 'i', false, true, false);

  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "-i";
  argv[2] = "3";

  int argc = 3;

  CLI::ParseCommandLine(argc, const_cast<char**>(argv));

  BOOST_REQUIRE(CLI::HasParam("int"));
  BOOST_REQUIRE_EQUAL(CLI::GetParam<int>("int"), 3);

  CLI::Destroy();
}

BOOST_AUTO_TEST_CASE(StringParamTest)
{
  AddRequiredCLIOptions();

  CLI::Add<std::string>("", "string", "Test string", 's', false, true, false);

  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "--string";
  argv[2] = "3";

  int argc = 3;

  CLI::ParseCommandLine(argc, const_cast<char**>(argv));

  BOOST_REQUIRE(CLI::HasParam("string"));
  BOOST_REQUIRE_EQUAL(CLI::GetParam<std::string>("string"), std::string("3"));

  CLI::Destroy();
}

BOOST_AUTO_TEST_CASE(DoubleParamTest)
{
  AddRequiredCLIOptions();

  CLI::Add<double>(0.0, "double", "Test double", 'd', false, true, false);

  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "--double";
  argv[2] = "3.12";

  int argc = 3;

  CLI::ParseCommandLine(argc, const_cast<char**>(argv));

  BOOST_REQUIRE(CLI::HasParam("double"));
  BOOST_REQUIRE_CLOSE(CLI::GetParam<double>("double"), 3.12, 1e-10);

  CLI::Destroy();
}

BOOST_AUTO_TEST_CASE(RequiredOptionTest)
{
  AddRequiredCLIOptions();

  CLI::Add<double>(0.0, "double", "Required test double", 'd', true, true,
      false);

  const char* argv[1];
  argv[0] = "./test";

  int argc = 1;

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(CLI::ParseCommandLine(argc, const_cast<char**>(argv)),
      std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

BOOST_AUTO_TEST_CASE(UnknownOptionTest)
{
  AddRequiredCLIOptions();

  const char* argv[2];
  argv[0] = "./test";
  argv[1] = "--unknown";

  int argc = 2;

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(CLI::ParseCommandLine(argc, const_cast<char**>(argv)),
      std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Test that GetUnmappedParam() works.
 */
BOOST_AUTO_TEST_CASE(UnmappedParamTest)
{
  AddRequiredCLIOptions();

  CLI::Add<arma::mat>(arma::mat(), "matrix", "Test matrix", 'm', false, true,
      true);
  CLI::Add<arma::mat>(arma::mat(), "matrix2", "Test matrix", 'M', false, false,
      true);
  CLI::Add<double>(0.0, "double", "Test double", 'd', false, true, false);
  CLI::Add<double>(0.0, "double2", "Test double", 'D', false, true, false);

  const char* argv[7];
  argv[0] = "./test";
  argv[1] = "--matrix_file";
  argv[2] = "file1.csv";
  argv[3] = "-M";
  argv[4] = "file2.csv";
  argv[5] = "-d";
  argv[6] = "1.334";

  int argc = 7;

  CLI::ParseCommandLine(argc, const_cast<char**>(argv));

  // Now check that we can get unmapped parameters.
  BOOST_REQUIRE_EQUAL(CLI::GetUnmappedParam<arma::mat>("matrix"), "file1.csv");
  BOOST_REQUIRE_EQUAL(CLI::GetUnmappedParam<arma::mat>("matrix2"), "file2.csv");
  BOOST_REQUIRE_CLOSE(CLI::GetUnmappedParam<double>("double"), 1.334, 1e-10);
  BOOST_REQUIRE_SMALL(CLI::GetUnmappedParam<double>("double2"), 1e-10);

  // Can we assign an unmapped parameter?
  CLI::GetUnmappedParam<arma::mat>("matrix2") =
      CLI::GetUnmappedParam<arma::mat>("matrix");

  BOOST_REQUIRE_EQUAL(CLI::GetUnmappedParam<arma::mat>("matrix2"), "file1.csv");
}

BOOST_AUTO_TEST_SUITE_END();
