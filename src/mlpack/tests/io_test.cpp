/**
 * @file tests/io_test.cpp
 * @author Matthew Amidon, Ryan Curtin
 *
 * Test for the IO input parameter system.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
// We'll use CLIOptions.
#include <mlpack/bindings/cli/cli_option.hpp>
#include <mlpack/bindings/cli/third_party/CLI/CLI11.hpp>

namespace mlpack {
namespace util {

template<typename T>
using Option = mlpack::bindings::cli::CLIOption<T>;

} // namespace util
} // namespace mlpack

static const std::string testName = "";

#include <mlpack/core/util/param.hpp>
#include <mlpack/bindings/cli/parse_command_line.hpp>
#include <mlpack/bindings/cli/end_program.hpp>

#include "catch.hpp"

using namespace mlpack;
using namespace mlpack::util;
using namespace mlpack::kernel;
using namespace mlpack::data;
using namespace mlpack::bindings::cli;
using namespace std;

// When we run these tests, we have to nuke the existing CLI object that's
// created by default.
struct IOTestDestroyer
{
  IOTestDestroyer() { IO::ClearSettings(); }
};

/**
 * Before running a test that uses the CLI options, we have to add the default
 * options that are required for CLI to function, since it will be destroyed at
 * the end of every test that uses CLI in this test suite.
 */
void AddRequiredCLIOptions()
{
  IO::ClearSettings();

  // These will register with CLI immediately.
  CLIOption<bool> help(false, "help", "Default help info.", "h", "bool");
  CLIOption<string> info("", "info", "Get help on a specific module or option.",
      "", "string");
  CLIOption<bool> verbose(false, "verbose", "Display information messages and "
      "the full list of parameters and timers at the end of execution.", "v",
      "bool");
  CLIOption<bool> version(false, "version", "Display the version of mlpack.",
      "V", "bool");
}

/**
 * Tests that CLI works as intended, namely that IO::Add propagates
 * successfully.
 */
TEST_CASE_METHOD(IOTestDestroyer, "TestCLIAdd", "[IOTest]")
{
  AddRequiredCLIOptions();

  // Check that the IO::HasParam returns false if no value has been specified
  // on the commandline and ignores any programmatical assignments.
  CLIOption<bool> b(false, "global/bool", "True or false.", "a", "bool");

  // IO::HasParam should return false here.
  REQUIRE(!IO::HasParam("global/bool"));

  // Check that our aliasing works.
  REQUIRE(IO::HasParam("global/bool") == IO::HasParam("a"));
  REQUIRE(IO::GetParam<bool>("global/bool") == IO::GetParam<bool>("a"));
}

/**
 * Tests that the various PARAM_* macros work properly.
 */
TEST_CASE_METHOD(IOTestDestroyer, "TestOption", "[IOTest]")
{
  AddRequiredCLIOptions();

  // This test will involve creating an option, and making sure CLI reflects
  // this.
  PARAM_IN(int, "test_parent/test", "test desc", "", 42, false);

  REQUIRE(IO::GetParam<int>("test_parent/test") == 42);
}

/**
 * Test that duplicate flags are filtered out correctly.
 */
TEST_CASE_METHOD(IOTestDestroyer, "TestDuplicateFlag", "[IOTest]")
{
  AddRequiredCLIOptions();

  PARAM_FLAG("test", "test", "t");

  int argc = 3;
  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "--test";
  argv[2] = "--test";

  // This should not throw an exception.
  REQUIRE_NOTHROW(
      ParseCommandLine(argc, const_cast<char**>(argv)));
}

/**
 * Test that duplicate options throw an exception.
 */
TEST_CASE_METHOD(IOTestDestroyer, "TestDuplicateParam",
                "[IOTest]")
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
  REQUIRE_THROWS_AS(ParseCommandLine(argc, const_cast<char**>(argv)),
      std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensure that a Boolean option which we define is set correctly.
 */
TEST_CASE_METHOD(IOTestDestroyer, "TestBooleanOption",
                "[IOTest]")
{
  AddRequiredCLIOptions();

  PARAM_FLAG("flag_test", "flag test description", "");

  REQUIRE(IO::HasParam("flag_test") == false);

  // Now check that CLI reflects that it is false by default.
  REQUIRE(IO::GetParam<bool>("flag_test") == false);

  // Now, if we specify this flag, it should be true.
  int argc = 2;
  const char* argv[2];
  argv[0] = "programname";
  argv[1] = "--flag_test";

  ParseCommandLine(argc, const_cast<char**>(argv));

  REQUIRE(IO::GetParam<bool>("flag_test") == true);
  REQUIRE(IO::HasParam("flag_test") == true);
}

/**
 * Test that a vector option works correctly.
 */
TEST_CASE_METHOD(IOTestDestroyer, "TestVectorOption",
                "[IOTest]")
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

  ParseCommandLine(argc, const_cast<char**>(argv));

  REQUIRE(IO::HasParam("test_vec"));

  vector<size_t> v = IO::GetParam<vector<size_t>>("test_vec");

  REQUIRE(v.size() == 3);
  REQUIRE(v[0] == 1);
  REQUIRE(v[1] == 2);
  REQUIRE(v[2] == 4);
}

/**
 * Test that we can use a vector option by specifying it many times.
 */
TEST_CASE_METHOD(IOTestDestroyer, "TestVectorOption2",
                "[IOTest]")
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

  ParseCommandLine(argc, const_cast<char**>(argv));

  REQUIRE(IO::HasParam("test2_vec"));

  vector<size_t> v = IO::GetParam<vector<size_t>>("test2_vec");

  REQUIRE(v.size() == 3);
  REQUIRE(v[0] == 1);
  REQUIRE(v[1] == 2);
  REQUIRE(v[2] == 4);
}

TEST_CASE_METHOD(IOTestDestroyer, "InputColVectorParamTest",
                "[IOTest]")
{
  AddRequiredCLIOptions();

  PARAM_COL_IN("vector", "Test vector", "l");

  // Fake arguments.
  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "-l";
  argv[2] = "iris_test_labels.csv";

  int argc = 3;

  // The const-cast is a little hacky but should be fine...
  ParseCommandLine(argc, const_cast<char**>(argv));

  // The --vector parameter should exist.
  REQUIRE(IO::HasParam("vector"));
  // The --vector_file parameter should not exist (it should be transparent from
  // inside the program).
  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(IO::HasParam("vector_file"), runtime_error);
  Log::Fatal.ignoreInput = false;

  arma::vec vec1 = IO::GetParam<arma::vec>("vector");
  arma::vec vec2 = IO::GetParam<arma::vec>("vector");

  REQUIRE(vec1.n_rows == 63);
  REQUIRE(vec2.n_rows == 63);

  for (size_t i = 0; i < vec1.n_elem; ++i)
    REQUIRE(vec1[i] == Approx(vec2[i]).epsilon(1e-12));
}

TEST_CASE_METHOD(IOTestDestroyer, "InputUnsignedColVectorParamTest",
                "[IOTest]")
{
  AddRequiredCLIOptions();

  PARAM_UCOL_IN("vector", "Test vector", "l");

  // Fake arguments.
  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "-l";
  argv[2] = "iris_test_labels.csv";

  int argc = 3;

  // The const-cast is a little hacky but should be fine...
  ParseCommandLine(argc, const_cast<char**>(argv));

  // The --vector parameter should exist.
  REQUIRE(IO::HasParam("vector"));
  // The --vector_file parameter should not exist (it should be transparent from
  // inside the program).
  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(IO::HasParam("vector_file"), runtime_error);
  Log::Fatal.ignoreInput = false;

  arma::Col<size_t> vec1 = IO::GetParam<arma::Col<size_t>>("vector");
  arma::Col<size_t> vec2 = IO::GetParam<arma::Col<size_t>>("vector");

  REQUIRE(vec1.n_rows == 63);
  REQUIRE(vec2.n_rows == 63);

  for (size_t i = 0; i < vec1.n_elem; ++i)
    REQUIRE(vec1[i] == vec2[i]);
}

TEST_CASE_METHOD(IOTestDestroyer, "InputRowVectorParamTest",
                "[IOTest]")
{
  AddRequiredCLIOptions();

  PARAM_ROW_IN("row", "Test vector", "l");

  // Fake arguments.
  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "-l";
  argv[2] = "testRes.csv";

  int argc = 3;

  // The const-cast is a little hacky but should be fine...
  ParseCommandLine(argc, const_cast<char**>(argv));

  // The --vector parameter should exist.
  REQUIRE(IO::HasParam("row"));
  // The --vector_file parameter should not exist (it should be transparent from
  // inside the program).
  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(IO::HasParam("row_file"), runtime_error);
  Log::Fatal.ignoreInput = false;

  arma::rowvec vec1 = IO::GetParam<arma::rowvec>("row");
  arma::rowvec vec2 = IO::GetParam<arma::rowvec>("row");

  REQUIRE(vec1.n_cols == 7);
  REQUIRE(vec2.n_cols == 7);

  for (size_t i = 0; i < vec1.n_elem; ++i)
    REQUIRE(vec1[i] == Approx(vec2[i]).epsilon(1e-12));
}

TEST_CASE_METHOD(IOTestDestroyer, "InputUnsignedRowVectorParamTest",
                "[IOTest]")
{
  AddRequiredCLIOptions();

  PARAM_UROW_IN("row", "Test vector", "l");

  // Fake arguments.
  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "-l";
  argv[2] = "testRes.csv";

  int argc = 3;

  // The const-cast is a little hacky but should be fine...
  ParseCommandLine(argc, const_cast<char**>(argv));

  // The --vector parameter should exist.
  REQUIRE(IO::HasParam("row"));
  // The --vector_file parameter should not exist (it should be transparent from
  // inside the program).
  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(IO::HasParam("row_file"), runtime_error);
  Log::Fatal.ignoreInput = false;

  arma::Row<size_t> vec1 = IO::GetParam<arma::Row<size_t>>("row");
  arma::Row<size_t> vec2 = IO::GetParam<arma::Row<size_t>>("row");

  REQUIRE(vec1.n_cols == 7);
  REQUIRE(vec2.n_cols == 7);

  for (size_t i = 0; i < vec1.n_elem; ++i)
    REQUIRE(vec1[i] == vec2[i]);
}

TEST_CASE_METHOD(IOTestDestroyer, "OutputColParamTest",
                "[IOTest]")
{
  AddRequiredCLIOptions();

  // --vector is an output parameter.
  PARAM_COL_OUT("vector", "Test vector", "l");

  // Set some fake arguments.
  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "-l";
  argv[2] = "test.csv";

  int argc = 3;

  ParseCommandLine(argc, const_cast<char**>(argv));

  // The --vector parameter should exist.
  REQUIRE(IO::HasParam("vector"));
  // The --vector_file parameter should not exist (it should be transparent from
  // inside the program).
  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(IO::HasParam("vector_file"), runtime_error);
  Log::Fatal.ignoreInput = false;

  // Since it's an output parameter, we don't need any input and don't need to
  // call ParseCommandLine().
  arma::vec dataset = arma::randu<arma::vec>(100);
  IO::GetParam<arma::vec>("vector") = dataset;

  // Write the file.
  EndProgram();
  IO::ClearSettings();
  AddRequiredCLIOptions();

  // Now load the vector back and make sure it was saved correctly.
  arma::vec dataset2;
  data::Load("test.csv", dataset2);

  REQUIRE(dataset.n_rows == dataset2.n_rows);
  for (size_t i = 0; i < dataset.n_elem; ++i)
    REQUIRE(dataset[i] == Approx(dataset2[i]).epsilon(1e-12));

  // Remove the file.
  remove("test.csv");
}

TEST_CASE_METHOD(IOTestDestroyer, "OutputUnsignedColParamTest",
                "[IOTest]")
{
  AddRequiredCLIOptions();

  // --vector is an output parameter.
  PARAM_UCOL_OUT("vector", "Test vector", "l");

  // Set some fake arguments.
  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "-l";
  argv[2] = "test.csv";

  int argc = 3;

  ParseCommandLine(argc, const_cast<char**>(argv));

  // The --vector parameter should exist.
  REQUIRE(IO::HasParam("vector"));
  // The --vector_file parameter should not exist (it should be transparent from
  // inside the program).
  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(IO::HasParam("vector_file"), runtime_error);
  Log::Fatal.ignoreInput = false;

  // Since it's an output parameter, we don't need any input and don't need to
  // call ParseCommandLine().
  arma::Col<size_t> dataset = arma::randi<arma::Col<size_t>>(100);
  IO::GetParam<arma::Col<size_t>>("vector") = dataset;

  // Write the file.
  EndProgram();
  IO::ClearSettings();
  AddRequiredCLIOptions();

  // Now load the vector back and make sure it was saved correctly.
  arma::Col<size_t> dataset2;
  data::Load("test.csv", dataset2);

  REQUIRE(dataset.n_rows == dataset2.n_rows);
  for (size_t i = 0; i < dataset.n_elem; ++i)
    REQUIRE(dataset[i] == dataset2[i]);

  // Remove the file.
  remove("test.csv");
}

TEST_CASE_METHOD(IOTestDestroyer, "OutputRowParamTest",
                "[IOTest]")
{
  AddRequiredCLIOptions();

  // --row is an output parameter.
  PARAM_ROW_OUT("row", "Test vector", "l");

  // Set some fake arguments.
  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "-l";
  argv[2] = "test.csv";

  int argc = 3;

  ParseCommandLine(argc, const_cast<char**>(argv));

  // The --row parameter should exist.
  REQUIRE(IO::HasParam("row"));
  // The --row_file parameter should not exist (it should be transparent from
  // inside the program).
  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(IO::HasParam("row_file"), runtime_error);
  Log::Fatal.ignoreInput = false;

  // Since it's an output parameter, we don't need any input and don't need to
  // call ParseCommandLine().
  arma::rowvec dataset = arma::randu<arma::rowvec>(100);
  IO::GetParam<arma::rowvec>("row") = dataset;

  // Write the file.
  EndProgram();
  IO::ClearSettings();
  AddRequiredCLIOptions();

  // Now load the row vector back and make sure it was saved correctly.
  arma::rowvec dataset2;
  data::Load("test.csv", dataset2);

  REQUIRE(dataset.n_cols == dataset2.n_cols);
  for (size_t i = 0; i < dataset.n_elem; ++i)
    REQUIRE(dataset[i] == Approx(dataset2[i]).epsilon(1e-12));

  // Remove the file.
  remove("test.csv");
}

TEST_CASE_METHOD(IOTestDestroyer, "OutputUnsignedRowParamTest", "[IOTest]")
{
  AddRequiredCLIOptions();

  // --row is an output parameter.
  PARAM_UROW_OUT("row", "Test vector", "l");

  // Set some fake arguments.
  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "-l";
  argv[2] = "test.csv";

  int argc = 3;

  ParseCommandLine(argc, const_cast<char**>(argv));

  // The --row parameter should exist.
  REQUIRE(IO::HasParam("row"));
  // The --row_file parameter should not exist (it should be transparent from
  // inside the program).
  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(IO::HasParam("row_file"), runtime_error);
  Log::Fatal.ignoreInput = false;

  // Since it's an output parameter, we don't need any input and don't need to
  // call ParseCommandLine().
  arma::Row<size_t> dataset = arma::randi<arma::Row<size_t>>(100);
  IO::GetParam<arma::Row<size_t>>("row") = dataset;

  // Write the file.
  EndProgram();
  IO::ClearSettings();
  AddRequiredCLIOptions();

  // Now load the row vector back and make sure it was saved correctly.
  arma::Row<size_t> dataset2;
  data::Load("test.csv", dataset2);

  REQUIRE(dataset.n_cols == dataset2.n_cols);
  for (size_t i = 0; i < dataset.n_elem; ++i)
    REQUIRE(dataset[i] == dataset2[i]);

  // Remove the file.
  remove("test.csv");
}

TEST_CASE_METHOD(IOTestDestroyer, "InputMatrixParamTest",
                "[IOTest]")
{
  AddRequiredCLIOptions();

  // --matrix is an input parameter; it won't be transposed.
  PARAM_MATRIX_IN("matrix", "Test matrix", "m");

  // Set some fake arguments.
  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "-m";
  argv[2] = "test_data_3_1000.csv";

  int argc = 3;

  // The const-cast is a little hacky but should be fine...
  ParseCommandLine(argc, const_cast<char**>(argv));

  // The --matrix parameter should exist.
  REQUIRE(IO::HasParam("matrix"));
  // The --matrix_file parameter should not exist (it should be transparent from
  // inside the program).
  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(IO::HasParam("matrix_file"), runtime_error);
  Log::Fatal.ignoreInput = false;

  arma::mat dataset = IO::GetParam<arma::mat>("matrix");
  arma::mat dataset2 = IO::GetParam<arma::mat>("matrix");

  REQUIRE(dataset.n_rows == 3);
  REQUIRE(dataset.n_cols == 1000);
  REQUIRE(dataset2.n_rows == 3);
  REQUIRE(dataset2.n_cols == 1000);

  for (size_t i = 0; i < dataset.n_elem; ++i)
    REQUIRE(dataset[i] == Approx(dataset2[i]).epsilon(1e-12));
}

// Make sure we can correctly load required matrix parameters.
TEST_CASE_METHOD(IOTestDestroyer, "RequiredInputMatrixParamTest",
                "[IOTest]")
{
  AddRequiredCLIOptions();

  // --matrix is an input parameter; it won't be transposed.
  PARAM_MATRIX_IN_REQ("matrix", "Test matrix", "m");

  // Set some fake arguments.
  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "--matrix_file";
  argv[2] = "test_data_3_1000.csv";

  int argc = 3;

  // The const-cast is a little hacky but should be fine...
  ParseCommandLine(argc, const_cast<char**>(argv));

  // The --matrix parameter should exist.
  REQUIRE(IO::HasParam("matrix"));
  // The --matrix_file parameter should not exist (it should be transparent from
  // inside the program).
  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(IO::HasParam("matrix_file"), runtime_error);
  Log::Fatal.ignoreInput = false;

  arma::mat dataset = IO::GetParam<arma::mat>("matrix");
  arma::mat dataset2 = IO::GetParam<arma::mat>("matrix");

  REQUIRE(dataset.n_rows == 3);
  REQUIRE(dataset.n_cols == 1000);
  REQUIRE(dataset2.n_rows == 3);
  REQUIRE(dataset2.n_cols == 1000);

  for (size_t i = 0; i < dataset.n_elem; ++i)
    REQUIRE(dataset[i] == Approx(dataset2[i]).epsilon(1e-12));
}

// Make sure loading required matrix options by alias succeeds.
TEST_CASE_METHOD(IOTestDestroyer, "RequiredInputMatrixParamAliasTest",
                "[IOTest]")
{
  AddRequiredCLIOptions();

  // --matrix is an input parameter; it won't be transposed.
  PARAM_MATRIX_IN_REQ("matrix", "Test matrix", "m");

  // Set some fake arguments.
  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "-m";
  argv[2] = "test_data_3_1000.csv";

  int argc = 3;

  // The const-cast is a little hacky but should be fine...
  ParseCommandLine(argc, const_cast<char**>(argv));

  // The --matrix parameter should exist.
  REQUIRE(IO::HasParam("matrix"));
  // The --matrix_file parameter should not exist (it should be transparent from
  // inside the program).
  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(IO::HasParam("matrix_file"), runtime_error);
  Log::Fatal.ignoreInput = false;

  arma::mat dataset = IO::GetParam<arma::mat>("matrix");
  arma::mat dataset2 = IO::GetParam<arma::mat>("matrix");

  REQUIRE(dataset.n_rows == 3);
  REQUIRE(dataset.n_cols == 1000);
  REQUIRE(dataset2.n_rows == 3);
  REQUIRE(dataset2.n_cols == 1000);

  for (size_t i = 0; i < dataset.n_elem; ++i)
    REQUIRE(dataset[i] == Approx(dataset2[i]).epsilon(1e-12));
}

// Make sure that when we don't pass a required matrix, parsing fails.
TEST_CASE_METHOD(IOTestDestroyer, "RequiredUnspecifiedInputMatrixParamTest",
                "[IOTest]")
{
  AddRequiredCLIOptions();

  // --matrix is an input parameter; it won't be transposed.
  PARAM_MATRIX_IN_REQ("matrix", "Test matrix", "m");

  // Set some fake arguments.
  const char* argv[1];
  argv[0] = "./test";

  int argc = 1;

  // The const-cast is a little hacky but should be fine...
  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(ParseCommandLine(argc, const_cast<char**>(argv)),
      std::exception);
  Log::Fatal.ignoreInput = false;
}

TEST_CASE_METHOD(IOTestDestroyer, "InputMatrixNoTransposeParamTest",
                "[IOTest]")
{
  AddRequiredCLIOptions();

  // --matrix is a non-transposed input parameter.
  PARAM_TMATRIX_IN("matrix", "Test matrix", "m");

  // Set some fake arguments.
  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "--matrix_file";
  argv[2] = "test_data_3_1000.csv";

  int argc = 3;

  // The const-cast is a little hacky but should be fine...
  ParseCommandLine(argc, const_cast<char**>(argv));

  // The --matrix parameter should exist.
  REQUIRE(IO::HasParam("matrix"));
  // The --matrix_file parameter should not exist (it should be transparent from
  // inside the program).
  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(IO::HasParam("matrix_file"), runtime_error);
  Log::Fatal.ignoreInput = false;

  arma::mat dataset = IO::GetParam<arma::mat>("matrix");
  arma::mat dataset2 = IO::GetParam<arma::mat>("matrix");

  REQUIRE(dataset.n_rows == 1000);
  REQUIRE(dataset.n_cols == 3);
  REQUIRE(dataset2.n_rows == 1000);
  REQUIRE(dataset2.n_cols == 3);

  for (size_t i = 0; i < dataset.n_elem; ++i)
    REQUIRE(dataset[i] == Approx(dataset2[i]).epsilon(1e-12));
}

TEST_CASE_METHOD(IOTestDestroyer, "OutputMatrixParamTest",
                "[IOTest]")
{
  AddRequiredCLIOptions();

  // --matrix is an output parameter.
  PARAM_MATRIX_OUT("matrix", "Test matrix", "m");

  // Set some fake arguments.
  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "-m";
  argv[2] = "test.csv";

  int argc = 3;

  ParseCommandLine(argc, const_cast<char**>(argv));

  // The --matrix parameter should exist.
  REQUIRE(IO::HasParam("matrix"));
  // The --matrix_file parameter should not exist (it should be transparent from
  // inside the program).
  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(IO::HasParam("matrix_file"), runtime_error);
  Log::Fatal.ignoreInput = false;

  // Since it's an output parameter, we don't need any input and don't need to
  // call ParseCommandLine().
  arma::mat dataset = arma::randu<arma::mat>(3, 100);
  IO::GetParam<arma::mat>("matrix") = dataset;

  // Write the file.
  EndProgram();
  IO::ClearSettings();
  AddRequiredCLIOptions();

  // Now load the matrix back and make sure it was saved correctly.
  arma::mat dataset2;
  data::Load("test.csv", dataset2);

  REQUIRE(dataset.n_cols == dataset2.n_cols);
  REQUIRE(dataset.n_rows == dataset2.n_rows);
  for (size_t i = 0; i < dataset.n_elem; ++i)
    REQUIRE(dataset[i] == Approx(dataset2[i]).epsilon(1e-12));

  // Remove the file.
  remove("test.csv");
}

TEST_CASE_METHOD(IOTestDestroyer, "OutputMatrixNoTransposeParamTest", "[IOTest]")
{
  AddRequiredCLIOptions();

  // --matrix is an output parameter.
  PARAM_TMATRIX_OUT("matrix", "Test matrix", "m");

  // Set some fake arguments.
  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "-m";
  argv[2] = "test.csv";

  int argc = 3;

  ParseCommandLine(argc, const_cast<char**>(argv));

  // The --matrix parameter should exist.
  REQUIRE(IO::HasParam("matrix"));
  // The --matrix_file parameter should not exist (it should be transparent from
  // inside the program).
  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(IO::HasParam("matrix_file"), runtime_error);
  Log::Fatal.ignoreInput = false;

  // Since it's an output parameter, we don't need any input and don't need to
  // call ParseCommandLine().
  arma::mat dataset = arma::randu<arma::mat>(3, 100);
  IO::GetParam<arma::mat>("matrix") = dataset;

  // Write the file.
  EndProgram();
  IO::ClearSettings();
  AddRequiredCLIOptions();

  // Now load the matrix back and make sure it was saved correctly.
  arma::mat dataset2;
  data::Load("test.csv", dataset2, true, false);

  REQUIRE(dataset.n_cols == dataset2.n_cols);
  REQUIRE(dataset.n_rows == dataset2.n_rows);
  for (size_t i = 0; i < dataset.n_elem; ++i)
    REQUIRE(dataset[i] == Approx(dataset2[i]).epsilon(1e-12));

  // Remove the file.
  remove("test.csv");
}

TEST_CASE_METHOD(IOTestDestroyer, "IntParamTest",
                "[IOTest]")
{
  AddRequiredCLIOptions();

  PARAM_INT_IN("int", "Test int", "i", 0);

  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "-i";
  argv[2] = "3";

  int argc = 3;

  ParseCommandLine(argc, const_cast<char**>(argv));

  REQUIRE(IO::HasParam("int"));
  REQUIRE(IO::GetParam<int>("int") == 3);
}

TEST_CASE_METHOD(IOTestDestroyer, "StringParamTest",
                "[IOTest]")
{
  AddRequiredCLIOptions();

  PARAM_STRING_IN("string", "Test string", "s", "");

  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "--string";
  argv[2] = "3";

  int argc = 3;

  ParseCommandLine(argc, const_cast<char**>(argv));

  REQUIRE(IO::HasParam("string"));
  REQUIRE(IO::GetParam<string>("string") == string("3"));
}

TEST_CASE_METHOD(IOTestDestroyer, "DoubleParamTest",
                "[IOTest]")
{
  AddRequiredCLIOptions();

  PARAM_DOUBLE_IN("double", "Test double", "d", 0.0);

  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "--double";
  argv[2] = "3.12";

  int argc = 3;

  ParseCommandLine(argc, const_cast<char**>(argv));

  REQUIRE(IO::HasParam("double"));
  REQUIRE(IO::GetParam<double>("double") == Approx(3.12).epsilon(1e-12));
}

TEST_CASE_METHOD(IOTestDestroyer, "RequiredOptionTest", "[IOTest]")
{
  AddRequiredCLIOptions();

  PARAM_DOUBLE_IN_REQ("double", "Required test double", "d");

  const char* argv[1];
  argv[0] = "./test";

  int argc = 1;

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(ParseCommandLine(argc, const_cast<char**>(argv)),
      runtime_error);
  Log::Fatal.ignoreInput = false;
}

TEST_CASE_METHOD(IOTestDestroyer, "UnknownOptionTest",
                "[IOTest]")
{
  AddRequiredCLIOptions();

  const char* argv[2];
  argv[0] = "./test";
  argv[1] = "--unknown";

  int argc = 2;

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(ParseCommandLine(argc, const_cast<char**>(argv)),
      runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Test that GetPrintableParam() works.
 */
TEST_CASE_METHOD(IOTestDestroyer, "UnmappedParamTest",
                "[IOTest]")
{
  AddRequiredCLIOptions();

  PARAM_MATRIX_IN("matrix", "Test matrix", "m");
  PARAM_MATRIX_OUT("matrix2", "Test matrix", "M");
  PARAM_MODEL_IN(GaussianKernel, "kernel", "Test kernel", "k");
  PARAM_MODEL_OUT(GaussianKernel, "kernel2", "Test kernel", "K");

  const char* argv[9];
  argv[0] = "./test";
  argv[1] = "--matrix_file";
  argv[2] = "test_data_3_1000.csv";
  argv[3] = "-M";
  argv[4] = "file2.csv";
  argv[5] = "-k";
  argv[6] = "kernel.txt";
  argv[7] = "-K";
  argv[8] = "kernel2.txt";

  int argc = 9;

  ParseCommandLine(argc, const_cast<char**>(argv));

  // Now check that we can get unmapped parameters.
  REQUIRE(IO::GetPrintableParam<arma::mat>("matrix") ==
      "'test_data_3_1000.csv' (3x1000 matrix)");
  // This will have size 0x0 since it's an output parameter, and it hasn't been
  // set since ParseCommandLine() was called.
  REQUIRE(IO::GetPrintableParam<arma::mat>("matrix2") ==
      "'file2.csv' (0x0 matrix)");
  REQUIRE(IO::GetPrintableParam<GaussianKernel*>("kernel") ==
      "kernel.txt");
  REQUIRE(IO::GetPrintableParam<GaussianKernel*>("kernel2") ==
      "kernel2.txt");

  remove("kernel.txt");
}

/**
 * Test that we can serialize a model and then deserialize it through the CLI
 * interface.
 */
TEST_CASE_METHOD(IOTestDestroyer, "IOSerializationTest", "[IOTest]")
{
  AddRequiredCLIOptions();

  PARAM_MODEL_OUT(GaussianKernel, "kernel", "Test kernel", "k");

  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "--kernel_file";
  argv[2] = "kernel.txt";

  int argc = 3;

  ParseCommandLine(argc, const_cast<char**>(argv));

  // Create the kernel we'll save.
  GaussianKernel* gk = new GaussianKernel(0.5);

  IO::GetParam<GaussianKernel*>("kernel") = gk;

  // Save it.
  EndProgram();
  IO::ClearSettings();

  // Now create a new CLI object and load it.
  AddRequiredCLIOptions();

  PARAM_MODEL_IN(GaussianKernel, "kernel", "Test kernel", "k");

  ParseCommandLine(argc, const_cast<char**>(argv));

  // Load the kernel from file.
  GaussianKernel* gk2 = IO::GetParam<GaussianKernel*>("kernel");

  REQUIRE(gk2->Bandwidth() == Approx(0.5).epsilon(1e-7));

  // Clean up the memory...
  delete gk2;

  // Now remove the file we made.
  remove("kernel.txt");
}

/**
 * Test that an exception is thrown when a required model is not specified.
 */
TEST_CASE_METHOD(IOTestDestroyer, "RequiredModelTest",
                "[IOTest]")
{
  AddRequiredCLIOptions();

  PARAM_MODEL_IN_REQ(GaussianKernel, "kernel", "Test kernel", "k");

  // Don't specify any input parameters.
  const char* argv[1];
  argv[0] = "./test";

  int argc = 1;

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(ParseCommandLine(argc, const_cast<char**>(argv)),
      runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Test that we can load both a dataset and its associated info.
 */
TEST_CASE_METHOD(IOTestDestroyer, "MatrixAndDatasetInfoTest",
                "[IOTest]")
{
  AddRequiredCLIOptions();

  // Write test file to load.
  fstream f;
  f.open("test.arff", fstream::out);
  f << "@relation test" << endl;
  f << endl;
  f << "@attribute one STRING" << endl;
  f << "@attribute two REAL" << endl;
  f << endl;
  f << "@attribute three STRING" << endl;
  f << endl;
  f << "%% a comment line " << endl;
  f << endl;
  f << "@data" << endl;
  f << "hello, 1, moo" << endl;
  f << "cheese, 2.34, goodbye" << endl;
  f << "seven, 1.03e+5, moo" << endl;
  f << "hello, -1.3, goodbye" << endl;
  f.close();

  // Add options.
  typedef tuple<DatasetInfo, arma::mat> TupleType;
  PARAM_MATRIX_AND_INFO_IN("dataset", "Test dataset", "d");

  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "--dataset_file";
  argv[2] = "test.arff";

  int argc = 3;

  ParseCommandLine(argc, const_cast<char**>(argv));

  // Get the dataset and info.
  DatasetInfo info = move(get<0>(IO::GetParam<TupleType>("dataset")));
  arma::mat dataset = move(get<1>(IO::GetParam<TupleType>("dataset")));

  REQUIRE(info.Dimensionality() == 3);

  REQUIRE(info.Type(0) == Datatype::categorical);
  REQUIRE(info.NumMappings(0) == 3);
  REQUIRE(info.Type(1) == Datatype::numeric);
  REQUIRE(info.Type(2) == Datatype::categorical);
  REQUIRE(info.NumMappings(2) == 2);

  REQUIRE(dataset.n_rows == 3);
  REQUIRE(dataset.n_cols == 4);

  // The first dimension must all be different (except the ones that are the
  // same).
  REQUIRE(dataset(0, 0) == dataset(0, 3));
  REQUIRE(dataset(0, 0) != dataset(0, 1));
  REQUIRE(dataset(0, 1) != dataset(0, 2));
  REQUIRE(dataset(0, 2) != dataset(0, 0));

  REQUIRE(dataset(1, 0) == Approx(1.0).epsilon(1e-7));
  REQUIRE(dataset(1, 1) == Approx(2.34).epsilon(1e-7));
  REQUIRE(dataset(1, 2) == Approx(1.03e5).epsilon(1e-7));
  REQUIRE(dataset(1, 3) == Approx(-1.3).epsilon(1e-7));

  REQUIRE(dataset(2, 0) == dataset(2, 2));
  REQUIRE(dataset(2, 1) == dataset(2, 3));
  REQUIRE(dataset(2, 0) != dataset(2, 1));

  remove("test.arff");
}

/**
 * Test that we can access a parameter before we load it.
 */
TEST_CASE_METHOD(IOTestDestroyer, "RawIntegralParameter", "[IOTest]")
{
  AddRequiredCLIOptions();

  PARAM_DOUBLE_IN("double", "Test double", "d", 0.0);

  const char* argv[1];
  argv[0] = "./test";
  int argc = 1;

  ParseCommandLine(argc, const_cast<char**>(argv));

  // Set the double.
  IO::GetRawParam<double>("double") = 3.0;

  // Now when we get it, it should be what we just set it to.
  REQUIRE(IO::GetParam<double>("double") == Approx(3.0).epsilon(1e-7));
}

/**
 * Test that we can load a dataset with a pre-set mapping through
 * IO::GetRawParam().
 */
TEST_CASE_METHOD(IOTestDestroyer, "RawDatasetInfoLoadParameter", "[IOTest]")
{
  AddRequiredCLIOptions();

  // Create the ARFF that we will read.
  fstream f;
  f.open("test.arff", fstream::out);
  f << "@relation test" << endl;
  f << endl;
  f << "@attribute one STRING" << endl;
  f << "@attribute two REAL" << endl;
  f << endl;
  f << "@attribute three STRING" << endl;
  f << endl;
  f << "%% a comment line " << endl;
  f << endl;
  f << "@data" << endl;
  f << "hello, 1, moo" << endl;
  f << "cheese, 2.34, goodbye" << endl;
  f << "seven, 1.03e+5, moo" << endl;
  f << "hello, -1.3, goodbye" << endl;
  f.close();

  PARAM_MATRIX_AND_INFO_IN("tuple", "Test tuple", "t");

  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "--tuple_file";
  argv[2] = "test.arff";
  int argc = 3;

  ParseCommandLine(argc, const_cast<char**>(argv));

  // Create a pre-filled DatasetInfo object.
  DatasetInfo info(3);
  info.Type(0) = Datatype::categorical;
  info.Type(2) = Datatype::categorical;
  info.MapString<size_t>("seven", 0); // This will have mapped value 0.
  info.MapString<size_t>("cheese", 0); // This will have mapped value 1.
  info.MapString<size_t>("hello", 0); // This will have mapped value 2.
  info.MapString<size_t>("goodbye", 2); // This will have mapped value 0.
  info.MapString<size_t>("moo", 2); // This will have mapped value 1.

  // Now set the dataset info.
  std::get<0>(IO::GetRawParam<tuple<DatasetInfo, arma::mat>>("tuple")) = info;

  // Now load the dataset.
  arma::mat dataset =
      std::get<1>(IO::GetParam<tuple<DatasetInfo, arma::mat>>("tuple"));

  // Check the values.
  REQUIRE(dataset(0, 0) == Approx(2.0).epsilon(1e-7));
  REQUIRE(dataset(1, 0) == Approx(1.0).epsilon(1e-7));
  REQUIRE(dataset(2, 0) == Approx(1.0).epsilon(1e-7));
  REQUIRE(dataset(0, 1) == Approx(1.0).epsilon(1e-7));
  REQUIRE(dataset(1, 1) == Approx(2.34).epsilon(1e-7));
  REQUIRE(dataset(2, 1) == Approx(0.0).margin(1e-5));
  REQUIRE(dataset(0, 2) == Approx(0.0).margin(1e-5));
  REQUIRE(dataset(1, 2) == Approx(1.03e+5).epsilon(1e-7));
  REQUIRE(dataset(2, 2) == Approx(1.0).epsilon(1e-7));
  REQUIRE(dataset(0, 3) == Approx(2.0).epsilon(1e-7));
  REQUIRE(dataset(1, 3) == Approx(-1.3).epsilon(1e-7));
  REQUIRE(dataset(2, 3) == Approx(0.0).margin(1e-5));

  remove("test.arff");
}

/**
 * Make sure typenames are properly stored.
 */
TEST_CASE_METHOD(IOTestDestroyer, "CppNameTest",
                "[IOTest]")
{
  AddRequiredCLIOptions();

  // Add a few parameters.
  PARAM_MATRIX_IN("matrix", "Test matrix", "m");
  PARAM_DOUBLE_IN("double", "Test double", "d", 0.0);

  // Check that the C++ typenames are right.
  REQUIRE(IO::Parameters().at("matrix").cppType == "arma::mat");
  REQUIRE(IO::Parameters().at("help").cppType == "bool");
  REQUIRE(IO::Parameters().at("double").cppType == "double");
}
