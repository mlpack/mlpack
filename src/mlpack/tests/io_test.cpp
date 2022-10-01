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

#include <mlpack/core/util/param.hpp>
#include <mlpack/bindings/cli/parse_command_line.hpp>
#include <mlpack/bindings/cli/end_program.hpp>

#include "catch.hpp"

using namespace mlpack;
using namespace mlpack::util;
using namespace mlpack::data;
using namespace mlpack::bindings::cli;
using namespace std;

/**
 * Before running a test that uses the CLI options, we have to add the default
 * options that are required for CLI to function, since it will be destroyed at
 * the end of every test that uses CLI in this test suite.
 */
void AddRequiredCLIOptions(const std::string& testName)
{
  // These will register with CLI immediately.
  CLIOption<bool> help(false, "help", "Default help info.", "h", "bool", false,
      true, false, testName);
  CLIOption<string> info("", "info", "Get help on a specific module or option.",
      "", "string", false, true, false, testName);
  CLIOption<bool> verbose(false, "verbose", "Display information messages and "
      "the full list of parameters and timers at the end of execution.", "v",
      "bool", false, true, false, testName);
  CLIOption<bool> version(false, "version", "Display the version of mlpack.",
      "V", "bool", false, true, false, testName);
}

/**
 * Tests that CLI works as intended, namely that IO::Add propagates
 * successfully.
 */
TEST_CASE("TestCLIAdd", "[IOTest]")
{
  AddRequiredCLIOptions("TestCLIAdd");

  // Check that the IO::HasParam returns false if no value has been specified
  // on the commandline and ignores any programmatical assignments.
  CLIOption<bool> b(false, "global_bool", "True or false.", "a", "bool",
      false, true, false, "TestCLIAdd");

  util::Params p = IO::Parameters("TestCLIAdd");

  // IO::HasParam should return false here.
  REQUIRE(!p.Has("global_bool"));

  // Check that our aliasing works.
  REQUIRE(p.Has("global_bool") == p.Has("a"));
  REQUIRE(p.Get<bool>("global_bool") == p.Get<bool>("a"));
}

/**
 * Tests that the various PARAM_* macros work properly.
 */
TEST_CASE("TestOption", "[IOTest]")
{
  AddRequiredCLIOptions("TestOption");

  // This test will involve creating an option, and making sure CLI reflects
  // this.
  #define BINDING_NAME TestOption
  PARAM_IN(int, "test", "test desc", "", 42, false);
  #undef BINDING_NAME

  util::Params p = IO::Parameters("TestOption");
  REQUIRE(p.Get<int>("test") == 42);
}

/**
 * Test that duplicate flags are filtered out correctly.
 */
TEST_CASE("TestDuplicateFlag", "[IOTest]")
{
  AddRequiredCLIOptions("TestDuplicateFlag");

  #define BINDING_NAME TestDuplicateFlag
  PARAM_FLAG("test", "test", "t");
  #undef BINDING_NAME

  int argc = 3;
  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "--test";
  argv[2] = "--test";

  // This should not throw an exception.
  REQUIRE_NOTHROW(
      ParseCommandLine(argc, const_cast<char**>(argv), "TestDuplicateFlag"));
}

/**
 * Test that duplicate options throw an exception.
 */
TEST_CASE("TestDuplicateParam", "[IOTest]")
{
  AddRequiredCLIOptions("TestDuplicateParam");

  int argc = 5;
  const char* argv[5];
  argv[0] = "./test";
  argv[1] = "--info";
  argv[2] = "test1";
  argv[3] = "--info";
  argv[4] = "test2";

  // This should throw an exception.
  REQUIRE_THROWS_AS(ParseCommandLine(argc, const_cast<char**>(argv),
      "TestDuplicateParam"), std::runtime_error);
}

/**
 * Ensure that a Boolean option which we define is set correctly.
 */
TEST_CASE("TestBooleanOption", "[IOTest]")
{
  AddRequiredCLIOptions("TestBooleanOption");

  #define BINDING_NAME TestBooleanOption
  PARAM_FLAG("flag_test", "flag test description", "");
  #undef BINDING_NAME

  util::Params p = IO::Parameters("TestBooleanOption");
  REQUIRE(p.Has("flag_test") == false);

  // Now check that CLI reflects that it is false by default.
  REQUIRE(p.Get<bool>("flag_test") == false);

  // Now, if we specify this flag, it should be true.
  int argc = 2;
  const char* argv[2];
  argv[0] = "programname";
  argv[1] = "--flag_test";

  p = ParseCommandLine(argc, const_cast<char**>(argv), "TestBooleanOption");

  REQUIRE(p.Get<bool>("flag_test") == true);
  REQUIRE(p.Has("flag_test") == true);
}

/**
 * Test that a vector option works correctly.
 */
TEST_CASE("TestVectorOption", "[IOTest]")
{
  AddRequiredCLIOptions("TestVectorOption");

  #define BINDING_NAME TestVectorOption
  PARAM_VECTOR_IN(size_t, "test_vec", "test description", "t");
  #undef BINDING_NAME

  int argc = 5;
  const char* argv[5];
  argv[0] = "./test";
  argv[1] = "--test_vec";
  argv[2] = "1";
  argv[3] = "2";
  argv[4] = "4";

  util::Params p = ParseCommandLine(argc, const_cast<char**>(argv),
      "TestVectorOption");

  REQUIRE(p.Has("test_vec"));

  vector<size_t> v = p.Get<vector<size_t>>("test_vec");

  REQUIRE(v.size() == 3);
  REQUIRE(v[0] == 1);
  REQUIRE(v[1] == 2);
  REQUIRE(v[2] == 4);
}

/**
 * Test that we can use a vector option by specifying it many times.
 */
TEST_CASE("TestVectorOption2", "[IOTest]")
{
  AddRequiredCLIOptions("TestVectorOption2");

  #define BINDING_NAME TestVectorOption2
  PARAM_VECTOR_IN(size_t, "test2_vec", "test description", "T");
  #undef BINDING_NAME

  int argc = 7;
  const char* argv[7];
  argv[0] = "./test";
  argv[1] = "--test2_vec";
  argv[2] = "1";
  argv[3] = "--test2_vec";
  argv[4] = "2";
  argv[5] = "--test2_vec";
  argv[6] = "4";

  util::Params p = ParseCommandLine(argc, const_cast<char**>(argv),
      "TestVectorOption2");

  REQUIRE(p.Has("test2_vec"));

  vector<size_t> v = p.Get<vector<size_t>>("test2_vec");

  REQUIRE(v.size() == 3);
  REQUIRE(v[0] == 1);
  REQUIRE(v[1] == 2);
  REQUIRE(v[2] == 4);
}

TEST_CASE("InputColVectorParamTest", "[IOTest]")
{
  AddRequiredCLIOptions("InputColVectorParamTest");

  #define BINDING_NAME InputColVectorParamTest
  PARAM_COL_IN("vector", "Test vector", "l");
  #undef BINDING_NAME

  // Fake arguments.
  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "-l";
  argv[2] = "iris_test_labels.csv";

  int argc = 3;

  // The const-cast is a little hacky but should be fine...
  util::Params p = ParseCommandLine(argc, const_cast<char**>(argv),
      "InputColVectorParamTest");

  // The --vector parameter should exist.
  REQUIRE(p.Has("vector"));
  // The --vector_file parameter should not exist (it should be transparent from
  // inside the program).
  REQUIRE_THROWS_AS(p.Has("vector_file"), runtime_error);

  arma::vec vec1 = p.Get<arma::vec>("vector");
  arma::vec vec2 = p.Get<arma::vec>("vector");

  REQUIRE(vec1.n_rows == 63);
  REQUIRE(vec2.n_rows == 63);

  for (size_t i = 0; i < vec1.n_elem; ++i)
    REQUIRE(vec1[i] == Approx(vec2[i]).epsilon(1e-12));
}

TEST_CASE("InputUnsignedColVectorParamTest", "[IOTest]")
{
  AddRequiredCLIOptions("InputUnsignedColVectorParamTest");

  #define BINDING_NAME InputUnsignedColVectorParamTest
  PARAM_UCOL_IN("vector", "Test vector", "l");
  #undef BINDING_NAME

  // Fake arguments.
  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "-l";
  argv[2] = "iris_test_labels.csv";

  int argc = 3;

  // The const-cast is a little hacky but should be fine...
  util::Params p = ParseCommandLine(argc, const_cast<char**>(argv),
      "InputUnsignedColVectorParamTest");

  // The --vector parameter should exist.
  REQUIRE(p.Has("vector"));
  // The --vector_file parameter should not exist (it should be transparent from
  // inside the program).
  REQUIRE_THROWS_AS(p.Has("vector_file"), runtime_error);

  arma::Col<size_t> vec1 = p.Get<arma::Col<size_t>>("vector");
  arma::Col<size_t> vec2 = p.Get<arma::Col<size_t>>("vector");

  REQUIRE(vec1.n_rows == 63);
  REQUIRE(vec2.n_rows == 63);

  for (size_t i = 0; i < vec1.n_elem; ++i)
    REQUIRE(vec1[i] == vec2[i]);
}

TEST_CASE("InputRowVectorParamTest", "[IOTest]")
{
  AddRequiredCLIOptions("InputRowVectorParamTest");

  #define BINDING_NAME InputRowVectorParamTest
  PARAM_ROW_IN("row", "Test vector", "l");
  #undef BINDING_NAME

  // Fake arguments.
  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "-l";
  argv[2] = "testRes.csv";

  int argc = 3;

  // The const-cast is a little hacky but should be fine...
  util::Params p = ParseCommandLine(argc, const_cast<char**>(argv),
      "InputRowVectorParamTest");

  // The --vector parameter should exist.
  REQUIRE(p.Has("row"));
  // The --vector_file parameter should not exist (it should be transparent from
  // inside the program).
  REQUIRE_THROWS_AS(p.Has("row_file"), runtime_error);

  arma::rowvec vec1 = p.Get<arma::rowvec>("row");
  arma::rowvec vec2 = p.Get<arma::rowvec>("row");

  REQUIRE(vec1.n_cols == 7);
  REQUIRE(vec2.n_cols == 7);

  for (size_t i = 0; i < vec1.n_elem; ++i)
    REQUIRE(vec1[i] == Approx(vec2[i]).epsilon(1e-12));
}

TEST_CASE("InputUnsignedRowVectorParamTest", "[IOTest]")
{
  AddRequiredCLIOptions("InputUnsignedRowVectorParamTest");

  #define BINDING_NAME InputUnsignedRowVectorParamTest
  PARAM_UROW_IN("row", "Test vector", "l");
  #undef BINDING_NAME

  // Fake arguments.
  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "-l";
  argv[2] = "testRes.csv";

  int argc = 3;

  // The const-cast is a little hacky but should be fine...
  util::Params p = ParseCommandLine(argc, const_cast<char**>(argv),
      "InputUnsignedRowVectorParamTest");

  // The --vector parameter should exist.
  REQUIRE(p.Has("row"));
  // The --vector_file parameter should not exist (it should be transparent from
  // inside the program).
  REQUIRE_THROWS_AS(p.Has("row_file"), runtime_error);

  arma::Row<size_t> vec1 = p.Get<arma::Row<size_t>>("row");
  arma::Row<size_t> vec2 = p.Get<arma::Row<size_t>>("row");

  REQUIRE(vec1.n_cols == 7);
  REQUIRE(vec2.n_cols == 7);

  for (size_t i = 0; i < vec1.n_elem; ++i)
    REQUIRE(vec1[i] == vec2[i]);
}

TEST_CASE("OutputColParamTest", "[IOTest]")
{
  AddRequiredCLIOptions("OutputColParamTest");

  // --vector is an output parameter.
  #define BINDING_NAME OutputColParamTest
  PARAM_COL_OUT("vector", "Test vector", "l");
  #undef BINDING_NAME

  // Set some fake arguments.
  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "-l";
  argv[2] = "test.csv";

  int argc = 3;

  util::Params p = ParseCommandLine(argc, const_cast<char**>(argv),
      "OutputColParamTest");

  // The --vector parameter should exist.
  REQUIRE(p.Has("vector"));
  // The --vector_file parameter should not exist (it should be transparent from
  // inside the program).
  REQUIRE_THROWS_AS(p.Has("vector_file"), runtime_error);

  // Since it's an output parameter, we don't need any input and don't need to
  // call ParseCommandLine().
  arma::vec dataset = arma::randu<arma::vec>(100);
  p.Get<arma::vec>("vector") = dataset;

  // Write the file.
  util::Timers t;
  EndProgram(p, t);

  // Now load the vector back and make sure it was saved correctly.
  arma::vec dataset2;
  if (!data::Load("test.csv", dataset2))
    FAIL("Cannot load dataset test.csv");

  REQUIRE(dataset.n_rows == dataset2.n_rows);
  for (size_t i = 0; i < dataset.n_elem; ++i)
    REQUIRE(dataset[i] == Approx(dataset2[i]).epsilon(1e-12));

  // Remove the file.
  remove("test.csv");
}

TEST_CASE("OutputUnsignedColParamTest", "[IOTest]")
{
  AddRequiredCLIOptions("OutputUnsignedColParamTest");

  // --vector is an output parameter.
  #define BINDING_NAME OutputUnsignedColParamTest
  PARAM_UCOL_OUT("vector", "Test vector", "l");
  #undef BINDING_NAME

  // Set some fake arguments.
  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "-l";
  argv[2] = "test.csv";

  int argc = 3;

  util::Params p = ParseCommandLine(argc, const_cast<char**>(argv),
      "OutputUnsignedColParamTest");

  // The --vector parameter should exist.
  REQUIRE(p.Has("vector"));
  // The --vector_file parameter should not exist (it should be transparent from
  // inside the program).
  REQUIRE_THROWS_AS(p.Has("vector_file"), runtime_error);

  // Since it's an output parameter, we don't need any input and don't need to
  // call ParseCommandLine().
  arma::Col<size_t> dataset = arma::randi<arma::Col<size_t>>(100);
  p.Get<arma::Col<size_t>>("vector") = dataset;

  // Write the file.
  util::Timers t;
  EndProgram(p, t);

  // Now load the vector back and make sure it was saved correctly.
  arma::Col<size_t> dataset2;
  if (!data::Load("test.csv", dataset2))
    FAIL("Cannot load dataset test.csv");

  REQUIRE(dataset.n_rows == dataset2.n_rows);
  for (size_t i = 0; i < dataset.n_elem; ++i)
    REQUIRE(dataset[i] == dataset2[i]);

  // Remove the file.
  remove("test.csv");
}

TEST_CASE("OutputRowParamTest", "[IOTest]")
{
  AddRequiredCLIOptions("OutputRowParamTest");

  // --row is an output parameter.
  #define BINDING_NAME OutputRowParamTest
  PARAM_ROW_OUT("row", "Test vector", "l");
  #undef BINDING_NAME

  // Set some fake arguments.
  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "-l";
  argv[2] = "test.csv";

  int argc = 3;

  util::Params p = ParseCommandLine(argc, const_cast<char**>(argv),
      "OutputRowParamTest");

  // The --row parameter should exist.
  REQUIRE(p.Has("row"));
  // The --row_file parameter should not exist (it should be transparent from
  // inside the program).
  REQUIRE_THROWS_AS(p.Has("row_file"), runtime_error);

  // Since it's an output parameter, we don't need any input and don't need to
  // call ParseCommandLine().
  arma::rowvec dataset = arma::randu<arma::rowvec>(100);
  p.Get<arma::rowvec>("row") = dataset;

  // Write the file.
  util::Timers t;
  EndProgram(p, t);

  // Now load the row vector back and make sure it was saved correctly.
  arma::rowvec dataset2;
  if (!data::Load("test.csv", dataset2))
    FAIL("Cannot load dataset test.csv");

  REQUIRE(dataset.n_cols == dataset2.n_cols);
  for (size_t i = 0; i < dataset.n_elem; ++i)
    REQUIRE(dataset[i] == Approx(dataset2[i]).epsilon(1e-12));

  // Remove the file.
  remove("test.csv");
}

TEST_CASE("OutputUnsignedRowParamTest", "[IOTest]")
{
  AddRequiredCLIOptions("OutputUnsignedRowParamTest");

  // --row is an output parameter.
  #define BINDING_NAME OutputUnsignedRowParamTest
  PARAM_UROW_OUT("row", "Test vector", "l");
  #undef BINDING_NAME

  // Set some fake arguments.
  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "-l";
  argv[2] = "test.csv";

  int argc = 3;

  util::Params p = ParseCommandLine(argc, const_cast<char**>(argv),
      "OutputUnsignedRowParamTest");

  // The --row parameter should exist.
  REQUIRE(p.Has("row"));
  // The --row_file parameter should not exist (it should be transparent from
  // inside the program).
  REQUIRE_THROWS_AS(p.Has("row_file"), runtime_error);

  // Since it's an output parameter, we don't need any input and don't need to
  // call ParseCommandLine().
  arma::Row<size_t> dataset = arma::randi<arma::Row<size_t>>(100);
  p.Get<arma::Row<size_t>>("row") = dataset;

  // Write the file.
  util::Timers t;
  EndProgram(p, t);

  // Now load the row vector back and make sure it was saved correctly.
  arma::Row<size_t> dataset2;
  if (!data::Load("test.csv", dataset2))
    FAIL("Cannot load dataset test.csv");

  REQUIRE(dataset.n_cols == dataset2.n_cols);
  for (size_t i = 0; i < dataset.n_elem; ++i)
    REQUIRE(dataset[i] == dataset2[i]);

  // Remove the file.
  remove("test.csv");
}

TEST_CASE("InputMatrixParamTest", "[IOTest]")
{
  AddRequiredCLIOptions("InputMatrixParamTest");

  // --matrix is an input parameter; it won't be transposed.
  #define BINDING_NAME InputMatrixParamTest
  PARAM_MATRIX_IN("matrix", "Test matrix", "m");
  #undef BINDING_NAME

  // Set some fake arguments.
  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "-m";
  argv[2] = "test_data_3_1000.csv";

  int argc = 3;

  // The const-cast is a little hacky but should be fine...
  util::Params p = ParseCommandLine(argc, const_cast<char**>(argv),
      "InputMatrixParamTest");

  // The --matrix parameter should exist.
  REQUIRE(p.Has("matrix"));
  // The --matrix_file parameter should not exist (it should be transparent from
  // inside the program).
  REQUIRE_THROWS_AS(p.Has("matrix_file"), runtime_error);

  arma::mat dataset = p.Get<arma::mat>("matrix");
  arma::mat dataset2 = p.Get<arma::mat>("matrix");

  REQUIRE(dataset.n_rows == 3);
  REQUIRE(dataset.n_cols == 1000);
  REQUIRE(dataset2.n_rows == 3);
  REQUIRE(dataset2.n_cols == 1000);

  for (size_t i = 0; i < dataset.n_elem; ++i)
    REQUIRE(dataset[i] == Approx(dataset2[i]).epsilon(1e-12));
}

// Make sure we can correctly load required matrix parameters.
TEST_CASE("RequiredInputMatrixParamTest", "[IOTest]")
{
  AddRequiredCLIOptions("RequiredInputMatrixParamTest");

  // --matrix is an input parameter; it won't be transposed.
  #define BINDING_NAME RequiredInputMatrixParamTest
  PARAM_MATRIX_IN_REQ("matrix", "Test matrix", "m");
  #undef BINDING_NAME

  // Set some fake arguments.
  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "--matrix_file";
  argv[2] = "test_data_3_1000.csv";

  int argc = 3;

  // The const-cast is a little hacky but should be fine...
  util::Params p = ParseCommandLine(argc, const_cast<char**>(argv),
      "RequiredInputMatrixParamTest");

  // The --matrix parameter should exist.
  REQUIRE(p.Has("matrix"));
  // The --matrix_file parameter should not exist (it should be transparent from
  // inside the program).
  REQUIRE_THROWS_AS(p.Has("matrix_file"), runtime_error);

  arma::mat dataset = p.Get<arma::mat>("matrix");
  arma::mat dataset2 = p.Get<arma::mat>("matrix");

  REQUIRE(dataset.n_rows == 3);
  REQUIRE(dataset.n_cols == 1000);
  REQUIRE(dataset2.n_rows == 3);
  REQUIRE(dataset2.n_cols == 1000);

  for (size_t i = 0; i < dataset.n_elem; ++i)
    REQUIRE(dataset[i] == Approx(dataset2[i]).epsilon(1e-12));
}

// Make sure loading required matrix options by alias succeeds.
TEST_CASE("RequiredInputMatrixParamAliasTest", "[IOTest]")
{
  AddRequiredCLIOptions("RequiredInputMatrixParamAliasTest");

  // --matrix is an input parameter; it won't be transposed.
  #define BINDING_NAME RequiredInputMatrixParamAliasTest
  PARAM_MATRIX_IN_REQ("matrix", "Test matrix", "m");
  #undef BINDING_NAME

  // Set some fake arguments.
  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "-m";
  argv[2] = "test_data_3_1000.csv";

  int argc = 3;

  // The const-cast is a little hacky but should be fine...
  util::Params p = ParseCommandLine(argc, const_cast<char**>(argv),
      "RequiredInputMatrixParamAliasTest");

  // The --matrix parameter should exist.
  REQUIRE(p.Has("matrix"));
  // The --matrix_file parameter should not exist (it should be transparent from
  // inside the program).
  REQUIRE_THROWS_AS(p.Has("matrix_file"), runtime_error);

  arma::mat dataset = p.Get<arma::mat>("matrix");
  arma::mat dataset2 = p.Get<arma::mat>("matrix");

  REQUIRE(dataset.n_rows == 3);
  REQUIRE(dataset.n_cols == 1000);
  REQUIRE(dataset2.n_rows == 3);
  REQUIRE(dataset2.n_cols == 1000);

  for (size_t i = 0; i < dataset.n_elem; ++i)
    REQUIRE(dataset[i] == Approx(dataset2[i]).epsilon(1e-12));
}

// Make sure that when we don't pass a required matrix, parsing fails.
TEST_CASE("RequiredUnspecifiedInputMatrixParamTest", "[IOTest]")
{
  AddRequiredCLIOptions("RequiredUnspecifiedInputMatrixParamTest");

  // --matrix is an input parameter; it won't be transposed.
  #define BINDING_NAME RequiredUnspecifiedInputMatrixParamTest
  PARAM_MATRIX_IN_REQ("matrix", "Test matrix", "m");
  #undef BINDING_NAME

  // Set some fake arguments.
  const char* argv[1];
  argv[0] = "./test";

  int argc = 1;

  // The const-cast is a little hacky but should be fine...
  REQUIRE_THROWS_AS(ParseCommandLine(argc, const_cast<char**>(argv),
      "RequiredUnspecifiedInputMatrixParamTest"), std::exception);
}

TEST_CASE("InputMatrixNoTransposeParamTest", "[IOTest]")
{
  AddRequiredCLIOptions("InputMatrixNoTransposeParamTest");

  // --matrix is a non-transposed input parameter.
  #define BINDING_NAME InputMatrixNoTransposeParamTest
  PARAM_TMATRIX_IN("matrix", "Test matrix", "m");
  #undef BINDING_NAME

  // Set some fake arguments.
  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "--matrix_file";
  argv[2] = "test_data_3_1000.csv";

  int argc = 3;

  // The const-cast is a little hacky but should be fine...
  util::Params p = ParseCommandLine(argc, const_cast<char**>(argv),
      "InputMatrixNoTransposeParamTest");

  // The --matrix parameter should exist.
  REQUIRE(p.Has("matrix"));
  // The --matrix_file parameter should not exist (it should be transparent from
  // inside the program).
  REQUIRE_THROWS_AS(p.Has("matrix_file"), runtime_error);

  arma::mat dataset = p.Get<arma::mat>("matrix");
  arma::mat dataset2 = p.Get<arma::mat>("matrix");

  REQUIRE(dataset.n_rows == 1000);
  REQUIRE(dataset.n_cols == 3);
  REQUIRE(dataset2.n_rows == 1000);
  REQUIRE(dataset2.n_cols == 3);

  for (size_t i = 0; i < dataset.n_elem; ++i)
    REQUIRE(dataset[i] == Approx(dataset2[i]).epsilon(1e-12));
}

TEST_CASE("OutputMatrixParamTest", "[IOTest]")
{
  AddRequiredCLIOptions("OutputMatrixParamTest");

  // --matrix is an output parameter.
  #define BINDING_NAME OutputMatrixParamTest
  PARAM_MATRIX_OUT("matrix", "Test matrix", "m");
  #undef BINDING_NAME

  // Set some fake arguments.
  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "-m";
  argv[2] = "test.csv";

  int argc = 3;

  util::Params p = ParseCommandLine(argc, const_cast<char**>(argv),
      "OutputMatrixParamTest");

  // The --matrix parameter should exist.
  REQUIRE(p.Has("matrix"));
  // The --matrix_file parameter should not exist (it should be transparent from
  // inside the program).
  REQUIRE_THROWS_AS(p.Has("matrix_file"), runtime_error);

  // Since it's an output parameter, we don't need any input and don't need to
  // call ParseCommandLine().
  arma::mat dataset = arma::randu<arma::mat>(3, 100);
  p.Get<arma::mat>("matrix") = dataset;

  // Write the file.
  util::Timers t;
  EndProgram(p, t);

  // Now load the matrix back and make sure it was saved correctly.
  arma::mat dataset2;
  if (!data::Load("test.csv", dataset2))
    FAIL("Cannot load dataset test.csv");

  REQUIRE(dataset.n_cols == dataset2.n_cols);
  REQUIRE(dataset.n_rows == dataset2.n_rows);
  for (size_t i = 0; i < dataset.n_elem; ++i)
    REQUIRE(dataset[i] == Approx(dataset2[i]).epsilon(1e-12));

  // Remove the file.
  remove("test.csv");
}

TEST_CASE("OutputMatrixNoTransposeParamTest", "[IOTest]")
{
  AddRequiredCLIOptions("OutputMatrixNoTransposeParamTest");

  // --matrix is an output parameter.
  #define BINDING_NAME OutputMatrixNoTransposeParamTest
  PARAM_TMATRIX_OUT("matrix", "Test matrix", "m");
  #undef BINDING_NAME

  // Set some fake arguments.
  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "-m";
  argv[2] = "test.csv";

  int argc = 3;

  util::Params p = ParseCommandLine(argc, const_cast<char**>(argv),
      "OutputMatrixNoTransposeParamTest");

  // The --matrix parameter should exist.
  REQUIRE(p.Has("matrix"));
  // The --matrix_file parameter should not exist (it should be transparent from
  // inside the program).
  REQUIRE_THROWS_AS(p.Has("matrix_file"), runtime_error);

  // Since it's an output parameter, we don't need any input and don't need to
  // call ParseCommandLine().
  arma::mat dataset = arma::randu<arma::mat>(3, 100);
  p.Get<arma::mat>("matrix") = dataset;

  // Write the file.
  util::Timers t;
  EndProgram(p, t);

  // Now load the matrix back and make sure it was saved correctly.
  arma::mat dataset2;
  if (!data::Load("test.csv", dataset2, false, false))
    FAIL("Cannot load dataset test.csv");

  REQUIRE(dataset.n_cols == dataset2.n_cols);
  REQUIRE(dataset.n_rows == dataset2.n_rows);
  for (size_t i = 0; i < dataset.n_elem; ++i)
    REQUIRE(dataset[i] == Approx(dataset2[i]).epsilon(1e-12));

  // Remove the file.
  remove("test.csv");
}

TEST_CASE("IntParamTest", "[IOTest]")
{
  AddRequiredCLIOptions("IntParamTest");

  #define BINDING_NAME IntParamTest
  PARAM_INT_IN("int", "Test int", "i", 0);
  #undef BINDING_NAME

  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "-i";
  argv[2] = "3";

  int argc = 3;

  util::Params p = ParseCommandLine(argc, const_cast<char**>(argv),
      "IntParamTest");

  REQUIRE(p.Has("int"));
  REQUIRE(p.Get<int>("int") == 3);
}

TEST_CASE("StringParamTest", "[IOTest]")
{
  AddRequiredCLIOptions("StringParamTest");

  #define BINDING_NAME StringParamTest
  PARAM_STRING_IN("string", "Test string", "s", "");
  #undef BINDING_NAME

  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "--string";
  argv[2] = "3";

  int argc = 3;

  util::Params p = ParseCommandLine(argc, const_cast<char**>(argv),
      "StringParamTest");

  REQUIRE(p.Has("string"));
  REQUIRE(p.Get<string>("string") == string("3"));
}

TEST_CASE("DoubleParamTest", "[IOTest]")
{
  AddRequiredCLIOptions("DoubleParamTest");

  #define BINDING_NAME DoubleParamTest
  PARAM_DOUBLE_IN("double", "Test double", "d", 0.0);
  #undef BINDING_NAME

  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "--double";
  argv[2] = "3.12";

  int argc = 3;

  util::Params p = ParseCommandLine(argc, const_cast<char**>(argv),
      "DoubleParamTest");

  REQUIRE(p.Has("double"));
  REQUIRE(p.Get<double>("double") == Approx(3.12).epsilon(1e-12));
}

TEST_CASE("RequiredOptionTest", "[IOTest]")
{
  AddRequiredCLIOptions("RequiredOptionTest");

  #define BINDING_NAME RequiredOptionTest
  PARAM_DOUBLE_IN_REQ("double", "Required test double", "d");
  #undef BINDING_NAME

  const char* argv[1];
  argv[0] = "./test";

  int argc = 1;

  REQUIRE_THROWS_AS(ParseCommandLine(argc, const_cast<char**>(argv),
      "RequiredOptionTest"), runtime_error);
}

TEST_CASE("UnknownOptionTest", "[IOTest]")
{
  AddRequiredCLIOptions("UnknownOptionTest");

  const char* argv[2];
  argv[0] = "./test";
  argv[1] = "--unknown";

  int argc = 2;

  REQUIRE_THROWS_AS(ParseCommandLine(argc, const_cast<char**>(argv),
      "UnknownOptionTest"), runtime_error);
}

/**
 * Test that GetPrintableParam() works.
 */
TEST_CASE("UnmappedParamTest", "[IOTest]")
{
  AddRequiredCLIOptions("UnmappedParamTest");

  #define BINDING_NAME UnmappedParamTest
  PARAM_MATRIX_IN("matrix", "Test matrix", "m");
  PARAM_MATRIX_OUT("matrix2", "Test matrix", "M");
  PARAM_MODEL_IN(GaussianKernel, "kernel", "Test kernel", "k");
  PARAM_MODEL_OUT(GaussianKernel, "kernel2", "Test kernel", "K");
  #undef BINDING_NAME

  const char* argv[9];
  argv[0] = "./test";
  argv[1] = "--matrix_file";
  argv[2] = "test_data_3_1000.csv";
  argv[3] = "-M";
  argv[4] = "file2.csv";
  argv[5] = "-k";
  argv[6] = "kernel.json";
  argv[7] = "-K";
  argv[8] = "kernel2.json";

  int argc = 9;

  util::Params p = ParseCommandLine(argc, const_cast<char**>(argv),
      "UnmappedParamTest");

  // Now check that we can get unmapped parameters.
  REQUIRE(p.GetPrintable<arma::mat>("matrix") ==
      "'test_data_3_1000.csv' (1000x3 matrix)");
  // This will have size 0x0 since it's an output parameter, and it hasn't been
  // set since ParseCommandLine() was called.
  REQUIRE(p.GetPrintable<arma::mat>("matrix2") ==
      "'file2.csv' (0x0 matrix)");
  REQUIRE(p.GetPrintable<GaussianKernel*>("kernel") == "kernel.json");
  REQUIRE(p.GetPrintable<GaussianKernel*>("kernel2") == "kernel2.json");

  remove("kernel.json");
}

/**
 * Test that we can serialize a model and then deserialize it through the CLI
 * interface.
 */
TEST_CASE("IOSerializationTest", "[IOTest]")
{
  AddRequiredCLIOptions("IOSerializationTest");

  #define BINDING_NAME IOSerializationTest
  PARAM_MODEL_OUT(GaussianKernel, "kernel", "Test kernel", "k");
  #undef BINDING_NAME

  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "--kernel_file";
  argv[2] = "kernel.json";

  int argc = 3;

  util::Params p = ParseCommandLine(argc, const_cast<char**>(argv),
      "IOSerializationTest");

  // Create the kernel we'll save.
  GaussianKernel* gk = new GaussianKernel(0.5);

  p.Get<GaussianKernel*>("kernel") = gk;

  // Save it.
  util::Timers t;
  EndProgram(p, t);

  // Now create a new CLI object and load it.
  AddRequiredCLIOptions("IOSerializationTest_2");

  #define BINDING_NAME IOSerializationTest_2
  PARAM_MODEL_IN(GaussianKernel, "kernel", "Test kernel", "k");
  #undef BINDING_NAME

  p = ParseCommandLine(argc, const_cast<char**>(argv), "IOSerializationTest_2");

  // Load the kernel from file.
  GaussianKernel* gk2 = p.Get<GaussianKernel*>("kernel");

  REQUIRE(gk2->Bandwidth() == Approx(0.5).epsilon(1e-7));

  // Clean up the memory...
  delete gk2;

  // Now remove the file we made.
  remove("kernel.json");
}

/**
 * Test that an exception is thrown when a required model is not specified.
 */
TEST_CASE("RequiredModelTest", "[IOTest]")
{
  AddRequiredCLIOptions("RequiredModelTest");

  #define BINDING_NAME RequiredModelTest
  PARAM_MODEL_IN_REQ(GaussianKernel, "kernel", "Test kernel", "k");
  #undef BINDING_NAME

  // Don't specify any input parameters.
  const char* argv[1];
  argv[0] = "./test";

  int argc = 1;

  REQUIRE_THROWS_AS(ParseCommandLine(argc, const_cast<char**>(argv),
      "RequiredModelTest"), runtime_error);
}

/**
 * Test that we can load both a dataset and its associated info.
 */
TEST_CASE("MatrixAndDatasetInfoTest", "[IOTest]")
{
  AddRequiredCLIOptions("MatrixAndDatasetInfoTest");

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
  #define BINDING_NAME MatrixAndDatasetInfoTest
  PARAM_MATRIX_AND_INFO_IN("dataset", "Test dataset", "d");
  #undef BINDING_NAME

  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "--dataset_file";
  argv[2] = "test.arff";

  int argc = 3;

  util::Params p = ParseCommandLine(argc, const_cast<char**>(argv),
      "MatrixAndDatasetInfoTest");

  // Get the dataset and info.
  DatasetInfo info = move(get<0>(p.Get<TupleType>("dataset")));
  arma::mat dataset = move(get<1>(p.Get<TupleType>("dataset")));

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
TEST_CASE("RawIntegralParameter", "[IOTest]")
{
  AddRequiredCLIOptions("RawIntegralParameter");

  #define BINDING_NAME RawIntegralParameter
  PARAM_DOUBLE_IN("double", "Test double", "d", 0.0);
  #undef BINDING_NAME

  const char* argv[1];
  argv[0] = "./test";
  int argc = 1;

  util::Params p = ParseCommandLine(argc, const_cast<char**>(argv),
      "RawIntegralParameter");

  // Set the double.
  p.GetRaw<double>("double") = 3.0;

  // Now when we get it, it should be what we just set it to.
  REQUIRE(p.Get<double>("double") == Approx(3.0).epsilon(1e-7));
}

/**
 * Test that we can load a dataset with a pre-set mapping through
 * IO::GetRawParam().
 */
TEST_CASE("RawDatasetInfoLoadParameter", "[IOTest]")
{
  AddRequiredCLIOptions("RawDatasetInfoLoadParameter");

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

  #define BINDING_NAME RawDatasetInfoLoadParameter
  PARAM_MATRIX_AND_INFO_IN("tuple", "Test tuple", "t");
  #undef BINDING_NAME

  const char* argv[3];
  argv[0] = "./test";
  argv[1] = "--tuple_file";
  argv[2] = "test.arff";
  int argc = 3;

  util::Params p = ParseCommandLine(argc, const_cast<char**>(argv),
      "RawDatasetInfoLoadParameter");

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
  std::get<0>(p.GetRaw<tuple<DatasetInfo, arma::mat>>("tuple")) = info;

  // Now load the dataset.
  arma::mat dataset =
      std::get<1>(p.Get<tuple<DatasetInfo, arma::mat>>("tuple"));

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
TEST_CASE("CppNameTest", "[IOTest]")
{
  AddRequiredCLIOptions("CppNameTest");

  // Add a few parameters.
  #define BINDING_NAME CppNameTest
  PARAM_MATRIX_IN("matrix", "Test matrix", "m");
  PARAM_DOUBLE_IN("double", "Test double", "d", 0.0);
  #undef BINDING_NAME

  util::Params p = IO::Parameters("CppNameTest");

  // Check that the C++ typenames are right.
  REQUIRE(p.Parameters().at("matrix").cppType == "arma::mat");
  REQUIRE(p.Parameters().at("help").cppType == "bool");
  REQUIRE(p.Parameters().at("double").cppType == "double");
}
