/**
 * @file preprocess_string_test.cpp
 * @author Jeffin Sam
 *
 * Test mlpackMain() of preprocess_string_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
static const std::string testName = "PreprocessString";

#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/preprocess/preprocess_string_main.cpp>

#include "test_helper.hpp"
#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct PreprocessStringTestFixture
{
 public:
  PreprocessStringTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~PreprocessStringTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(PreprocessStringMainTest,
                         PreprocessStringTestFixture);

bool CompareFiles(const std::string& fileName1, const std::string& fileName2)
{
  std::ifstream f1(fileName1), f2(fileName2);
  std::string file1line, file2line;
  if(!f1.is_open())
    Log::Fatal << fileName1 << " can't be opened" << std::endl;
  if(!f2.is_open())
    Log::Fatal << fileName2 << " can't be opened" << std::endl;
  while(getline(f1, file1line) && getline(f2, file2line))
  {
    if (file1line != file2line)
      return false;
  }
  return true;
}

/**
 * Convert lower Case from string input dataset.
 */
BOOST_AUTO_TEST_CASE(LowerCaseTest)
{
  SetInputParam("actual_dataset", (std::string) "string_test.txt");
  SetInputParam("preprocess_dataset",
      (std::string) "output_preprocess_string_lower_test.txt");
  std::vector<std::string> dimension = {"1", "3-4"};
  SetInputParam("dimension", dimension);
  SetInputParam("lowercase", true );

  mlpackMain();

  BOOST_REQUIRE_EQUAL(CompareFiles("output_preprocess_string_lower_test.txt",
      "preprocess_string_lower_test.txt"), true);
  remove("output_preprocess_string_lower_test.txt");
}

/**
 * Remove Punctuation from string input dataset.
 */
BOOST_AUTO_TEST_CASE(PunctuationCaseTest)
{
  SetInputParam("actual_dataset", (std::string) "string_test.txt");
  SetInputParam("preprocess_dataset",
      (std::string) "output_preprocess_punctuation_test.txt");
  std::vector<std::string> dimension = {"1", "3-4"};
  SetInputParam("dimension", dimension);
  SetInputParam("punctuation", true );

  mlpackMain();
  BOOST_REQUIRE_EQUAL(CompareFiles("output_preprocess_punctuation_test.txt",
      "preprocess_string_punctuation_test.txt"), true);
  remove("output_preprocess_punctuation_test.txt");
}

BOOST_AUTO_TEST_SUITE_END();
