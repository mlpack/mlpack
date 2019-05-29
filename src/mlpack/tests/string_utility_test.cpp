/**
 * @file string_utility_test.cpp
 * @author Jeffin Sam
 *
 * Tests for String Utility Class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/data/string_cleaning.hpp>
#include "mlpack/core/boost_backport/boost_backport_string_view.hpp"
#include <mlpack/core/data/tokenizer/strtok.hpp>
#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::data;
using namespace std;

BOOST_AUTO_TEST_SUITE(StringUtilityTest);

/**
 * Test for removal of stop words
 */
BOOST_AUTO_TEST_CASE(StopWordsTest)
{
  std::vector<string> arr(2);
  arr[0] = "isn't mlpack great?";
  arr[1] = "2019 gsoc is great.";
  mlpack::data::RemoveStopWords(arr, data::Strtok(" "));
  BOOST_CHECK_EQUAL(arr[0], "mlpack great?");
  BOOST_CHECK_EQUAL(arr[1], "2019 gsoc great.");
}

/**
 * Test for removal of punctuation
 */
BOOST_AUTO_TEST_CASE(PunctuationTest)
{
  std::vector<string> arr(2);
  arr[0] = "isn't mlpack great?";
  arr[1] = "2019 gsoc is great.";
  mlpack::data::RemovePunctuation(arr);
  BOOST_CHECK_EQUAL(arr[0], "isnt mlpack great");
  BOOST_CHECK_EQUAL(arr[1], "2019 gsoc is great");
}

/**
 * Test for converting to lower case.
 */
BOOST_AUTO_TEST_CASE(LowerCaseTest)
{
  std::vector<string> arr(2);
  arr[0] = "IsN'T MlPaCk GrEaT?";
  arr[1] = "2019 gSoC iS grEat.";
  mlpack::data::LowerCase(arr);
  BOOST_CHECK_EQUAL(arr[0], "isn't mlpack great?");
  BOOST_CHECK_EQUAL(arr[1], "2019 gsoc is great.");
}

BOOST_AUTO_TEST_SUITE_END();
