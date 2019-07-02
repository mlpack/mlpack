/**
 * @file string_cleaning_test.cpp
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
#include <mlpack/core/data/tokenizer/split_by_char.hpp>
#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::data;
using namespace std;

BOOST_AUTO_TEST_SUITE(StringCleaningTest);

/**
 * Test for removal of stop words
 */
BOOST_AUTO_TEST_CASE(StopWordsTest)
{
  std::vector<string> arr(2);
  arr[0] = "isn't mlpack great?";
  arr[1] = "2019 gsoc is great.";
  mlpack::data::StringCleaning obj;
  std::unordered_set<boost::string_view,
      boost::hash<boost::string_view>>stopword;
  stopword.insert("isn't");
  stopword.insert("is");
  obj.RemoveStopWords(arr, stopword, data::SplitByChar(" "));
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
  mlpack::data::StringCleaning obj;
  obj.RemovePunctuation(arr);
  BOOST_CHECK_EQUAL(arr[0], "isnt mlpack great");
  BOOST_CHECK_EQUAL(arr[1], "2019 gsoc is great");
}

/**
 * Test for removal of char
 */
BOOST_AUTO_TEST_CASE(CharRemovalTest)
{
  std::vector<string> arr(2);
  arr[0] = "isn't mlpack great?";
  arr[1] = "2019 gsoc is great.";
  mlpack::data::StringCleaning obj;
  obj.RemoveChar(arr, []( char c){ return ispunct(c); });
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
  mlpack::data::StringCleaning obj;
  obj.LowerCase(arr);
  BOOST_CHECK_EQUAL(arr[0], "isn't mlpack great?");
  BOOST_CHECK_EQUAL(arr[1], "2019 gsoc is great.");
}

/**
 * Test for converting to Upper case.
 */
BOOST_AUTO_TEST_CASE(UpperCaseTest)
{
  std::vector<string> arr(2);
  arr[0] = "IsN'T MlPaCk GrEaT?";
  arr[1] = "2019 gSoC iS grEat.";
  mlpack::data::StringCleaning obj;
  obj.UpperCase(arr);
  BOOST_CHECK_EQUAL(arr[0], "ISN'T MLPACK GREAT?");
  BOOST_CHECK_EQUAL(arr[1], "2019 GSOC IS GREAT.");
}

BOOST_AUTO_TEST_SUITE_END();
