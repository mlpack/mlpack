/**
 * @file string_utility_test.cpp
 * @author Ryan Curtin
 *
 * Tests for String Utility Class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/data/dictionary_encoding.hpp>
#include "mlpack/core/boost_backport/boost_backport_string_view.hpp"
#include <mlpack/core/data/tokenizer/strtok.hpp>
#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::data;
using namespace std;

BOOST_AUTO_TEST_SUITE(StringUtilityTest);

boost::string_view tokenizer(boost::string_view& str)
{
  boost::string_view retval;
  while (retval.empty()) {
    std::size_t pos = str.find_first_of(" ");
    if (pos == str.npos) {
      retval = str;
      str.clear();
      return retval;
    }
    retval = str.substr(0, pos);
    str.remove_prefix(pos + 1);
  }
  return retval;
}

/**
 * Test dictionary encoding.
 */
BOOST_AUTO_TEST_CASE(DictionaryEncodingTest)
{
  std::vector<string>arr(3);
  arr[0] = "hello how are you";
  arr[1] = "i am good";
  arr[2] = "Good how are you";
  arma::sp_mat output;
  data::DicitonaryEncoding en;
  en.Encode(arr, output, tokenizer);
  const std::unordered_map<boost::string_view, size_t, data::Hasher>maps = en.Mappings();
  // Checking that everything is mapped to different numbers
  std::unordered_map<size_t, size_t>cnt;
  for (auto it = maps.begin(); it != maps.end(); it++)
  {
    cnt[it->second]++;
    // Every token should be mapped only once
    BOOST_REQUIRE_EQUAL(cnt[it->second], 1);
  }
}

/**
* Test for Strtok class.
*/
BOOST_AUTO_TEST_CASE(StrtokTest)
{
  std::vector<string>arr(3);
  arr[0] = "hello how are you";
  arr[1] = "i am good";
  arr[2] = "Good how are you";
  arma::sp_mat output;
  data::DicitonaryEncoding en;
  en.Encode(arr, output, data::Strtok(" "));
  const std::unordered_map<boost::string_view, size_t, data::Hasher>maps = en.Mappings();
  // Checking that everything is mapped to different numbers.
  std::unordered_map<size_t, size_t>cnt;
  for (auto it = maps.begin(); it != maps.end(); it++)
  {
    cnt[it->second]++;
    // Every token should be mappend only once.
    BOOST_REQUIRE_EQUAL(cnt[it->second], 1);
  }
}

/**
* Test Dictionary encoding for characters using lamda function.
*/
BOOST_AUTO_TEST_CASE(DictionaryEncodingCharTest)
{
  std::vector<string>arr(3);
  arr[0] = "GACCA";
  arr[1] = "ABCABCD";
  arr[2] = "GAB";
  arma::sp_mat output;
  data::DicitonaryEncoding en;
  // Passing a empty string to encode characters
  en.Encode(arr, output, [](boost::string_view& str) {
      if (str.empty())
        return str;
      boost::string_view retval = str.substr(0, 1);
      str.remove_prefix(1);
      return retval;
  });
  const std::unordered_map<boost::string_view, size_t, data::Hasher>maps = en.Mappings();
  // Checking that everything is mapped to different numbers.
  std::unordered_map<size_t, size_t>cnt;
  for (auto it = maps.begin(); it != maps.end(); it++)
  {
    cnt[it->second]++;
    // Every token should be mappend only once.
    BOOST_REQUIRE_EQUAL(cnt[it->second], 1);
  }
}

/**
* Test Dictionary encoding for output with no padding.
*/
BOOST_AUTO_TEST_CASE(DictionaryEncodingNoPaddingTest)
{
  std::vector<string>arr(3);
  arr[0] = "GACCA";
  arr[1] = "ABCABCD";
  arr[2] = "GAB";
  data::DicitonaryEncoding en;
  std::vector<std::vector<size_t> > output;
  en.Encode(arr, output, [](boost::string_view& str) {
      if (str.empty())
        return str;
      boost::string_view retval = str.substr(0, 1);
      str.remove_prefix(1);
      return retval;
  });
  const std::unordered_map<boost::string_view, size_t, data::Hasher>maps = en.Mappings();
  // Checking that everything is mapped to different numbers.
  std::unordered_map<size_t, size_t>cnt;
  for (auto it = maps.begin(); it != maps.end(); it++)
  {
    cnt[it->second]++;
    // Every token should be mappend only once.
    BOOST_REQUIRE_EQUAL(cnt[it->second], 1);
  }
}

BOOST_AUTO_TEST_SUITE_END();
