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
#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::data;
using namespace std;

BOOST_AUTO_TEST_SUITE(StringUtilityTest);

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
  en.DictEncode(arr, output);
  const std::unordered_map<std::string, size_t>maps = en.Mappings();
  // Checking that everying is mapped to different numbers
  std::unordered_map<size_t, size_t>cnt;
  for (auto it = maps.begin(); it != maps.end(); it++)
  {
    cnt[it->second]++;
    // Every token should be mapped only once
    BOOST_REQUIRE_EQUAL(cnt[it->second], 1);
  }
}

/**
* Test Dictionary custom tokenization
*/
BOOST_AUTO_TEST_CASE(DictionaryEncodingCustomTest)
{
  std::vector<string>arr(3);
  arr[0] = "hellobadhowbadarebadyou";
  arr[1] = "ibadambadgood";
  arr[2] = "Goodbadhowbadarebadyou";
  arma::sp_mat output;
  data::DicitonaryEncoding en;
  // Passing a empty string to encode characters
  en.DictEncode(arr, "bad" ,
        [](boost::string_view& str, boost::string_view& deliminator)
        {
          if (str.empty())
          {
            return str;
          }
          size_t pos = str.find(deliminator);
          if (pos == string::npos)
          {
            boost::string_view retval = str;
            str.clear();
            return retval;
          }
          boost::string_view retval = str.substr(0, pos);
          str.remove_prefix(pos + deliminator.length());
          return retval;
        }, output);
  const std::unordered_map<std::string, size_t>maps = en.Mappings();
  // Checking that everying is mapped to different numbers
  std::unordered_map<size_t, size_t>cnt;
  for (auto it = maps.begin(); it != maps.end(); it++)
  {
    cnt[it->second]++;
    // Every token should be mappend only once.
    BOOST_REQUIRE_EQUAL(cnt[it->second], 1);
  }
}

/**
* Test Dictionary encoding for characters
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
  en.DictEncode(arr, output, "");
  const std::unordered_map<std::string, size_t>maps = en.Mappings();
  // Checking that everying is mapped to different numbers
  std::unordered_map<size_t, size_t>cnt;
  for (auto it = maps.begin(); it != maps.end(); it++)
  {
    cnt[it->second]++;
    // Every token should be mappend only once.
    BOOST_REQUIRE_EQUAL(cnt[it->second], 1);
  }
}

/**
* Test Dictionary encoding fot output with no padding.
*/
BOOST_AUTO_TEST_CASE(DictionaryEncodingNoPaddingTest)
{
  std::vector<string>arr(3);
  arr[0] = "GACCA";
  arr[1] = "ABCABCD";
  arr[2] = "GAB";
  data::DicitonaryEncoding en;
  // Passing a empty string to encode characters
  std::vector<std::vector<int> > output;
  en.DictEncode(arr, output, "");
  const std::unordered_map<std::string, size_t>maps = en.Mappings();
  // Checking that everying is mapped to different numbers
  std::unordered_map<size_t, size_t>cnt;
  for (auto it = maps.begin(); it != maps.end(); it++)
  {
    cnt[it->second]++;
    // Every token should be mappend only once.
    BOOST_REQUIRE_EQUAL(cnt[it->second], 1);
  }
}

BOOST_AUTO_TEST_SUITE_END();
