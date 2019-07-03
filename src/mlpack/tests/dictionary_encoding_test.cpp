/**
 * @file dictionary_encoding_test.cpp
 * @author Jeffin Sam
 *
 * Tests for Dictionary Encoding Class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/data/dictionary_encoding.hpp>
#include "mlpack/core/boost_backport/boost_backport_string_view.hpp"
#include <mlpack/core/data/tokenizer/char_split.hpp>
#include <mlpack/core/data/tokenizer/split_by_char.hpp>
#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::data;
using namespace std;

BOOST_AUTO_TEST_SUITE(DictionaryEncodingTest);

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
  arma::mat output;
  data::DictionaryEncoding en;
  en.Encode(arr, output, tokenizer);
  const std::unordered_map<boost::string_view, size_t,
      boost::hash<boost::string_view>>& maps = en.Mappings();
  // Checking that everything is mapped to different numbers
  std::unordered_map<size_t, size_t>cnt;
  for (auto it = maps.begin(); it != maps.end(); it++)
  {
    cnt[it->second]++;
    // Every token should be mapped only once
    BOOST_REQUIRE_EQUAL(cnt[it->second], 1);
  }
  arma::mat target = "1 2 3 4;"
                     "5 6 7 0;"
                     "8 2 3 4;";
  CheckMatrices(output, target);
}

/**
* Test for SplitByChar class.
*/
BOOST_AUTO_TEST_CASE(SplitByCharTest)
{
  std::vector<boost::string_view> tokens;
  boost::string_view strview = "hello how are you";
  SplitByChar obj(" ");
  boost::string_view token;
  token = obj(strview);
  while (!token.empty())
  {
    tokens.push_back(token);
    token = obj(strview);
  }
  BOOST_REQUIRE_EQUAL(tokens[0], "hello");
  BOOST_REQUIRE_EQUAL(tokens[1], "how");
  BOOST_REQUIRE_EQUAL(tokens[2], "are");
  BOOST_REQUIRE_EQUAL(tokens[3], "you");
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
  arma::mat output;
  data::DictionaryEncoding en;
  // Passing a empty string to encode characters
  en.Encode(arr, output, [](boost::string_view& str) {
      if (str.empty())
        return str;
      boost::string_view retval = str.substr(0, 1);
      str.remove_prefix(1);
      return retval;
  });
  const std::unordered_map<boost::string_view, size_t,
      boost::hash<boost::string_view>>& maps = en.Mappings();
  // Checking that everything is mapped to different numbers.
  std::unordered_map<size_t, size_t>cnt;
  for (auto it = maps.begin(); it != maps.end(); it++)
  {
    cnt[it->second]++;
    // Every token should be mappend only once.
    BOOST_REQUIRE_EQUAL(cnt[it->second], 1);
  }
  arma::mat target = "1 2 3 3 2 0 0;"
                     "2 4 3 2 4 3 5;"
                     "1 2 4 0 0 0 0;";
  CheckMatrices(output, target);
}

/**
* Testing the functionality of copy constructor.
*/
BOOST_AUTO_TEST_CASE(CopyConstructorCharTest)
{
  std::vector<string>arr(3);
  arr[0] = "hello how are you";
  arr[1] = "i am good";
  arr[2] = "Good how are you";
  arma::sp_mat output;
  data::DictionaryEncoding en;
  en.Encode(arr, output, SplitByChar(" "));
  const std::unordered_map<boost::string_view, size_t,
      boost::hash<boost::string_view>>& maps = en.Mappings();
  data::DictionaryEncoding en2 = en;
  const std::unordered_map<boost::string_view, size_t,
      boost::hash<boost::string_view>>& maps2 = en2.Mappings();
  // Comparing both of them.
  BOOST_REQUIRE_EQUAL(maps2 == maps, true);
}

/**
* Test Dictionary encoding for output with no padding.
*/
BOOST_AUTO_TEST_CASE(DictionaryEncodingNoPaddingTest)
{
  std::vector<string>arr(2);
  arr[0] = "GACCA";
  arr[1] = "GAB";
  data::DictionaryEncoding en;
  std::vector<std::vector<size_t> > output;
  en.Encode(arr, output, [](boost::string_view& str) {
      if (str.empty())
        return str;
      boost::string_view retval = str.substr(0, 1);
      str.remove_prefix(1);
      return retval;
  });
  std::vector<std::vector<size_t>> req_output = {{1, 2, 3, 3, 2}, {1, 2, 4}};
  for (size_t i = 0; i < arr.size(); i++)
  {
    for (size_t j = 0; j < req_output[i].size(); j++)
    {
      BOOST_REQUIRE_EQUAL(req_output[i][j], output[i][j]);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END();
