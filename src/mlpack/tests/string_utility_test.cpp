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
#include <mlpack/core/data/bow_encoding.hpp>
#include "mlpack/core/boost_backport/boost_backport_string_view.hpp"
#include <mlpack/core/data/tokenizer/char_split.hpp>
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
 * Test Bow encoding.
 */
BOOST_AUTO_TEST_CASE(BowEncodingTest)
{
  std::vector<string>arr(3);
  arr[0] = "hello how are you";
  arr[1] = "i am good";
  arr[2] = "Good how are you";
  arma::sp_mat output;
  data::Bow en;
  en.Encode(arr, output, tokenizer);
  const std::unordered_map<boost::string_view, size_t,
      boost::hash<boost::string_view>> maps = en.Mappings();
  // Checking that everything is mapped to different numbers
  std::unordered_map<size_t, size_t>cnt;
  for (auto it = maps.begin(); it != maps.end(); it++)
  {
    cnt[it->second]++;
    // Every token should be mapped only once
    BOOST_REQUIRE_EQUAL(cnt[it->second], 1);
  }
}

BOOST_AUTO_TEST_SUITE_END();
