/**
 * @file tfidf_encoding_test.cpp
 * @author Jeffin Sam
 *
 * Tests for TFIDF Encoding Class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/data/tfidf_encoding.hpp>
#include "mlpack/core/boost_backport/boost_backport_string_view.hpp"
#include <mlpack/core/data/tokenizer/char_split.hpp>
#include <mlpack/core/data/tokenizer/split_by_char.hpp>
#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::data;
using namespace std;

BOOST_AUTO_TEST_SUITE(TfIdfEncodingTest);

/**
 * Test TfIdf encoding.
 */
BOOST_AUTO_TEST_CASE(TfidfEncodingTest)
{
  std::vector<string>arr(3);
  arr[0] = "hello how are you";
  arr[1] = "i am good";
  arr[2] = "Good how are you";
  arma::mat output;
  data::TfIdf en;
  en.Encode(arr, output, SplitByChar(" "));
  arma::mat target = "0.1193 0.0440 0.0440 0.0440 0 0 0 0;"
                     "0 0 0 0 0.1590 0.1590 0.1590 0;"
                     "0 0.0440 0.0440 0.0440 0 0 0 0.1193;";
  CheckMatrices(output, target, 1e-01);
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
  data::TfIdf en;
  en.Encode(arr, output, SplitByChar(" "));
  const std::unordered_map<boost::string_view, size_t,
      boost::hash<boost::string_view>>& maps = en.Mappings();
  const std::unordered_map<boost::string_view, double,
      boost::hash<boost::string_view>>& idfmaps = en.IdfValues();

  data::TfIdf en2 = en;
  const std::unordered_map<boost::string_view, size_t,
      boost::hash<boost::string_view>>& maps2 = en2.Mappings();
  const std::unordered_map<boost::string_view, double,
      boost::hash<boost::string_view>>& idfmaps2 = en2.IdfValues();

  // Comparing both of them.
  BOOST_REQUIRE_EQUAL(maps2 == maps, true);
  BOOST_REQUIRE_EQUAL(idfmaps2 == idfmaps, true);
}

BOOST_AUTO_TEST_SUITE_END();
