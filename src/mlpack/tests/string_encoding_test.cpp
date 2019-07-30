/**
 * @file string_encoding_test.cpp
 * @author Jeffin Sam
 *
 * Tests for the StringEncoding class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/boost_backport/boost_backport_string_view.hpp>
#include <mlpack/core/data/tokenizers/split_by_any_of.hpp>
#include <mlpack/core/data/tokenizers/char_extract.hpp>
#include <mlpack/core/data/string_encoding.hpp>
#include <mlpack/core/data/string_encoding_policies/dictionary_encoding_policy.hpp>
#include <boost/test/unit_test.hpp>
#include <memory>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::data;
using namespace std;

BOOST_AUTO_TEST_SUITE(StringEncodingTest);

//! Common input for some tests.
static vector<string> stringEncodingInput = {
    "mlpack is an intuitive, fast, and flexible C++ machine learning library "
    "with bindings to other languages. ",
    "It is meant to be a machine learning analog to LAPACK, and aims to "
    "implement a wide array of machine learning methods and functions "
    "as a \"swiss army knife\" for machine learning researchers.",
    "In addition to its powerful C++ interface, mlpack also provides "
    "command-line programs and Python bindings."
  };


/**
 * Test the dictionary encoding algorithm.
 */
BOOST_AUTO_TEST_CASE(DictionaryEncodingTest)
{
  using DictionaryType = StringEncodingDictionary<boost::string_view>;

  arma::mat output;
  DictionaryEncoding<SplitByAnyOf::TokenType> encoder;
  SplitByAnyOf tokenizer(" .,\"");
  
  encoder.Encode(stringEncodingInput, output, tokenizer);

  const DictionaryType& dictionary = encoder.Dictionary();
  
  // Checking that everything is mapped to different numbers
  std::unordered_map<size_t, size_t> keysCount;
  for (auto& keyValue : dictionary.Mapping())
  {
    keysCount[keyValue.second]++;
    // Every token should be mapped only once
    BOOST_REQUIRE_EQUAL(keysCount[keyValue.second], 1);
  }
  
  arma::mat expected = " 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16  0 "
                       " 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 ;"
                       "17  2 18 14 19 20  9 10 21 14 22  6 23 14 24 20 25 "
                       "26 27  9 10 28  6 29 30 20 31 32 33 34  9 10 35;"
                       "36 37 14 38 39  8 40  1 41 42 43 44  6 45 13  0  0 "
                       " 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 ;";

  CheckMatrices(output, expected);
}

/**
 * Test the one pass modification of the dictionary encoding algorithm.
 */
BOOST_AUTO_TEST_CASE(OnePassDictionaryEncodingTest)
{
  using DictionaryType = StringEncodingDictionary<boost::string_view>;

  vector<vector<size_t>> output;
  DictionaryEncoding<SplitByAnyOf::TokenType> encoder(
      (DictionaryEncodingPolicy()));
  SplitByAnyOf tokenizer(" .,\"");
  
  encoder.Encode(stringEncodingInput, output, tokenizer);

  const DictionaryType& dictionary = encoder.Dictionary();
  
  // Checking that everything is mapped to different numbers
  std::unordered_map<size_t, size_t> keysCount;
  for (auto& keyValue : dictionary.Mapping())
  {
    keysCount[keyValue.second]++;
    // Every token should be mapped only once
    BOOST_REQUIRE_EQUAL(keysCount[keyValue.second], 1);
  }
  
  vector<vector<size_t>> expected = {
    {  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16 },
    { 17,  2, 18, 14, 19, 20,  9, 10, 21, 14, 22,  6, 23, 14, 24, 20, 25,
      26, 27,  9, 10, 28,  6, 29, 30, 20, 31, 32, 33, 34,  9, 10, 35 },
    { 36, 37, 14, 38, 39,  8, 40,  1, 41, 42, 43, 44,  6, 45, 13 }
  };
  
  BOOST_REQUIRE(output == expected);
}


/**
 * Test for the SplitByAnyOf tokenizer.
 */
BOOST_AUTO_TEST_CASE(SplitByAnyOfTokenizerTest)
{
  std::vector<boost::string_view> tokens;
  boost::string_view line(stringEncodingInput[0]);
  SplitByAnyOf tokenizer(" ,.");
  boost::string_view token = tokenizer(line);

  while (!token.empty())
  {
    tokens.push_back(token);
    token = tokenizer(line);
  }
  
  vector<string> expected = { "mlpack", "is", "an", "intuitive", "fast",
    "and", "flexible", "C++", "machine", "learning", "library", "with",
    "bindings", "to", "other", "languages"
  };
  
  BOOST_REQUIRE_EQUAL(tokens.size(), expected.size());
  
  for (size_t i = 0; i < tokens.size(); i++)
    BOOST_REQUIRE_EQUAL(tokens[i], expected[i]);
}

/**
* Test Dictionary encoding for characters using lamda function.
*/
BOOST_AUTO_TEST_CASE(DictionaryEncodingIndividualCharactersTest)
{
  vector<string> input = {
    "GACCA",
    "ABCABCD",
    "GAB"
  };

  arma::mat output;
  DictionaryEncoding<CharExtract::TokenType> encoder;

  // Passing a empty string to encode characters
  encoder.Encode(input, output, CharExtract());

  arma::mat target = "1 2 3 3 2 0 0;"
                     "2 4 3 2 4 3 5;"
                     "1 2 4 0 0 0 0;";
  CheckMatrices(output, target);
}

/**
 * Test the one pass modification of the dictionary encoding algorithm
 * in case of individual character encoding.
 */
BOOST_AUTO_TEST_CASE(OnePassDictionaryEncodingIndividualCharactersTest)
{
  std::vector<string> input = {
    "GACCA",
    "ABCABCD",
    "GAB"
  };

  vector<vector<size_t>> output;
  DictionaryEncoding<CharExtract::TokenType> encoder;

  // Passing a empty string to encode characters
  encoder.Encode(input, output, CharExtract());

  vector<vector<size_t>> expected = {
    { 1, 2, 3, 3, 2 },
    { 2, 4, 3, 2, 4, 3, 5 },
    { 1, 2, 4 }
  };

  BOOST_REQUIRE(output == expected);
}

/**
 * Test the functionality of copy constructor.
 */
BOOST_AUTO_TEST_CASE(StringEncodingCopyTest)
{
  using DictionaryType = StringEncodingDictionary<boost::string_view>;
  arma::sp_mat output;
  DictionaryEncoding<SplitByAnyOf::TokenType> encoderCopy;
  SplitByAnyOf tokenizer(" ,.");
  
  vector<pair<string, size_t>> naiveDictionary;

  {
    DictionaryEncoding<SplitByAnyOf::TokenType> encoder;
    encoder.Encode(stringEncodingInput, output, tokenizer);
    
    for (const string& token : encoder.Dictionary().Tokens())
    {
      naiveDictionary.emplace_back(token, encoder.Dictionary().Value(token));
    }
    
    encoderCopy = DictionaryEncoding<SplitByAnyOf::TokenType>(encoder);
  }
  
  const DictionaryType& copiedDictionary = encoderCopy.Dictionary();

  BOOST_REQUIRE_EQUAL(naiveDictionary.size(), copiedDictionary.Size());

  for (const pair<string, size_t>& keyValue : naiveDictionary)
  {
    BOOST_REQUIRE(copiedDictionary.HasToken(keyValue.first));
    BOOST_REQUIRE_EQUAL(copiedDictionary.Value(keyValue.first), 
        keyValue.second);
  }
}

/**
 * Test the move assignment operator.
 */
BOOST_AUTO_TEST_CASE(StringEncodingMoveTest)
{
  using DictionaryType = StringEncodingDictionary<boost::string_view>;
  arma::sp_mat output;
  DictionaryEncoding<SplitByAnyOf::TokenType> encoderCopy;
  SplitByAnyOf tokenizer(" ,.");
  
  vector<pair<string, size_t>> naiveDictionary;

  {
    DictionaryEncoding<SplitByAnyOf::TokenType> encoder;
    encoder.Encode(stringEncodingInput, output, tokenizer);
    
    for (const string& token : encoder.Dictionary().Tokens())
    {
      naiveDictionary.emplace_back(token, encoder.Dictionary().Value(token));
    }
    
    encoderCopy = std::move(encoder);
  }
  
  const DictionaryType& copiedDictionary = encoderCopy.Dictionary();

  BOOST_REQUIRE_EQUAL(naiveDictionary.size(), copiedDictionary.Size());

  for (const pair<string, size_t>& keyValue : naiveDictionary)
  {
    BOOST_REQUIRE(copiedDictionary.HasToken(keyValue.first));
    BOOST_REQUIRE_EQUAL(copiedDictionary.Value(keyValue.first), 
        keyValue.second);
  }
}


BOOST_AUTO_TEST_SUITE_END();

