/**
 * @file string_encoding_test.cpp
 * @author Jeffin Sam
 * @author Mikhail Lozhnikov
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
#include "serialization.hpp"

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

//! Common UTF-8 input for some unicode tests.
static vector<string> stringEncodingUtf8Input = {
    "mlpack "
    "\xE2\x93\x9C\xE2\x93\x9B\xE2\x93\x9F\xE2\x93\x90\xE2\x93\x92\xE2\x93\x9A",
    "MLPACK "
    "\xE2\x93\x82\xE2\x93\x81\xE2\x93\x85\xE2\x92\xB6\xE2\x92\xB8\xE2\x93\x80 "
    "mlpack",
    "\xF0\x9F\x84\xBC\xF0\x9F\x84\xBB\xF0\x9F\x84\xBF\xF0\x9F\x84\xB0"
    "\xF0\x9F\x84\xB2\xF0\x9F\x84\xBA "
    "\xE2\x93\x9C\xE2\x93\x9B\xE2\x93\x9F\xE2\x93\x90\xE2\x93\x92\xE2\x93\x9A "
    "MLPACK "
    "\xF0\x9F\x84\xBC\xF0\x9F\x84\xBB\xF0\x9F\x84\xBF\xF0\x9F\x84\xB0"
    "\xF0\x9F\x84\xB2\xF0\x9F\x84\xBA "
    "\xE2\x93\x82\xE2\x93\x81\xE2\x93\x85\xE2\x92\xB6\xE2\x92\xB8\xE2\x93\x80"
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

  // Checking that each token has a unique label.
  std::unordered_map<size_t, size_t> keysCount;

  for (auto& keyValue : dictionary.Mapping())
  {
    keysCount[keyValue.second]++;

    BOOST_REQUIRE_EQUAL(keysCount[keyValue.second], 1);
  }

  arma::mat expected = {
    {  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,  0,
       0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 },
    { 17,  2, 18, 14, 19, 20,  9, 10, 21, 14, 22,  6, 23, 14, 24, 20, 25,
      26, 27,  9, 10, 28,  6, 29, 30, 20, 31, 32, 33, 34,  9, 10, 35 },
    { 36, 37, 14, 38, 39,  8, 40,  1, 41, 42, 43, 44,  6, 45, 13,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 }
  };

  CheckMatrices(output, expected);
}

/**
 * Test the dictionary encoding algorithm with unicode characters.
 */
BOOST_AUTO_TEST_CASE(UnicodeDictionaryEncodingTest)
{
  using DictionaryType = StringEncodingDictionary<boost::string_view>;

  arma::mat output;
  DictionaryEncoding<SplitByAnyOf::TokenType> encoder;
  SplitByAnyOf tokenizer(" .,\"");

  encoder.Encode(stringEncodingUtf8Input, output, tokenizer);

  const DictionaryType& dictionary = encoder.Dictionary();

  // Checking that each token has a unique label.
  std::unordered_map<size_t, size_t> keysCount;

  for (auto& keyValue : dictionary.Mapping())
  {
    keysCount[keyValue.second]++;

    BOOST_REQUIRE_EQUAL(keysCount[keyValue.second], 1);
  }

  arma::mat expected = {
    { 1, 2, 0, 0, 0 },
    { 3, 4, 1, 0, 0 },
    { 5, 2, 3, 5, 4 }
  };

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

  // Checking that each token has a unique label.
  std::unordered_map<size_t, size_t> keysCount;

  for (auto& keyValue : dictionary.Mapping())
  {
    keysCount[keyValue.second]++;

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
 * Test the SplitByAnyOf tokenizer.
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
 * Test the SplitByAnyOf tokenizer in case of unicode characters.
 */
BOOST_AUTO_TEST_CASE(SplitByAnyOfTokenizerUnicodeTest)
{
  vector<string> expectedUtf8Tokens = {
    "\xF0\x9F\x84\xBC\xF0\x9F\x84\xBB\xF0\x9F\x84\xBF\xF0\x9F\x84\xB0"
    "\xF0\x9F\x84\xB2\xF0\x9F\x84\xBA",
    "\xE2\x93\x9C\xE2\x93\x9B\xE2\x93\x9F\xE2\x93\x90\xE2\x93\x92\xE2\x93\x9A",
    "MLPACK",
    "\xF0\x9F\x84\xBC\xF0\x9F\x84\xBB\xF0\x9F\x84\xBF\xF0\x9F\x84\xB0"
    "\xF0\x9F\x84\xB2\xF0\x9F\x84\xBA",
    "\xE2\x93\x82\xE2\x93\x81\xE2\x93\x85\xE2\x92\xB6\xE2\x92\xB8\xE2\x93\x80"
  };

  std::vector<boost::string_view> tokens;
  boost::string_view line(stringEncodingUtf8Input[2]);
  SplitByAnyOf tokenizer(" ,.");
  boost::string_view token = tokenizer(line);

  while (!token.empty())
  {
    tokens.push_back(token);
    token = tokenizer(line);
  }

  BOOST_REQUIRE_EQUAL(tokens.size(), expectedUtf8Tokens.size());

  for (size_t i = 0; i < tokens.size(); i++)
    BOOST_REQUIRE_EQUAL(tokens[i], expectedUtf8Tokens[i]);
}

/**
* Test the CharExtract tokenizer.
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

  encoder.Encode(input, output, CharExtract());

  arma::mat target = {
    { 1, 2, 3, 3, 2, 0, 0 },
    { 2, 4, 3, 2, 4, 3, 5 },
    { 1, 2, 4, 0, 0, 0, 0 }
  };
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

/**
 * The function checks that the given dictionaries contain the same data.
 */
template<typename TokenType>
void CheckDictionaries(const StringEncodingDictionary<TokenType>& expected,
                       const StringEncodingDictionary<TokenType>& obtained)
{
  // MapType is equal to std::unordered_map<Token, size_t>.
  using MapType = typename StringEncodingDictionary<TokenType>::MapType;

  const MapType& mapping = obtained.Mapping();
  const MapType& expectedMapping = expected.Mapping();

  BOOST_REQUIRE_EQUAL(mapping.size(), expectedMapping.size());

  for (auto& keyVal : expectedMapping)
  {
    BOOST_REQUIRE_EQUAL(mapping.at(keyVal.first), keyVal.second);
  }

  for (auto& keyVal : mapping)
  {
    BOOST_REQUIRE_EQUAL(expectedMapping.at(keyVal.first), keyVal.second);
  }
}

/**
 * This is a specialization of the CheckDictionaries() function for
 * the boost::string_view token type.
 */
template<>
void CheckDictionaries(
    const StringEncodingDictionary<boost::string_view>& expected,
    const StringEncodingDictionary<boost::string_view>& obtained)
{
  /* MapType is equal to
   *
   * std::unordered_map<boost::string_view,
   *                    size_t,
   *                    boost::hash<boost::string_view>>.
   */
  using MapType =
      typename StringEncodingDictionary<boost::string_view>::MapType;

  const std::deque<std::string>& expectedTokens = expected.Tokens();
  const std::deque<std::string>& tokens = obtained.Tokens();
  const MapType& expectedMapping = expected.Mapping();
  const MapType& mapping = obtained.Mapping();

  BOOST_REQUIRE_EQUAL(tokens.size(), expectedTokens.size());
  BOOST_REQUIRE_EQUAL(mapping.size(), expectedMapping.size());
  BOOST_REQUIRE_EQUAL(mapping.size(), tokens.size());

  for (size_t i = 0; i < tokens.size(); i++)
  {
    BOOST_REQUIRE_EQUAL(tokens[i], expectedTokens[i]);
    BOOST_REQUIRE_EQUAL(expectedMapping.at(tokens[i]), mapping.at(tokens[i]));
  }
}

/**
 * This is a specialization of the CheckDictionaries() function for
 * the integer token type.
 */
template<>
void CheckDictionaries(const StringEncodingDictionary<int>& expected,
                       const StringEncodingDictionary<int>& obtained)
{
  // MapType is equal to std::arry<size_t, 256>.
  using MapType = typename StringEncodingDictionary<int>::MapType;

  const MapType& expectedMapping = expected.Mapping();
  const MapType& mapping = obtained.Mapping();

  BOOST_REQUIRE_EQUAL(expected.Size(), obtained.Size());

  for (size_t i = 0; i < mapping.size(); i++)
  {
    BOOST_REQUIRE_EQUAL(mapping[i], expectedMapping[i]);
  }
}

/**
 * Serialization test for the general template of the StringEncodingDictionary
 * class.
 */
BOOST_AUTO_TEST_CASE(StringEncodingDictionarySerialization)
{
  using DictionaryType = StringEncodingDictionary<string>;

  DictionaryType dictionary;
  SplitByAnyOf tokenizer(" ,.");

  for (const string& line : stringEncodingInput)
  {
    boost::string_view lineView(line);

    boost::string_view token = tokenizer(lineView);

    while (!tokenizer.IsTokenEmpty(token))
    {
      dictionary.AddToken(string(token));

      token = tokenizer(lineView);
    }
  }

  DictionaryType xmlDictionary, textDictionary, binaryDictionary;

  SerializeObjectAll(dictionary, xmlDictionary, textDictionary,
      binaryDictionary);

  CheckDictionaries(dictionary, xmlDictionary);
  CheckDictionaries(dictionary, textDictionary);
  CheckDictionaries(dictionary, binaryDictionary);
}

/**
 * Serialization test for the dictionary encoding algorithm with
 * the SplitByAnyOf tokenizer.
 */
BOOST_AUTO_TEST_CASE(SplitByAnyOfDictionaryEncodingSerialization)
{
  using EncoderType = DictionaryEncoding<SplitByAnyOf::TokenType>;

  EncoderType encoder;
  SplitByAnyOf tokenizer(" ,.");
  arma::mat output;

  encoder.Encode(stringEncodingInput, output, tokenizer);

  EncoderType xmlEncoder, textEncoder, binaryEncoder;
  arma::mat xmlOutput, textOutput, binaryOutput;

  SerializeObjectAll(encoder, xmlEncoder, textEncoder, binaryEncoder);

  CheckDictionaries(encoder.Dictionary(), xmlEncoder.Dictionary());
  CheckDictionaries(encoder.Dictionary(), textEncoder.Dictionary());
  CheckDictionaries(encoder.Dictionary(), binaryEncoder.Dictionary());

  xmlEncoder.Encode(stringEncodingInput, xmlOutput, tokenizer);
  textEncoder.Encode(stringEncodingInput, textOutput, tokenizer);
  binaryEncoder.Encode(stringEncodingInput, binaryOutput, tokenizer);

  CheckMatrices(output, xmlOutput, textOutput, binaryOutput);
}

/**
 * Serialization test for the dictionary encoding algorithm with
 * the CharExtract tokenizer.
 */
BOOST_AUTO_TEST_CASE(CharExtractDictionaryEncodingSerialization)
{
  using EncoderType = DictionaryEncoding<CharExtract::TokenType>;

  EncoderType encoder;
  CharExtract tokenizer;
  arma::mat output;

  encoder.Encode(stringEncodingInput, output, tokenizer);

  EncoderType xmlEncoder, textEncoder, binaryEncoder;
  arma::mat xmlOutput, textOutput, binaryOutput;

  SerializeObjectAll(encoder, xmlEncoder, textEncoder, binaryEncoder);

  CheckDictionaries(encoder.Dictionary(), xmlEncoder.Dictionary());
  CheckDictionaries(encoder.Dictionary(), textEncoder.Dictionary());
  CheckDictionaries(encoder.Dictionary(), binaryEncoder.Dictionary());

  xmlEncoder.Encode(stringEncodingInput, xmlOutput, tokenizer);
  textEncoder.Encode(stringEncodingInput, textOutput, tokenizer);
  binaryEncoder.Encode(stringEncodingInput, binaryOutput, tokenizer);

  CheckMatrices(output, xmlOutput, textOutput, binaryOutput);
}

BOOST_AUTO_TEST_SUITE_END();

