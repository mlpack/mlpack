/**
 * @file tests/string_encoding_test.cpp
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
#include <mlpack/core/data/string_encoding_policies/bag_of_words_encoding_policy.hpp>
#include <mlpack/core/data/string_encoding_policies/tf_idf_encoding_policy.hpp>
#include <boost/test/unit_test.hpp>
#include <memory>
#include "test_catch_tools.hpp"
#include "catch.hpp"
#include "serialization_catch.hpp"

using namespace mlpack;
using namespace mlpack::data;
using namespace std;

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
 * Check the values of two 2D vectors.
 */
template<typename ValueType>
void CheckVectors(const vector<vector<ValueType>>& a,
                  const vector<vector<ValueType>>& b,
                  const ValueType tolerance = 1e-5)
{
  REQUIRE(a.size() == b.size());

  for (size_t i = 0; i < a.size(); ++i)
  {
    REQUIRE(a[i].size() == b[i].size());

    for (size_t j = 0; j < a[i].size(); ++j)
      REQUIRE(a[i][j] == Approx(b[i][j]).epsilon(tolerance / 100));
  }
}

/**
 * Test the dictionary encoding algorithm.
 */
TEST_CASE("DictionaryEncodingTest", "[StringEncodingTest]")
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

    REQUIRE(keysCount[keyValue.second] == 1);
  }

  arma::mat expected = {
    {  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,  0,
       0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 },
    { 17,  2, 18, 14, 19, 20,  9, 10, 21, 14, 22,  6, 23, 14, 24, 20, 25,
      26, 27,  9, 10, 28,  6, 29, 30, 20, 31, 32, 33, 34,  9, 10, 35 },
    { 36, 37, 14, 38, 39,  8, 40,  1, 41, 42, 43, 44,  6, 45, 13,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 }
  };

  CheckMatrices(output, expected.t());
}

/**
 * Test the dictionary encoding algorithm with unicode characters.
 */
TEST_CASE("UnicodeDictionaryEncodingTest", "[StringEncodingTest]")
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

    REQUIRE(keysCount[keyValue.second] == 1);
  }

  arma::mat expected = {
    { 1, 2, 0, 0, 0 },
    { 3, 4, 1, 0, 0 },
    { 5, 2, 3, 5, 4 }
  };

  CheckMatrices(output, expected.t());
}

/**
 * Test the one pass modification of the dictionary encoding algorithm.
 */
TEST_CASE("OnePassDictionaryEncodingTest", "[StringEncodingTest]")
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

    REQUIRE(keysCount[keyValue.second] == 1);
  }

  vector<vector<size_t>> expected = {
    {  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16 },
    { 17,  2, 18, 14, 19, 20,  9, 10, 21, 14, 22,  6, 23, 14, 24, 20, 25,
      26, 27,  9, 10, 28,  6, 29, 30, 20, 31, 32, 33, 34,  9, 10, 35 },
    { 36, 37, 14, 38, 39,  8, 40,  1, 41, 42, 43, 44,  6, 45, 13 }
  };

  REQUIRE(output == expected);
}


/**
 * Test the SplitByAnyOf tokenizer.
 */
TEST_CASE("SplitByAnyOfTokenizerTest", "[StringEncodingTest]")
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

  REQUIRE(tokens.size() == expected.size());

  for (size_t i = 0; i < tokens.size(); ++i)
    REQUIRE(tokens[i] == expected[i]);
}

/**
 * Test the SplitByAnyOf tokenizer in case of unicode characters.
 */
TEST_CASE("SplitByAnyOfTokenizerUnicodeTest", "[StringEncodingTest]")
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

  REQUIRE(tokens.size() == expectedUtf8Tokens.size());

  for (size_t i = 0; i < tokens.size(); ++i)
    REQUIRE(tokens[i] == expectedUtf8Tokens[i]);
}

/**
 * Test the CharExtract tokenizer.
 */
TEST_CASE("DictionaryEncodingIndividualCharactersTest", "[StringEncodingTest]")
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
  CheckMatrices(output, target.t());
}

/**
 * Test the one pass modification of the dictionary encoding algorithm
 * in case of individual character encoding.
 */
TEST_CASE("OnePassDictionaryEncodingIndividualCharactersTest", "[StringEncodingTest]")
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

  REQUIRE(output == expected);
}

/**
 * Test the functionality of copy constructor.
 */
TEST_CASE("StringEncodingCopyTest", "[StringEncodingTest]")
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

  REQUIRE(naiveDictionary.size() == copiedDictionary.Size());

  for (const pair<string, size_t>& keyValue : naiveDictionary)
  {
    REQUIRE(copiedDictionary.HasToken(keyValue.first));
    REQUIRE(copiedDictionary.Value(keyValue.first) ==
        keyValue.second);
  }
}

/**
 * Test the move assignment operator.
 */
TEST_CASE("StringEncodingMoveTest", "[StringEncodingTest]")
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

  REQUIRE(naiveDictionary.size() == copiedDictionary.Size());

  for (const pair<string, size_t>& keyValue : naiveDictionary)
  {
    REQUIRE(copiedDictionary.HasToken(keyValue.first));
    REQUIRE(copiedDictionary.Value(keyValue.first) ==
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

  REQUIRE(mapping.size() == expectedMapping.size());

  for (auto& keyVal : expectedMapping)
  {
    REQUIRE(mapping.at(keyVal.first) == keyVal.second);
  }

  for (auto& keyVal : mapping)
  {
    REQUIRE(expectedMapping.at(keyVal.first) == keyVal.second);
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

  REQUIRE(tokens.size() == expectedTokens.size());
  REQUIRE(mapping.size() == expectedMapping.size());
  REQUIRE(mapping.size() == tokens.size());

  for (size_t i = 0; i < tokens.size(); ++i)
  {
    REQUIRE(tokens[i] == expectedTokens[i]);
    REQUIRE(expectedMapping.at(tokens[i]) == mapping.at(tokens[i]));
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

  REQUIRE(expected.Size() == obtained.Size());

  for (size_t i = 0; i < mapping.size(); ++i)
  {
    REQUIRE(mapping[i] == expectedMapping[i]);
  }
}

/**
 * Serialization test for the general template of the StringEncodingDictionary
 * class.
 */
TEST_CASE("StringEncodingDictionarySerialization", "[StringEncodingTest]")
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
TEST_CASE("SplitByAnyOfDictionaryEncodingSerialization", "[StringEncodingTest]")
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
TEST_CASE("CharExtractDictionaryEncodingSerialization", "[StringEncodingTest]")
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

/**
 * Test the Bag of Words encoding algorithm.
 */ 
TEST_CASE("BagOfWordsEncodingTest", "[StringEncodingTest]")
{
  using DictionaryType = StringEncodingDictionary<boost::string_view>;

  arma::mat output;
  BagOfWordsEncoding<SplitByAnyOf::TokenType> encoder;
  SplitByAnyOf tokenizer(" ,.");

  encoder.Encode(stringEncodingInput, output, tokenizer);

  const DictionaryType& dictionary = encoder.Dictionary();

  // Checking that each token has a unique label.
  std::unordered_map<size_t, size_t> keysCount;

  for (auto& keyValue : dictionary.Mapping())
  {
    keysCount[keyValue.second]++;

    REQUIRE(keysCount[keyValue.second] == 1);
  }

/* The expected values were obtained by the following Python script:

  from sklearn.feature_extraction.text import CountVectorizer
  from collections import OrderedDict
  import re

  string_encoding_input = [
      "mlpack is an intuitive, fast, and flexible C++ machine learning library "
      "with bindings to other languages. ",
      "It is meant to be a machine learning analog to LAPACK, and aims to "
      "implement a wide array of machine learning methods and functions "
      "as a \"swiss army knife\" for machine learning researchers.",
      "In addition to its powerful C++ interface, mlpack also provides "
      "command-line programs and Python bindings."
  ]

  dictionary = OrderedDict()

  count = 0
  for line in string_encoding_input:
      for word in re.split(' |,|\.', line):
        if word and (not (word in dictionary)):
            dictionary[word] = count
            count += 1

  def tokenizer(line):
      return re.split(' |,|\.', line)

  vectorizer = CountVectorizer(strip_accents=False, lowercase=False,
      preprocessor=None, tokenizer=tokenizer, stop_words=None,
      vocabulary=dictionary, binary=False)

  X = vectorizer.fit_transform(string_encoding_input)

  for row in X.toarray():
      print("{ " + ", ".join(map(str, row)) + " },")
*/

  arma::mat expected = {
    { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 1, 0, 0, 0, 2, 0, 0, 3, 3, 0, 0, 0, 3, 0, 0, 1, 1, 1, 3, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }
  };

  CheckMatrices(output, expected.t());
}

/**
 * Test the Bag of Words encoding algorithm. The output is saved into a vector.
 */ 
TEST_CASE("VectorBagOfWordsEncodingTest", "[StringEncodingTest]")
{
  using DictionaryType = StringEncodingDictionary<boost::string_view>;

  vector<vector<size_t>> output;
  BagOfWordsEncoding<SplitByAnyOf::TokenType> encoder(
      (BagOfWordsEncodingPolicy()));
  SplitByAnyOf tokenizer(" ,.");

  encoder.Encode(stringEncodingInput, output, tokenizer);

  const DictionaryType& dictionary = encoder.Dictionary();

  // Checking that each token has a unique label.
  std::unordered_map<size_t, size_t> keysCount;

  for (auto& keyValue : dictionary.Mapping())
  {
    keysCount[keyValue.second]++;

    REQUIRE(keysCount[keyValue.second] == 1);
  }

  /* The expected values were obtained by the same script as in
     BagOfWordsEncodingTest. */
  vector<vector<size_t>> expected = {
    { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 1, 0, 0, 0, 2, 0, 0, 3, 3, 0, 0, 0, 3, 0, 0, 1, 1, 1, 3, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }
  };

  REQUIRE(output == expected);
}

/**
 * Test the Bag of Words algorithm for individual characters.
 */
TEST_CASE("BagOfWordsEncodingIndividualCharactersTest", "[StringEncodingTest]")
{
  vector<string> input = {
    "GACCA",
    "ABCABCD",
    "GAB"
  };

  arma::mat output;
  BagOfWordsEncoding<CharExtract::TokenType> encoder;

  encoder.Encode(input, output, CharExtract());

  arma::mat target = {
    { 1, 2, 2, 0, 0 },
    { 0, 2, 2, 2, 1 },
    { 1, 1, 0, 1, 0 }
  };

  CheckMatrices(output, target.t());
}

/**
 * Test the Bag of Words encoding algorithm in case of individual
 * characters encoding. The output type is vector<vector<size_t>>.
 */
TEST_CASE("VectorBagOfWordsEncodingIndividualCharactersTest", "[StringEncodingTest]")
{
  std::vector<string> input = {
    "GACCA",
    "ABCABCD",
    "GAB"
  };

  vector<vector<size_t>> output;
  BagOfWordsEncoding<CharExtract::TokenType> encoder;

  encoder.Encode(input, output, CharExtract());

  vector<vector<size_t>> expected = {
    { 1, 2, 2, 0, 0 },
    { 0, 2, 2, 2, 1 },
    { 1, 1, 0, 1, 0 }
  };

  REQUIRE(output == expected);
}

/**
 * Test the Tf-Idf encoding algorithm with the raw count term frequency type
 * and the smooth inverse document frequency type. These parameters are
 * the default ones.
 */
TEST_CASE("RawCountSmoothIdfEncodingTest", "[StringEncodingTest]")
{
  using DictionaryType = StringEncodingDictionary<boost::string_view>;

  arma::mat output;
  TfIdfEncoding<SplitByAnyOf::TokenType> encoder;
  SplitByAnyOf tokenizer(" ,.");

  encoder.Encode(stringEncodingInput, output, tokenizer);
  const DictionaryType& dictionary = encoder.Dictionary();

  // Checking that each token has a unique label.
  std::unordered_map<size_t, size_t> keysCount;

  for (auto& keyValue : dictionary.Mapping())
  {
    keysCount[keyValue.second]++;

    REQUIRE(keysCount[keyValue.second] == 1);
  }

  /* The expected values were obtained by the following Python script:

  from sklearn.feature_extraction.text import TfidfVectorizer
  from collections import OrderedDict
  import re

  string_encoding_input = [
      "mlpack is an intuitive, fast, and flexible C++ machine learning library "
      "with bindings to other languages. ",
      "It is meant to be a machine learning analog to LAPACK, and aims to "
      "implement a wide array of machine learning methods and functions "
      "as a \"swiss army knife\" for machine learning researchers.",
      "In addition to its powerful C++ interface, mlpack also provides "
      "command-line programs and Python bindings."
  ]

  smooth_idf = True
  tf_type = 'raw_count'

  dictionary = OrderedDict()

  count = 0
  for line in string_encoding_input:
      for word in re.split(' |,|\.', line):
          if word and (not (word in dictionary)):
              dictionary[word] = count
              count += 1

  def tokenizer(line):
      return re.split(' |,|\.', line)

  if tf_type == 'raw_count':
      binary = False
      sublinear_tf = False
  elif tf_type == 'binary':
      binary = True
      sublinear_tf = False
  elif tf_type == 'sublinear_tf':
      binary = False
      sublinear_tf = True

  vectorizer = TfidfVectorizer(strip_accents=False, lowercase=False,
      preprocessor=None, tokenizer=tokenizer, stop_words=None,
      vocabulary=dictionary, binary=binary, norm=None, smooth_idf=smooth_idf,
      sublinear_tf=sublinear_tf)

  X = vectorizer.fit_transform(string_encoding_input)

  def format_result(value):
      if value == int(value):
          return str(int(value))
      else:
          return "{0:.8f}".format(value)

  for row in X.toarray():
      print("{ " + ", ".join(map(format_result, row)) + " },")
  */
  arma::mat expected = {
    { 1.28768207, 1.28768207, 1.69314718, 1.69314718, 1.69314718, 1, 1.69314718,
      1.28768207, 1.28768207, 1.28768207, 1.69314718, 1.69314718, 1.28768207, 1,
      1.69314718, 1.69314718, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 1.28768207, 0, 0, 0, 2, 0, 0, 3.86304622, 3.86304622, 0, 0, 0, 3, 0,
      0, 1.69314718, 1.69314718, 1.69314718, 5.07944154, 1.69314718, 1.69314718,
      1.69314718, 1.69314718, 1.69314718, 1.69314718, 1.69314718, 1.69314718,
      1.69314718, 1.69314718, 1.69314718, 1.69314718, 1.69314718, 1.69314718,
      1.69314718, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 1.28768207, 0, 0, 0, 0, 1, 0, 1.28768207, 0, 0, 0, 0, 1.28768207, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.69314718,
      1.69314718, 1.69314718, 1.69314718, 1.69314718, 1.69314718, 1.69314718,
      1.69314718, 1.69314718, 1.69314718 }
  };

  CheckMatrices(output, expected.t(), 1e-6);
}

/**
 * Test the Tf-Idf encoding algorithm with the raw count term frequency type
 * and the smooth inverse document frequency type. These parameters are
 * the default ones. The output type is vector<vector<double>>.
 */
TEST_CASE("VectorRawCountSmoothIdfEncodingTest", "[StringEncodingTest]")
{
  using DictionaryType = StringEncodingDictionary<boost::string_view>;

  vector<vector<double>> output;
  TfIdfEncoding<SplitByAnyOf::TokenType> encoder(
      (TfIdfEncodingPolicy()));
  SplitByAnyOf tokenizer(" ,.");

  encoder.Encode(stringEncodingInput, output, tokenizer);

  const DictionaryType& dictionary = encoder.Dictionary();

  // Checking that each token has a unique label.
  std::unordered_map<size_t, size_t> keysCount;

  for (auto& keyValue : dictionary.Mapping())
  {
    keysCount[keyValue.second]++;

    REQUIRE(keysCount[keyValue.second] == 1);
  }

  /* The expected values were obtained by the same script as in
     RawCountSmoothIdfEncodingTest. */
  vector<vector<double>> expected = {
    { 1.28768207, 1.28768207, 1.69314718, 1.69314718, 1.69314718, 1, 1.69314718,
      1.28768207, 1.28768207, 1.28768207, 1.69314718, 1.69314718, 1.28768207, 1,
      1.69314718, 1.69314718, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 1.28768207, 0, 0, 0, 2, 0, 0, 3.86304622, 3.86304622, 0, 0, 0, 3, 0,
      0, 1.69314718, 1.69314718, 1.69314718, 5.07944154, 1.69314718, 1.69314718,
      1.69314718, 1.69314718, 1.69314718, 1.69314718, 1.69314718, 1.69314718,
      1.69314718, 1.69314718, 1.69314718, 1.69314718, 1.69314718, 1.69314718,
      1.69314718, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 1.28768207, 0, 0, 0, 0, 1, 0, 1.28768207, 0, 0, 0, 0, 1.28768207, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.69314718,
      1.69314718, 1.69314718, 1.69314718, 1.69314718, 1.69314718, 1.69314718,
      1.69314718, 1.69314718, 1.69314718 }
  };
  CheckVectors(output, expected, 1e-6);
}

/**
 * Test the Tf-Idf encoding algorithm for individual characters with the
 * raw count term frequency type and the smooth inverse document frequency type.
 * These parameters are the default ones.
 */
TEST_CASE("RawCountSmoothIdfEncodingIndividualCharactersTest", "[StringEncodingTest]")
{
  vector<string> input = {
    "GACCA",
    "ABCABCD",
    "GAB"
  };

  arma::mat output;
  TfIdfEncoding<CharExtract::TokenType> encoder;

  encoder.Encode(input, output, CharExtract());

  /* The expected values were obtained by the following Python script:

  from sklearn.feature_extraction.text import TfidfVectorizer
  from collections import OrderedDict
  import re

  input_string = [
      "GACCA",
      "ABCABCD",
      "GAB"
  ]

  smooth_idf = True
  tf_type = 'raw_count'

  dictionary = OrderedDict()

  count = 0
  for line in input_string:
      for word in list(line):
          if word and (not (word in dictionary)):
              dictionary[word] = count
              count += 1

  def tokenizer(line):
      return list(line)

  if tf_type == 'raw_count':
      binary = False
      sublinear_tf = False
  elif tf_type == 'binary':
      binary = True
      sublinear_tf = False
  elif tf_type == 'sublinear_tf':
      binary = False
      sublinear_tf = True

  vectorizer = TfidfVectorizer(strip_accents=False, lowercase=False,
      preprocessor=None, tokenizer=tokenizer, stop_words=None,
      vocabulary=dictionary, binary=binary, norm=None, smooth_idf=smooth_idf,
      sublinear_tf=sublinear_tf)

  X = vectorizer.fit_transform(input_string)

  def format_result(value):
      if value == int(value):
          return str(int(value))
      else:
          return "{0:.14f}".format(value)

  for row in X.toarray():
      print("{ " + ", ".join(map(format_result, row)) + " },")
  */
  arma::mat target = {
    { 1.28768207245178, 2, 2.57536414490356, 0, 0 },
    { 0, 2, 2.57536414490356, 2.57536414490356, 1.69314718055995 },
    { 1.28768207245178, 1, 0, 1.28768207245178, 0 }
  };

  CheckMatrices(output, target.t(), 1e-12);
}

/**
 * Test the Tf-Idf encoding algorithm for individual characters with the
 * raw count term frequency type and the smooth inverse document frequency type.
 * These parameters are the default ones. The output type is
 * vector<vector<double>>.
 */
TEST_CASE("VectorRawCountSmoothIdfEncodingIndividualCharactersTest", "[StringEncodingTest]")
{
  std::vector<string> input = {
    "GACCA",
    "ABCABCD",
    "GAB"
  };

  vector<vector<double>> output;
  TfIdfEncoding<CharExtract::TokenType> encoder;

  encoder.Encode(input, output, CharExtract());

  /* The expected values were obtained by the same script as in
     RawCountSmoothIdfEncodingIndividualCharactersTest. */
  vector<vector<double>> expected = {
    { 1.28768207245178, 2, 2.57536414490356, 0, 0 },
    { 0, 2, 2.57536414490356, 2.57536414490356, 1.69314718055995 },
    { 1.28768207245178, 1, 0, 1.28768207245178, 0 }
  };

  CheckVectors(output, expected, 1e-12);
}

/**
 * Test the Tf-Idf encoding algorithm with the raw count term frequency type
 * and the non-smooth inverse document frequency type.
 */
TEST_CASE("TfIdfRawCountEncodingTest", "[StringEncodingTest]")
{
  using DictionaryType = StringEncodingDictionary<boost::string_view>;

  arma::mat output;
  TfIdfEncoding<SplitByAnyOf::TokenType> encoder(
      TfIdfEncodingPolicy(TfIdfEncodingPolicy::TfTypes::RAW_COUNT, false));
  SplitByAnyOf tokenizer(" ,.");

  encoder.Encode(stringEncodingInput, output, tokenizer);

  const DictionaryType& dictionary = encoder.Dictionary();

  // Checking that each token has a unique label.
  std::unordered_map<size_t, size_t> keysCount;

  for (auto& keyValue : dictionary.Mapping())
  {
    keysCount[keyValue.second]++;

    REQUIRE(keysCount[keyValue.second] == 1);
  }

  /* The expected values were obtained by almost the same script as in
     RawCountSmoothIdfEncodingTest. The only difference is smooth_idf equals
     False. */
  arma::mat expected = {
    { 1.40546511, 1.40546511, 2.09861229, 2.09861229, 2.09861229, 1, 2.09861229,
      1.40546511, 1.40546511, 1.40546511, 2.09861229, 2.09861229, 1.40546511, 1,
      2.09861229, 2.09861229, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 1.40546511, 0, 0, 0, 2, 0, 0, 4.21639532, 4.21639532, 0, 0, 0, 3, 0, 0,
      2.09861229, 2.09861229, 2.09861229, 6.29583687, 2.09861229, 2.09861229,
      2.09861229, 2.09861229, 2.09861229, 2.09861229, 2.09861229, 2.09861229,
      2.09861229, 2.09861229, 2.09861229, 2.09861229, 2.09861229, 2.09861229,
      2.09861229, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 1.40546511, 0, 0, 0, 0, 1, 0, 1.40546511, 0, 0, 0, 0, 1.40546511, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.09861229,
      2.09861229, 2.09861229, 2.09861229, 2.09861229, 2.09861229, 2.09861229,
      2.09861229, 2.09861229, 2.09861229 }
  };

  CheckMatrices(output, expected.t(), 1e-6);
}

/**
 * Test the Tf-Idf encoding algorithm with the raw count term frequency type
 * and the non-smooth inverse document frequency type. The output type is
 * vector<vector<double>>.
 */
TEST_CASE("VectorTfIdfRawCountEncodingTest", "[StringEncodingTest]")
{
  using DictionaryType = StringEncodingDictionary<boost::string_view>;

  vector<vector<double>> output;
  TfIdfEncoding<SplitByAnyOf::TokenType>
      encoder(TfIdfEncodingPolicy::TfTypes::RAW_COUNT, false);
  SplitByAnyOf tokenizer(" ,.");

  encoder.Encode(stringEncodingInput, output, tokenizer);

  const DictionaryType& dictionary = encoder.Dictionary();

  // Checking that each token has a unique label.
  std::unordered_map<size_t, size_t> keysCount;
  for (auto& keyValue : dictionary.Mapping())
  {
    keysCount[keyValue.second]++;

    REQUIRE(keysCount[keyValue.second] == 1);
  }

  /* The expected values were obtained by almost the same script as in
     RawCountSmoothIdfEncodingTest. The only difference is smooth_idf equals
     False. */
  vector<vector<double>> expected = {
    { 1.40546511, 1.40546511, 2.09861229, 2.09861229, 2.09861229, 1, 2.09861229,
      1.40546511, 1.40546511, 1.40546511, 2.09861229, 2.09861229, 1.40546511, 1,
      2.09861229, 2.09861229, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 1.40546511, 0, 0, 0, 2, 0, 0, 4.21639532, 4.21639532, 0, 0, 0, 3, 0, 0,
      2.09861229, 2.09861229, 2.09861229, 6.29583687, 2.09861229, 2.09861229,
      2.09861229, 2.09861229, 2.09861229, 2.09861229, 2.09861229, 2.09861229,
      2.09861229, 2.09861229, 2.09861229, 2.09861229, 2.09861229, 2.09861229,
      2.09861229, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 1.40546511, 0, 0, 0, 0, 1, 0, 1.40546511, 0, 0, 0, 0, 1.40546511, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.09861229,
      2.09861229, 2.09861229, 2.09861229, 2.09861229, 2.09861229, 2.09861229,
      2.09861229, 2.09861229, 2.09861229 }
  };
  CheckVectors(output, expected, 1e-6);
}

/**
 * Test the Tf-Idf encoding algorithm for individual characters with the
 * raw count term frequency type and the non-smooth inverse document frequency
 * type.
 */
TEST_CASE("RawCountTfIdfEncodingIndividualCharactersTest", "[StringEncodingTest]")
{
  vector<string> input = {
    "GACCA",
    "ABCABCD",
    "GAB"
  };

  arma::mat output;
  TfIdfEncoding<CharExtract::TokenType> encoder(
      TfIdfEncodingPolicy::TfTypes::RAW_COUNT, false);

  encoder.Encode(input, output, CharExtract());

  /* The expected values were obtained by almost the same script as in
     RawCountSmoothIdfEncodingIndividualCharactersTest. The only difference is
     smooth_idf equals False. */
  arma::mat target = {
    { 1.40546510810816, 2, 2.81093021621633, 0, 0 },
    { 0, 2, 2.81093021621633, 2.81093021621633, 2.09861228866811 },
    { 1.40546510810816, 1, 0, 1.40546510810816, 0 }
  };

  CheckMatrices(output, target.t(), 1e-12);
}

/**
 * Test the Tf-Idf encoding algorithm for individual characters with the
 * raw count term frequency type and the non-smooth inverse document frequency
 * type. The output type is vector<vector<double>>.
 */
TEST_CASE("VectorRawCountTfIdfEncodingIndividualCharactersTest", "[StringEncodingTest]")
{
  std::vector<string> input = {
    "GACCA",
    "ABCABCD",
    "GAB"
  };

  vector<vector<double>> output;
  TfIdfEncoding<CharExtract::TokenType> encoder(
      TfIdfEncodingPolicy::TfTypes::RAW_COUNT, false);

  encoder.Encode(input, output, CharExtract());

  /* The expected values were obtained by almost the same script as in
     RawCountSmoothIdfEncodingIndividualCharactersTest. The only difference is
     smooth_idf equals False. */
  vector<vector<double>> expected = {
    { 1.40546510810816, 2, 2.81093021621633, 0, 0 },
    { 0, 2, 2.81093021621633, 2.81093021621633, 2.09861228866811 },
    { 1.40546510810816, 1, 0, 1.40546510810816, 0 }
  };

  CheckVectors(output, expected, 1e-12);
}

/**
 * Test the Tf-Idf encoding algorithm for individual characters with the
 * binary term frequency type and the smooth inverse document frequency type.
 */
TEST_CASE("BinarySmoothIdfEncodingIndividualCharactersTest", "[StringEncodingTest]")
{
  vector<string> input = {
    "GACCA",
    "ABCABCD",
    "GAB"
  };

  arma::mat output;
  TfIdfEncoding<CharExtract::TokenType> encoder(
      TfIdfEncodingPolicy::TfTypes::BINARY, true);

  encoder.Encode(input, output, CharExtract());

  /* The expected values were obtained by almost the same script as in
     RawCountSmoothIdfEncodingIndividualCharactersTest. The only difference is
     tf_type equals 'binary'. */
  arma::mat target = {
    { 1.28768207245178, 1, 1.28768207245178, 0, 0 },
    { 0, 1, 1.28768207245178, 1.28768207245178, 1.69314718055995 },
    { 1.28768207245178, 1, 0, 1.28768207245178, 0 }
  };

  CheckMatrices(output, target.t(), 1e-12);
}

/**
 * Test the Tf-Idf encoding algorithm for individual characters with the
 * binary term frequency type and the smooth inverse document frequency type.
 * The output type is vector<vector<double>>.
 */
TEST_CASE("VectorBinarySmoothIdfEncodingIndividualCharactersTest", "[StringEncodingTest]")
{
  std::vector<string> input = {
    "GACCA",
    "ABCABCD",
    "GAB"
  };

  vector<vector<double>> output;
  TfIdfEncoding<CharExtract::TokenType>
      encoder(TfIdfEncodingPolicy::TfTypes::BINARY, true);

  encoder.Encode(input, output, CharExtract());

  /* The expected values were obtained by almost the same script as in
     RawCountSmoothIdfEncodingIndividualCharactersTest. The only difference is
     tf_type equals 'binary'. */
  vector<vector<double>> expected = {
    { 1.28768207245178, 1, 1.28768207245178, 0, 0 },
    { 0, 1, 1.28768207245178, 1.28768207245178, 1.69314718055995 },
    { 1.28768207245178, 1, 0, 1.28768207245178, 0 }
  };

  CheckVectors(output, expected, 1e-12);
}

/**
 * Test the Tf-Idf encoding algorithm for individual characters with the
 * binary term frequency type and the non-smooth inverse document frequency
 * type.
 */
TEST_CASE("BinaryTfIdfEncodingIndividualCharactersTest", "[StringEncodingTest]")
{
  vector<string> input = {
    "GACCA",
    "ABCABCD",
    "GAB"
  };

  arma::mat output;
  TfIdfEncoding<CharExtract::TokenType> encoder(
      TfIdfEncodingPolicy::TfTypes::BINARY, false);

  encoder.Encode(input, output, CharExtract());

  /* The expected values were obtained by almost the same script as in
     RawCountSmoothIdfEncodingIndividualCharactersTest. The only difference is
     tf_type equals 'binary' and smooth_idf equals False. */
  arma::mat target = {
    { 1.40546510810816, 1, 1.40546510810816, 0, 0 },
    { 0, 1, 1.40546510810816, 1.40546510810816, 2.09861228866811 },
    { 1.40546510810816, 1, 0, 1.40546510810816, 0 }
  };

  CheckMatrices(output, target.t(), 1e-12);
}

/**
 * Test the Tf-Idf encoding algorithm for individual characters with the
 * sublinear term frequency type and the smooth inverse document frequency
 * type.
 */
TEST_CASE("SublinearSmoothIdfEncodingIndividualCharactersTest", "[StringEncodingTest]")
{
  vector<string> input = {
    "GACCA",
    "ABCABCD",
    "GAB"
  };

  arma::mat output;
  TfIdfEncoding<CharExtract::TokenType> encoder(
      TfIdfEncodingPolicy::TfTypes::SUBLINEAR_TF, true);

  encoder.Encode(input, output, CharExtract());

  /* The expected values were obtained by almost the same script as in
     RawCountSmoothIdfEncodingIndividualCharactersTest. The only difference is
     tf_type equals 'sublinear_tf'. */
  arma::mat target = {
    { 1.28768207245178, 1.69314718055995, 2.18023527042932, 0, 0 },
    { 0, 1.69314718055995, 2.18023527042932, 2.18023527042932,
      1.69314718055995 },
    { 1.28768207245178, 1, 0, 1.28768207245178, 0 }
  };

  CheckMatrices(output, target.t(), 1e-12);
}

/**
 * Test the Tf-Idf encoding algorithm for individual characters with the
 * sublinear term frequency type and the non-smooth inverse document frequency
 * type.
 */
TEST_CASE("SublinearTfIdfEncodingIndividualCharactersTest", "[StringEncodingTest]")
{
  vector<string> input = {
    "GACCA",
    "ABCABCD",
    "GAB"
  };

  arma::mat output;
  TfIdfEncoding<CharExtract::TokenType>
      encoder(TfIdfEncodingPolicy::TfTypes::SUBLINEAR_TF, false);

  encoder.Encode(input, output, CharExtract());

  /* The expected values were obtained by almost the same script as in
     RawCountSmoothIdfEncodingIndividualCharactersTest. The only difference is
     tf_type equals 'sublinear_tf' and smooth_idf equals False. */
  arma::mat target = {
    { 1.40546510810816, 1.69314718055995, 2.37965928516872, 0, 0 },
    { 0, 1.69314718055995, 2.37965928516872, 2.37965928516872,
      2.09861228866811 },
    { 1.40546510810816, 1, 0, 1.40546510810816, 0 }
  };

  CheckMatrices(output, target.t(), 1e-12);
}

/**
 * Test the Tf-Idf encoding algorithm for individual characters with the
 * standard term frequency type and the smooth inverse document frequency
 * type.
 */
TEST_CASE("TermFrequencySmoothIdfEncodingIndividualCharactersTest", "[StringEncodingTest]")
{
  vector<string> input = {
    "GACCA",
    "ABCABCD",
    "GAB"
  };

  arma::mat output;
  TfIdfEncoding<CharExtract::TokenType> encoder(
      TfIdfEncodingPolicy::TfTypes::TERM_FREQUENCY, true);

  encoder.Encode(input, output, CharExtract());

  /* The expected values were obtained by the following Python script:

  from sklearn.feature_extraction.text import CountVectorizer
  from sklearn.feature_extraction.text import TfidfTransformer
  from collections import OrderedDict
  import numpy as np
  import re

  input_string = [
      "GACCA",
      "ABCABCD",
      "GAB"
  ]

  smooth_idf = True

  dictionary = OrderedDict()

  count = 0
  for line in input_string:
      for word in list(line):
          if word and (not (word in dictionary)):
              dictionary[word] = count
              count += 1

  def tokenizer(line):
      return list(line)

  vectorizer = CountVectorizer(strip_accents=False, lowercase=False,
      preprocessor=None, tokenizer=tokenizer, stop_words=None,
      vocabulary=dictionary, binary=False)

  count = vectorizer.fit_transform(input_string)

  lens = np.array(list(map(len, input_string))).reshape(len(input_string), 1)

  tf = count.toarray() / lens

  transformer = TfidfTransformer(norm=None, smooth_idf=smooth_idf,
                                 sublinear_tf=False)

  X = transformer.fit_transform(tf)

  def format_result(value):
      if value == int(value):
          return str(int(value))
      else:
          return "{0:.16}".format(value)

  for row in X.toarray():
      print("{ " + ", ".join(map(format_result, row)) + " },")
  */
  arma::mat target = {
    { 0.2575364144903562, 0.4, 0.5150728289807124, 0, 0 },
    { 0, 0.2857142857142857, 0.3679091635576516, 0.3679091635576516,
      0.2418781686514208 },
    { 0.4292273574839269, 0.3333333333333333, 0, 0.4292273574839269, 0 }
  };

  CheckMatrices(output, target.t(), 1e-12);
}

/**
 * Test the Tf-Idf encoding algorithm for individual characters with the
 * standard term frequency type and the non-smooth inverse document frequency
 * type.
 */
TEST_CASE("TermFrequencyTfIdfEncodingIndividualCharactersTest", "[StringEncodingTest]")
{
  vector<string> input = {
    "GACCA",
    "ABCABCD",
    "GAB"
  };

  arma::mat output;
  TfIdfEncoding<CharExtract::TokenType> encoder(
      TfIdfEncodingPolicy::TfTypes::TERM_FREQUENCY, false);

  encoder.Encode(input, output, CharExtract());

  /* The expected values were obtained by almost the same script as in
     TermFrequencySmoothIdfEncodingIndividualCharactersTest. The only difference
     is smooth_idf equals False. */
  arma::mat target = {
    { 0.2810930216216329, 0.4, 0.5621860432432658, 0, 0 },
    { 0, 0.2857142857142857, 0.4015614594594755, 0.4015614594594755,
      0.2998017555240157 },
    { 0.4684883693693881, 0.3333333333333333, 0, 0.4684883693693881, 0 }
  };

  CheckMatrices(output, target.t(), 1e-12);
}

/**
 * Serialization test for the Tf-Idf encoding algorithm with
 * the SplitByAnyOf tokenizer.
 */
TEST_CASE("SplitByAnyOfTfIdfEncodingSerialization", "[StringEncodingTest]")
{
  using EncoderType = TfIdfEncoding<SplitByAnyOf::TokenType>;

  EncoderType encoder;
  SplitByAnyOf tokenizer(" ,.\"");
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
