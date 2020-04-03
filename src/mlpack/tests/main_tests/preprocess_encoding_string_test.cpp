/**
 * @file preprocess_encoding_string_test.cpp
 * @author Jeffin Sam
 *
 * Test mlpackMain() of preprocess_encoding_string_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
static const std::string testName = "PreprocessEncodingString";

#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/preprocess/preprocess_string_encoding_main.cpp>

#include "test_helper.hpp"
#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;
using namespace std;

bool CompareFiles(const string& fileName1, const string& fileName2)
{
  ifstream f1(fileName1), f2(fileName2);
  string file1line, file2line;
  if (!f1.is_open())
    Log::Fatal << fileName1 << " can't be opened" << endl;
  if (!f2.is_open())
    Log::Fatal << fileName2 << " can't be opened" << endl;
  while (getline(f1, file1line) && getline(f2, file2line))
  {
    for (size_t i = 0; i < file1line.length(); i++)
      if (file1line[i] != file2line[i])
        return false;
  }
  return true;
}

template<typename InputType>
void CreateFile(vector<vector<InputType> >& input,
                const string& filename,
                const string& columnDelimiter)
{
  ofstream file(filename);
  for (auto& row : input)
  {
    for (auto& colum : row)
    {
      file << colum << columnDelimiter;
    }
    file << "\n";
  }
}

//! Common input for some tests.
static vector<vector<string>> stringEncodingInput = {
    { "5", "7", "mlpack is an intuitive, fast, and flexible C++ machine"
      " learning library with bindings to other languages. " },
    { "10", "12", "It is meant to be a machine learning analog to LAPACK, and "
      "aims to implement a wide array of machine learning methods and "
      "functions as a \"swiss army knife\" for machine learning researchers."},
    { "9", "19", "In addition to its powerful C++ interface, mlpack also "
      "provides command-line programs and Python bindings." }
};

struct PreprocessEncodingStringTestFixture
{
 public:
  PreprocessEncodingStringTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
    CreateFile(stringEncodingInput, "string_test.txt", "\t");
  }

  ~PreprocessEncodingStringTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
    remove("string_test.txt");
  }
};

BOOST_FIXTURE_TEST_SUITE(PreprocessEncodingStringMainTest,
                         PreprocessEncodingStringTestFixture);
/**
 * Check for Dictionary encoding Type.
 */
BOOST_AUTO_TEST_CASE(DictionaryEncodingTest)
{
  vector<vector<size_t>> stringencodeddata = {
    {  5, 7, 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,  0,
       0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 },
    { 10, 12, 17,  2, 18, 14, 19, 20,  9, 10, 21, 14, 22,  6, 23, 14, 24, 20,
      25, 26, 27,  9, 10, 28,  6, 29, 30, 20, 31, 32, 33, 34,  9, 10, 35 },
    { 9, 19, 36, 37, 14, 38, 39,  8, 40,  1, 41, 42, 43, 44,  6, 45, 13,  0, 0,
       0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 }
  };
  CreateFile(stringencodeddata, "preprocess_string_encoded_test.txt", "\t");
  SetInputParam<string>("actual_dataset", "string_test.txt");
  SetInputParam<string>("preprocess_dataset",
      "output_preprocess_string_encoded_test.txt");
  SetInputParam<vector<string>>("dimension", {"2"});
  SetInputParam<string>("encoding_type", "DictionaryEncoding");
  SetInputParam<string>("delimiter", " .,\"");

  mlpackMain();

  BOOST_REQUIRE_EQUAL(CompareFiles("output_preprocess_string_encoded_test.txt",
      "preprocess_string_encoded_test.txt"), true);
  remove("output_preprocess_string_encoded_test.txt");
  remove("preprocess_string_encoded_test.txt");
}

/**
 * Check for Bag Of Words encoding type.
 */
BOOST_AUTO_TEST_CASE(BagOfWordsEncodingTest)
{
  vector<vector<size_t>> stringencodeddata = {
    {  5, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    {  10, 12, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {  9, 19, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }
  };
  CreateFile(stringencodeddata, "preprocess_string_encoded_test.txt", "\t");
  SetInputParam<string>("actual_dataset", "string_test.txt");
  SetInputParam<string>("preprocess_dataset",
      "output_preprocess_string_encoded_test.txt");
  SetInputParam<vector<string>>("dimension", {"2"});
  SetInputParam<string>("encoding_type", "BagOfWordsEncoding");
  SetInputParam<string>("delimiter", " .,\"");

  mlpackMain();

  BOOST_REQUIRE_EQUAL(CompareFiles("output_preprocess_string_encoded_test.txt",
      "preprocess_string_encoded_test.txt"), true);
  remove("output_preprocess_string_encoded_test.txt");
  remove("preprocess_string_encoded_test.txt");
}

/**
 * Check for Tf-Idf encoding type.
 */
BOOST_AUTO_TEST_CASE(TfIdfEncodingTest)
{
  vector<vector<double>> stringencodeddata = {
    {  5, 7, 1.28768207245178, 1.28768207245178, 1.69314718055995,
       1.69314718055995, 1.69314718055995, 1, 1.69314718055995,
       1.28768207245178, 1.28768207245178, 1.28768207245178, 1.69314718055995,
       1.69314718055995, 1.28768207245178, 1, 1.69314718055995,
       1.69314718055995, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    {  10, 12, 0, 1.28768207245178, 0, 0, 0, 2, 0, 0, 3.86304621735534,
       3.86304621735534, 0, 0, 0, 3, 0, 0, 1.69314718055995, 1.69314718055995,
       1.69314718055995, 5.07944154167984, 1.69314718055995, 1.69314718055995,
       1.69314718055995, 1.69314718055995, 1.69314718055995, 1.69314718055995,
       1.69314718055995, 1.69314718055995, 1.69314718055995, 1.69314718055995,
       1.69314718055995, 1.69314718055995, 1.69314718055995, 1.69314718055995,
       1.69314718055995, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    {  9, 19, 1.28768207245178, 0, 0, 0, 0, 1, 0, 1.28768207245178, 0, 0, 0, 0,
       1.28768207245178, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 1.69314718055995, 1.69314718055995, 1.69314718055995,
       1.69314718055995, 1.69314718055995, 1.69314718055995, 1.69314718055995,
       1.69314718055995, 1.69314718055995,  1.69314718055995 }
  };
  CreateFile(stringencodeddata, "preprocess_string_encoded_test.txt", "\t");
  SetInputParam<string>("actual_dataset", "string_test.txt");
  SetInputParam<string>("preprocess_dataset",
      "output_preprocess_string_encoded_test.txt");
  SetInputParam<vector<string>>("dimension", {"2"});
  SetInputParam<string>("encoding_type", "TfIdfEncoding");
  SetInputParam<string>("delimiter", " .,\"");

  mlpackMain();

  BOOST_REQUIRE_EQUAL(CompareFiles("output_preprocess_string_encoded_test.txt",
      "preprocess_string_encoded_test.txt"), true);
  remove("output_preprocess_string_encoded_test.txt");
  remove("preprocess_string_encoded_test.txt");
}

/**
 * Invalid row or dimension number throws error.
 */
BOOST_AUTO_TEST_CASE(InvalidDimesnionTest)
{
  vector<vector<size_t>> stringencodeddata = {
    {  5, 7, 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,  0,
       0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 },
    { 10, 12, 17,  2, 18, 14, 19, 20,  9, 10, 21, 14, 22,  6, 23, 14, 24, 20,
      25, 26, 27,  9, 10, 28,  6, 29, 30, 20, 31, 32, 33, 34,  9, 10, 35 },
    { 9, 19, 36, 37, 14, 38, 39,  8, 40,  1, 41, 42, 43, 44,  6, 45, 13,  0, 0,
       0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 }
  };
  CreateFile(stringencodeddata, "preprocess_string_encoded_test.txt", "\t");
  SetInputParam<string>("actual_dataset", "string_test.txt");
  SetInputParam<string>("preprocess_dataset",
      "output_preprocess_string_encoded_test.txt");
  SetInputParam<vector<string>>("dimension", {"1", "3-6"});
  SetInputParam<string>("encoding_type", "TfIdfEncoding");
  SetInputParam<string>("delimiter", " .,\"");

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  remove("output_preprocess_string_encoded_test.txt");
  remove("preprocess_string_encoded_test.txt");
}

/**
 * Check wether invalid encoding type throws error.
 */
BOOST_AUTO_TEST_CASE(InvalidEncodingTest)
{
  vector<vector<size_t>> stringencodeddata = {
    {  5, 7, 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,  0,
       0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 },
    { 10, 12, 17,  2, 18, 14, 19, 20,  9, 10, 21, 14, 22,  6, 23, 14, 24, 20,
      25, 26, 27,  9, 10, 28,  6, 29, 30, 20, 31, 32, 33, 34,  9, 10, 35 },
    { 9, 19, 36, 37, 14, 38, 39,  8, 40,  1, 41, 42, 43, 44,  6, 45, 13,  0, 0,
       0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 }
  };
  CreateFile(stringencodeddata, "preprocess_string_encoded_test.txt", "\t");
  SetInputParam<string>("actual_dataset", "string_test.txt");
  SetInputParam<string>("preprocess_dataset",
      "output_preprocess_string_encoded_test.txt");
  SetInputParam<vector<string>>("dimension", {"1", "3-6"});
  SetInputParam<string>("encoding_type", "InvalidEncodingTest");
  SetInputParam<string>("delimiter", " .,\"");

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  remove("output_preprocess_string_encoded_test.txt");
  remove("preprocess_string_encoded_test.txt");
}

/**
 * Check whether passing of extra option doesn't effect results.
 */
BOOST_AUTO_TEST_CASE(ExtraOptionTest)
{
  vector<vector<size_t>> stringencodeddata = {
    {  5, 7, 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,  0,
       0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 },
    { 10, 12, 17,  2, 18, 14, 19, 20,  9, 10, 21, 14, 22,  6, 23, 14, 24, 20,
      25, 26, 27,  9, 10, 28,  6, 29, 30, 20, 31, 32, 33, 34,  9, 10, 35 },
    { 9, 19, 36, 37, 14, 38, 39,  8, 40,  1, 41, 42, 43, 44,  6, 45, 13,  0, 0,
       0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 }
  };
  CreateFile(stringencodeddata, "preprocess_string_encoded_test.txt", "\t");
  SetInputParam<string>("actual_dataset", "string_test.txt");
  SetInputParam<string>("preprocess_dataset",
      "output_preprocess_string_encoded_test.txt");
  SetInputParam<vector<string>>("dimension", {"2"});
  SetInputParam<string>("encoding_type", "DictionaryEncoding");
  SetInputParam<string>("delimiter", " .,\"");
  SetInputParam("smooth_idf", true);

  mlpackMain();

  BOOST_REQUIRE_EQUAL(CompareFiles("output_preprocess_string_encoded_test.txt",
      "preprocess_string_encoded_test.txt"), true);
  remove("output_preprocess_string_encoded_test.txt");
  remove("preprocess_string_encoded_test.txt");
}

/**
 * Check whther smooth_idf really makes a different
 */
BOOST_AUTO_TEST_CASE(SmoothIdfTest)
{
  SetInputParam<string>("actual_dataset", "string_test.txt");
  SetInputParam<string>("preprocess_dataset",
      "output_smooth_idf_string_encoded_test.txt");
  SetInputParam<vector<string>>("dimension", {"2"});
  SetInputParam<string>("encoding_type", "TfIdfEncoding");
  SetInputParam<string>("delimiter", " .,\"");
  SetInputParam("smooth_idf", true);

  mlpackMain();

  SetInputParam<string>("actual_dataset", "string_test.txt");
  SetInputParam<string>("preprocess_dataset",
      "output_no_smooth_idf_encoded_test.txt");
  SetInputParam<vector<string>>("dimension", {"2"});
  SetInputParam<string>("encoding_type", "TfIdfEncoding");
  SetInputParam<string>("delimiter", " .,\"");
  SetInputParam("smooth_idf", false);

  mlpackMain();

  BOOST_REQUIRE_EQUAL(CompareFiles("output_no_smooth_idf_encoded_test.txt",
      "output_smooth_idf_string_encoded_test.txt"), false);
  remove("output_no_smooth_idf_string_encoded_test.txt");
  remove("output_no_smooth_idf_encoded_test.txt");
}

/**
 * Check whether two different encoding gives two different output.
 */
BOOST_AUTO_TEST_CASE(DifferentEncodingTest)
{
  SetInputParam<string>("actual_dataset", "string_test.txt");
  SetInputParam<string>("preprocess_dataset",
      "output_dictionary_encdoing_test.txt");
  SetInputParam<vector<string>>("dimension", {"2"});
  SetInputParam<string>("encoding_type", "DictionaryEncoding");
  SetInputParam<string>("delimiter", " .,\"");

  mlpackMain();

  SetInputParam<string>("actual_dataset", "string_test.txt");
  SetInputParam<string>("preprocess_dataset",
      "output_bow_encoding_string_encoded_test.txt");
  SetInputParam<vector<string>>("dimension", {"2"});
  SetInputParam<string>("encoding_type", "BagOfWordsEncoding");
  SetInputParam<string>("delimiter", " .,\"");

  mlpackMain();

  BOOST_REQUIRE_EQUAL(CompareFiles("output_dictionary_encdoing_test.txt",
      "output_bow_encoding_string_encoded_test.txt"), false);
  remove("output_bow_encoding_string_encoded_test.txt");
  remove("output_dictionary_encdoing_test.txt");
}

/**
 * Check whether two different Tf-Idf type gives two differnet ouput.
 */
BOOST_AUTO_TEST_CASE(DifferenTfIdfTest)
{
  SetInputParam<string>("actual_dataset", "string_test.txt");
  SetInputParam<string>("preprocess_dataset",
      "output_raw_count_encdoing_string_encoded_test.txt");
  SetInputParam<vector<string>>("dimension", {"2"});
  SetInputParam<string>("encoding_type", "TfIdfEncoding");
  SetInputParam<string>("delimiter", " .,\"");

  mlpackMain();

  SetInputParam<string>("actual_dataset", "string_test.txt");
  SetInputParam<string>("preprocess_dataset",
      "output_sublinear_tf_encoding_test.txt");
  SetInputParam<vector<string>>("dimension", {"2"});
  SetInputParam<string>("encoding_type", "TfIdfEncoding");
  SetInputParam<string>("delimiter", " .,\"");
  SetInputParam<string>("tfidf_encoding_type", "SublinearTf");

  mlpackMain();

  BOOST_REQUIRE_EQUAL(CompareFiles("output_sublinear_tf_encoding_test.txt",
      "output_raw_count_encdoing_string_encoded_test.txt"), false);
  remove("output_sublinear_tf_encoding_test.txt");
  remove("output_raw_count_encdoing_string_encoded_test.txt");
}

/**
 * Check whether invaid Tf-Idf type thrwows an error.
 */
BOOST_AUTO_TEST_CASE(InvalidTfIdfTypeEncodingTest)
{
  vector<vector<double>> stringencodeddata = {
    {  5, 7, 1.28768207245178, 1.28768207245178, 1.69314718055995,
       1.69314718055995, 1.69314718055995, 1, 1.69314718055995,
       1.28768207245178, 1.28768207245178, 1.28768207245178, 1.69314718055995,
       1.69314718055995, 1.28768207245178, 1, 1.69314718055995,
       1.69314718055995, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    {  10, 12, 0, 1.28768207245178, 0, 0, 0, 2, 0, 0, 3.86304621735534,
       3.86304621735534, 0, 0, 0, 3, 0, 0, 1.69314718055995, 1.69314718055995,
       1.69314718055995, 5.07944154167984, 1.69314718055995, 1.69314718055995,
       1.69314718055995, 1.69314718055995, 1.69314718055995, 1.69314718055995,
       1.69314718055995, 1.69314718055995, 1.69314718055995, 1.69314718055995,
       1.69314718055995, 1.69314718055995, 1.69314718055995, 1.69314718055995,
       1.69314718055995, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    {  9, 19, 1.28768207245178, 0, 0, 0, 0, 1, 0, 1.28768207245178, 0, 0, 0, 0,
       1.28768207245178, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 1.69314718055995, 1.69314718055995, 1.69314718055995,
       1.69314718055995, 1.69314718055995, 1.69314718055995, 1.69314718055995,
       1.69314718055995, 1.69314718055995,  1.69314718055995 }
  };
  CreateFile(stringencodeddata, "preprocess_string_encoded_test.txt", "\t");
  SetInputParam<string>("actual_dataset", "string_test.txt");
  SetInputParam<string>("preprocess_dataset",
      "output_preprocess_string_encoded_test.txt");
  SetInputParam<vector<string>>("dimension", {"2"});
  SetInputParam<string>("encoding_type", "TfIdfEncoding");
  SetInputParam<string>("delimiter", " .,\"");
  SetInputParam<string>("tfidf_encoding_type", "invalidetfidftype");

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  remove("output_preprocess_string_encoded_test.txt");
  remove("preprocess_string_encoded_test.txt");
}

BOOST_AUTO_TEST_SUITE_END();
