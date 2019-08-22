/**
 * @file preprocess_string_test.cpp
 * @author Jeffin Sam
 *
 * Test mlpackMain() of preprocess_string_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
static const std::string testName = "PreprocessString";

#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/preprocess/preprocess_string_main.cpp>

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

void CreateFile(vector<vector<string> >& input,
                const string& filename,
                const string& columnDelimiter)
{
  ofstream file(filename);
  for (auto& row : input)
  {
    for (auto& colum : row)
      file << colum << columnDelimiter;
    file << "\n";
  }
}

vector<vector<string>> stringCleaningInput = {
  { "5", "MLpaCk. Is!' a FAst! macHInE Learning.", "7",
    "MLpaCk. Is!' a FAst! macHInE Learning.", "GsOc. is!' GreAt!." },
  { "4", "MLpaCk. Is!' a FAst! macHInE Learning.", "9",
    "MLpaCk. Is!' a FAst! macHInE Learning.", "GsOc. is!' GreAt!." }
};

vector<vector<string>> stopwords ={ { "is" }, { "a" } , { "great" } };

struct PreprocessStringTestFixture
{
 public:
  PreprocessStringTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
    CreateFile(stringCleaningInput, "string_test.txt", "\t");
    CreateFile(stringCleaningInput, "string_test.csv", ",");
    CreateFile(stopwords, "stopwords.txt", "");
  }

  ~PreprocessStringTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
    remove("string_test.txt");
    remove("string_test.csv");
    remove("stopwords.txt");
  }
};

BOOST_FIXTURE_TEST_SUITE(PreprocessStringMainTest,
                         PreprocessStringTestFixture);
/**
 * Convert lower Case from string input datasetfor txt file.
 */
BOOST_AUTO_TEST_CASE(LowerCaseTest)
{
  vector<vector<string>> stringLowerInput = {
  { "5", "mlpack. is!' a fast! machine learning.", "7",
    "mlpack. is!' a fast! machine learning.", "gsoc. is!' great!." },
  { "4", "mlpack. is!' a fast! machine learning.", "9",
    "mlpack. is!' a fast! machine learning.", "gsoc. is!' great!." }
  };
  CreateFile(stringLowerInput, "preprocess_string_lower_test.txt", "\t");
  SetInputParam<string>("actual_dataset", "string_test.txt");
  SetInputParam<string>("preprocess_dataset",
      "output_preprocess_string_lower_test.txt");
  SetInputParam<vector<string>>("dimension", {"1", "3-4"});
  SetInputParam("lowercase", true);

  mlpackMain();

  BOOST_REQUIRE_EQUAL(CompareFiles("output_preprocess_string_lower_test.txt",
      "preprocess_string_lower_test.txt"), true);
  remove("output_preprocess_string_lower_test.txt");
  remove("preprocess_string_lower_test.txt");
}

/**
 * Remove Punctuation from string input dataset, for txt file.
 */
BOOST_AUTO_TEST_CASE(PunctuationTest)
{
  vector<vector<string>> stringPunctuationInput = {
  { "5", "MLpaCk Is a FAst macHInE Learning", "7",
    "MLpaCk Is a FAst macHInE Learning", "GsOc is GreAt" },
  { "4", "MLpaCk Is a FAst macHInE Learning", "9",
    "MLpaCk Is a FAst macHInE Learning", "GsOc is GreAt" }
  };
  CreateFile(stringPunctuationInput, "preprocess_string_punctuation_test.txt",
      "\t");
  SetInputParam<string>("actual_dataset", "string_test.txt");
  SetInputParam<string>("preprocess_dataset",
      "output_preprocess_punctuation_test.txt");
  SetInputParam<vector<string>>("dimension", {"1", "3-4"});
  SetInputParam("punctuation", true);

  mlpackMain();
  BOOST_REQUIRE_EQUAL(CompareFiles("output_preprocess_punctuation_test.txt",
      "preprocess_string_punctuation_test.txt"), true);
  remove("output_preprocess_punctuation_test.txt");
  remove("preprocess_string_punctuation_test.txt");
}

/**
 * Remove Stopwords from string input dataset, for txt file.
 */
BOOST_AUTO_TEST_CASE(StopWordsTest)
{
  vector<vector<string>> stringStopWordsInput = {
  { "5", "mlpack fast machine learning", "7",
    "mlpack fast machine learning", "gsoc" },
  { "4", "mlpack fast machine learning", "9",
    "mlpack fast machine learning", "gsoc" }
  };
  CreateFile(stringStopWordsInput, "preprocess_string_stop_words_test.txt",
      "\t");
  SetInputParam<string>("actual_dataset", "string_test.txt");
  SetInputParam<string>("preprocess_dataset",
      "output_preprocess_stop_words_test.txt");
  SetInputParam<vector<string>>("dimension", {"1", "3-4"});
  SetInputParam("punctuation", true);
  SetInputParam("lowercase", true);
  SetInputParam("stopwords", true);
  SetInputParam<string>("stopwordsfile", "stopwords.txt");

  mlpackMain();
  BOOST_REQUIRE_EQUAL(CompareFiles("output_preprocess_stop_words_test.txt",
      "preprocess_string_stop_words_test.txt"), true);
  remove("output_preprocess_stop_words_test.txt");
  remove("preprocess_string_stop_words_test.txt");
}

/**
 * Convert lower Case from string input datasetfor csv file.
 */
BOOST_AUTO_TEST_CASE(CsvLowerCaseTest)
{
  vector<vector<string>> stringLowerInput = {
  { "5", "mlpack. is!' a fast! machine learning.", "7",
    "mlpack. is!' a fast! machine learning.", "gsoc. is!' great!." },
  { "4", "mlpack. is!' a fast! machine learning.", "9",
    "mlpack. is!' a fast! machine learning.", "gsoc. is!' great!." }
  };
  CreateFile(stringLowerInput, "preprocess_string_lower_test.csv", ",");
  SetInputParam<string>("actual_dataset", "string_test.csv");
  SetInputParam<string>("preprocess_dataset",
      "output_preprocess_string_lower_test.csv");
  SetInputParam<vector<string>>("dimension", {"1", "3-4"});
  SetInputParam("lowercase", true);

  mlpackMain();

  BOOST_REQUIRE_EQUAL(CompareFiles("output_preprocess_string_lower_test.csv",
      "preprocess_string_lower_test.csv"), true);
  remove("output_preprocess_string_lower_test.csv");
  remove("preprocess_string_lower_test.csv");
}

/**
 * Remove Punctuation from string input dataset, for csv file.
 */
BOOST_AUTO_TEST_CASE(CsvPunctuationTest)
{
  vector<vector<string>> stringPunctuationInput = {
  { "5", "MLpaCk Is a FAst macHInE Learning", "7",
    "MLpaCk Is a FAst macHInE Learning", "GsOc is GreAt" },
  { "4", "MLpaCk Is a FAst macHInE Learning", "9",
    "MLpaCk Is a FAst macHInE Learning", "GsOc is GreAt" }
  };
  CreateFile(stringPunctuationInput, "preprocess_string_punctuation_test.csv",
      ",");
  SetInputParam<string>("actual_dataset", "string_test.csv");
  SetInputParam<string>("preprocess_dataset",
      "output_preprocess_punctuation_test.csv");
  SetInputParam<vector<string>>("dimension", {"1", "3-4"});
  SetInputParam("punctuation", true);

  mlpackMain();
  BOOST_REQUIRE_EQUAL(CompareFiles("output_preprocess_punctuation_test.csv",
      "preprocess_string_punctuation_test.csv"), true);
  remove("output_preprocess_punctuation_test.csv");
  remove("preprocess_string_punctuation_test.csv");
}

/**
 * Remove Stopwords from string input dataset, for txt file.
 */
BOOST_AUTO_TEST_CASE(CsvStopWordsTest)
{
  vector<vector<string>> stringStopWordsInput = {
  { "5", "mlpack fast machine learning", "7",
    "mlpack fast machine learning", "gsoc" },
  { "4", "mlpack fast machine learning", "9",
    "mlpack fast machine learning", "gsoc" } 
  };
  CreateFile(stringStopWordsInput, "preprocess_string_stop_words_test.csv",
      ",");
  SetInputParam<string>("actual_dataset", "string_test.csv");
  SetInputParam<string>("preprocess_dataset",
      "output_preprocess_stop_words_test.csv");
  SetInputParam<vector<string>>("dimension", {"1", "3-4"});
  SetInputParam("punctuation", true);
  SetInputParam("lowercase", true);
  SetInputParam("stopwords", true);
  SetInputParam<string>("stopwordsfile", "stopwords.txt");

  mlpackMain();
  BOOST_REQUIRE_EQUAL(CompareFiles("output_preprocess_stop_words_test.csv",
      "preprocess_string_stop_words_test.csv"), true);
  remove("output_preprocess_stop_words_test.csv");
  remove("preprocess_string_stop_words_test.csv");
}

/**
 * Invalid row or dimension number throws error.
 */
BOOST_AUTO_TEST_CASE(InvalidDimesnionTest)
{
  vector<vector<string>> stringStopWordsInput = {
  { "5", "mlpack fast machine learning", "7",
    "mlpack fast machine learning", "gsoc" },
  { "4", "mlpack fast machine learning", "9",
    "mlpack fast machine learning", "gsoc" } 
  };
  CreateFile(stringStopWordsInput, "preprocess_string_stop_words_test.csv",
      ",");
  SetInputParam<string>("actual_dataset", "string_test.csv");
  SetInputParam<string>("preprocess_dataset",
      "output_preprocess_stop_words_test.csv");
  SetInputParam<vector<string>>("dimension", {"1", "3-6"});
  SetInputParam("punctuation", true);
  SetInputParam("lowercase", true);
  SetInputParam("stopwords", true);
  SetInputParam<string>("stopwordsfile", "stopwords.txt");

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  remove("output_preprocess_stop_words_test.csv");
  remove("preprocess_string_stop_words_test.csv");
}

/**
 * Check if no stop words file throws error.
 */
BOOST_AUTO_TEST_CASE(NoStopwordsFileTest)
{
  vector<vector<string>> stringStopWordsInput = {
  { "5", "mlpack fast machine learning", "7",
    "mlpack fast machine learning", "gsoc" },
  { "4", "mlpack fast machine learning", "9",
    "mlpack fast machine learning", "gsoc" } 
  };
  CreateFile(stringStopWordsInput, "preprocess_string_stop_words_test.csv",
      ",");
  SetInputParam<string>("actual_dataset", "string_test.csv");
  SetInputParam<string>("preprocess_dataset",
      "output_preprocess_stop_words_test.csv");
  SetInputParam<vector<string>>("dimension", {"1", "3-4"});
  SetInputParam("punctuation", true);
  SetInputParam("lowercase", true);
  SetInputParam("stopwords", true);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  remove("output_preprocess_stop_words_test.csv");
  remove("preprocess_string_stop_words_test.csv");
}

/**
 * Check if dimenaion with characters throw error.
 */
BOOST_AUTO_TEST_CASE(CharDimesionTest)
{
  vector<vector<string>> stringStopWordsInput = {
  { "5", "mlpack fast machine learning", "7",
    "mlpack fast machine learning", "gsoc" },
  { "4", "mlpack fast machine learning", "9",
    "mlpack fast machine learning", "gsoc" } 
  };
  CreateFile(stringStopWordsInput, "preprocess_string_stop_words_test.csv",
      ",");
  SetInputParam<string>("actual_dataset", "string_test.csv");
  SetInputParam<string>("preprocess_dataset",
      "output_preprocess_stop_words_test.csv");
  SetInputParam<vector<string>>("dimension", {"1", "a1b-4"});
  SetInputParam("punctuation", true);
  SetInputParam("lowercase", true);
  SetInputParam("stopwords", true);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  remove("output_preprocess_stop_words_test.csv");
  remove("preprocess_string_stop_words_test.csv");
}

/**
 * Check if not output file throws an error.
 */
BOOST_AUTO_TEST_CASE(NoOutputFileTest)
{
  vector<vector<string>> stringLowerInput = {
  { "5", "mlpack. is!' a fast! machine learning.", "7",
    "mlpack. is!' a fast! machine learning.", "gsoc. is!' great!." },
  { "4", "mlpack. is!' a fast! machine learning.", "9",
    "mlpack. is!' a fast! machine learning.", "gsoc. is!' great!." }
  };
  CreateFile(stringLowerInput, "preprocess_string_lower_test.txt", "\t");

  SetInputParam<string>("actual_dataset", "string_test.txt");
  SetInputParam<vector<string>>("dimension", {"1", "3-4"});
  SetInputParam("lowercase", true);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  remove("output_preprocess_string_lower_test.txt");
  remove("preprocess_string_lower_test.txt");
}

/**
 * Check if negative dimesnion throws an error.
 */
BOOST_AUTO_TEST_CASE(NegativeDimesionTest)
{
  vector<vector<string>> stringLowerInput = {
  { "5", "mlpack. is!' a fast! machine learning.", "7",
    "mlpack. is!' a fast! machine learning.", "gsoc. is!' great!." },
  { "4", "mlpack. is!' a fast! machine learning.", "9",
    "mlpack. is!' a fast! machine learning.", "gsoc. is!' great!." }
  };
  CreateFile(stringLowerInput, "preprocess_string_lower_test.txt", "\t");

  SetInputParam<string>("actual_dataset", "string_test.txt");
  SetInputParam<vector<string>>("dimension", {"-1", "3-4"});
  SetInputParam("lowercase", true);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  remove("output_preprocess_string_lower_test.txt");
  remove("preprocess_string_lower_test.txt");
}

/**
 * Check if invalide column delimiter throws an error.
 */
BOOST_AUTO_TEST_CASE(invalidColumDelimiterTest)
{
  vector<vector<string>> stringLowerInput = {
  { "5", "mlpack. is!' a fast! machine learning.", "7",
    "mlpack. is!' a fast! machine learning.", "gsoc. is!' great!." },
  { "4", "mlpack. is!' a fast! machine learning.", "9",
    "mlpack. is!' a fast! machine learning.", "gsoc. is!' great!." }
  };
  CreateFile(stringLowerInput, "preprocess_string_lower_test.txt", "\t");

  SetInputParam<string>("actual_dataset", "string_test.txt");
  SetInputParam<vector<string>>("dimension", {"-1", "3-4"});
  SetInputParam("lowercase", true);
  SetInputParam<string>("column_delimiter", "@");

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  remove("output_preprocess_string_lower_test.txt");
  remove("preprocess_string_lower_test.txt");
}

BOOST_AUTO_TEST_SUITE_END();
