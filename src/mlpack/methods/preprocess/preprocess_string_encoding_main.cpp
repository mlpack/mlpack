/**
 * @file preprocess_string_encoding_main.cpp
 * @author Jeffin Sam
 *
 * A CLI executable to encode string dataset.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/core/math/random.hpp>
#include <mlpack/core/util/cli.hpp>
#include <mlpack/core/data/extension.hpp>
#include <unordered_set>
#include <mlpack/core/data/string_encoding.hpp>
#include <mlpack/core/data/tokenizers/split_by_any_of.hpp>
#include <mlpack/core/data/string_encoding_policies/dictionary_encoding_policy.hpp>
#include <mlpack/core/data/string_encoding_policies/bag_of_words_encoding_policy.hpp>
#include <mlpack/core/data/string_encoding_policies/tf_idf_encoding_policy.hpp>

PROGRAM_INFO("preprocess_string_encoding",
    // Short description.
    "A utility to encode string data. This utility can encode string using "
    "DictionaryEncoding, BagOfWordsEncoding and TfIdfEncoding methods.",
    // Long description.
    "This utility takes a dataset and the dimension and arguments and "
    "encodes string dataset according to arguments given."
    "\n\n"
    "The dataset may be given as the file name and the output may be saved as "
    + PRINT_PARAM_STRING("actual_dataset") + " and " +
    PRINT_PARAM_STRING("preprocess_dataset") + " ."
    "\n\n"
    " Following arguments may be given " + PRINT_PARAM_STRING("encoding_type") +
    " to encode the dataset using a specific encoding type and " + " Also the "
    "dimension which contains the string dataset " +
    PRINT_PARAM_STRING("dimension") + "."
    "\n\n"
    "So, a simple example where we want to encode string dataset " +
    PRINT_DATASET("X") + ", which is having string data in its 3 Column,"
    " using DictionaryEncoding as encoding type."
    "\n\n" +
    PRINT_CALL("preprocess_string", "actual_dataset", "X",
        "preprocess_dataset", "X", "dimension", 3, "encoding_type",
        "DictionaryEncoding") +
    "\n\n",
    SEE_ALSO("@preprocess_binarize", "#preprocess_binarize"),
    SEE_ALSO("@preprocess_describe", "#preprocess_describe"),
    SEE_ALSO("@preprocess_imputer", "#preprocess_imputer"));

PARAM_STRING_IN_REQ("actual_dataset", "File containing the reference dataset.",
    "t");
PARAM_STRING_IN_REQ("preprocess_dataset", "File containing the preprocess "
    "dataset.", "o");
PARAM_STRING_IN("column_delimiter", "delimeter used to seperate Column in files"
    "example '\\t' for '.tsv' and ',' for '.csv'.", "d", "\t");
PARAM_STRING_IN("delimiter", "A set of chars that is used as delimeter to"
    "tokenize the string dataset ", "D", " ");
PARAM_STRING_IN("encoding_type","Type of encoding","e","DictionaryEncoding")
PARAM_STRING_IN("tfidf_encoding_type","Type of tfidf encoding","E","RawCount")
PARAM_FLAG("smooth_idf", "True to have smooth_idf for Tf-Idf.", "s");
PARAM_VECTOR_IN_REQ(std::string, "dimension", "Column which contains the"
    "string data. (1 by default)", "c");

using namespace mlpack;
using namespace mlpack::util;
using namespace arma;
using namespace std;

/**
 * Function neccessary to create a vector<vector<string>> by readin
 * the contents of a file.
 *
 * @param filename Name of the file whose contents need to be preproccessed.
 * @param columnDelimiter Delimiter used to split the columns of file.
 */
static vector<vector<string>> CreateDataset(const string& filename,
                                            char columnDelimiter)
{
  vector<vector<string>> dataset;
  // Extracting the Contents of file
  // File stream.
  ifstream fin(filename);
  if (!fin.is_open())
    Log::Fatal << "Unable to open input file" << endl;
  string line, word;
  while (getline(fin, line))
  {
    stringstream streamLine(line);
    dataset.emplace_back();
    while (getline(streamLine, word, columnDelimiter))
    {
      dataset.back().push_back(word);
    }
  }
  return dataset;
}

static bool IsNumber(const string& column)
{
  for (auto i : column)
    if (!isdigit(i))
      return false;
  return true;
}

/**
 * Function used to get the columns which has non numeric dataset.
 *
 * @param tempDimesnion A vector of string passed which has column number or
 *    column ranges.
 */
static unordered_set<size_t> GetColumnIndices(const
                                              vector<string>& tempDimension)
{
  unordered_set<size_t> dimensions;
  int columnstartindex, columnendindex;
  size_t found;
  for (size_t i = 0; i < tempDimension.size(); i++)
  {
    found = tempDimension[i].find('-');
    if ( found != string::npos)
    {
      try
      {
        // Has a range include, something of type a-b.
        string subStringStart = tempDimension[i].substr(0, found);
        string subStringEnd = tempDimension[i].substr(found + 1,
            tempDimension[i].length());
        if (!IsNumber(subStringStart) || !IsNumber(subStringEnd))
        {
          Log::Fatal << "Dimension value accepts only digits, characters"
              " encountered, please recheck" << endl;
        }
        columnstartindex = stoi(subStringStart);
        columnendindex = stoi(subStringEnd);
        for (int i = columnstartindex; i <= columnendindex; i++)
          dimensions.insert(i);
      }
      catch (const exception& e)
      {
        Log::Fatal << "Dimension value not clear, either negatve or can't "
        "parse the range. Usage a-b" << endl;
      }
    }
    else
    {
      try
      {
        if (!IsNumber(tempDimension[i]))
        {
          Log::Fatal << "Dimension value accepts only digits, characters"
              " encountered, please recheck" << endl;
        }
        dimensions.insert(stoi(tempDimension[i]));
      }
      catch (const exception& e)
      {
        Log::Fatal << "Dimension value not appropriate " << endl;
      }
    }
  }
  return dimensions;
}

/**
 * Function used to write back the preproccessed data to a file.
 *
 * @param outputFilename Name of the file to save the data into.
 * @param dataset The actual dataset which was read from file.
 * @param nonNumericInput The preproccessed data
 * @param columnDelimiter Delimiter used to separate columns of the file.
 * @param dimesnions Dimesnion which we non numeric or of type string.
 */
static void WriteOutput(const string& outputFilename,
                        const vector<vector<string>>& dataset,
                        const string& columnDelimiter,
                        const unordered_set<size_t>& dimensions,
                        const unordered_map<size_t, arma::mat>& encodedResult)
{
  ofstream fout(outputFilename, ios::trunc);
  if (!fout.is_open())
    Log::Fatal << "Unable to open a file for writing output" << endl;
  for (size_t i = 0 ; i < dataset.size(); i++)
  {
    for (size_t j = 0 ; j < dataset[i].size(); j++)
    {
      if (dimensions.find(j) != dimensions.end())
      {
        if (j + 1 < dataset[i].size())
        {
          for (size_t k = 0; k < encodedResult.at(j).n_cols; k++)
          {
            fout << encodedResult.at(j)(i, k) << columnDelimiter;
          }
        }
        else
        {
          for (size_t k = 0; k < encodedResult.at(j).n_cols; k++)
          {
            if ( k < encodedResult.at(j).n_cols - 1)
              fout << encodedResult.at(j)(i, k) << columnDelimiter;
            else
              fout << encodedResult.at(j)(i, k);
          }
        }
      }
      else
      {
        if (j + 1 < dataset[i].size())
          fout << dataset[i][j] << columnDelimiter;
        else
          fout << dataset[i][j];
      }
    }
    fout << "\n";
  }
}

static void mlpackMain()
{
  // Parse command line options.
  // Extracting the filename
  const string filename = CLI::GetParam<string>("actual_dataset");
  // This is very dangerous, Let's add a check tommorrow.
  string columnDelimiter;
  if (CLI::HasParam("column_delimiter"))
  {
    columnDelimiter = CLI::GetParam<string>("column_delimiter");
    // Allow only 3 delimiters.
    RequireParamValue<string>("column_delimiter", [](const string del)
        { return del == "\t" || del == "," || del == " "; }, true,
        "Delimiter should be either \\t (tab) or , (comma) or ' ' (space) ");
  }
  else
  {
      if (data::Extension(filename) == "csv")
      {
        columnDelimiter = ",";
        Log::Warn << "Found csv Extension, taking , as "
            "columnDelimiter." << endl;
      }
      else if (data::Extension(filename) == "tsv" ||
          data::Extension(filename) == "txt")
      {
        columnDelimiter = "\t";
        Log::Warn << "Found tsv or txt Extension, taking \\t as "
        "columnDelimiter." << endl;
      }
      else
      {
        columnDelimiter = "\t";
        Log::Warn << "columnDelimiter not specified, taking default"
            " value" << endl;
      }
  }
  // Handling Dimension vector
  vector<string> tempDimension =
      CLI::GetParam<vector<string> >("dimension");
  unordered_set<size_t> dimensions = GetColumnIndices(tempDimension);
  vector<vector<string>> dataset = CreateDataset(filename, columnDelimiter[0]);
  for (auto colIndex : dimensions)
  {
    if (colIndex >= dataset.back().size())
      Log::Fatal << "The index given is out of range, please verify" << endl;
  }
  // Preparing the input dataset on which string manipulation has to be done.
  // vector<vector<string>> nonNumericInput(dimension.size());
  unordered_map<size_t , vector<string>> nonNumericInput;
  for (size_t i = 0; i < dataset.size(); i++)
  {
    for (auto& datasetCol : dimensions)
    {
      nonNumericInput[datasetCol].push_back(move(dataset[i][datasetCol]));
    }
  }
  unordered_map<size_t , arma::mat> encodedResult;
  const string delimiter = CLI::GetParam<string> ("delimiter");
  data::SplitByAnyOf tokenizer(delimiter);

  // Convert NonNumeric to their respective encodings.
  const string encodingType = CLI::GetParam<string>("encoding_type");
  arma::mat output;
  size_t encodedColumnCount = 0;
  for (auto& column : nonNumericInput)
  {

    if (encodingType == "DictionaryEncoding")
    {
        // dictionary Encoding.
      data::DictionaryEncoding<data::SplitByAnyOf::TokenType> encoder;
      encoder.Encode(column.second, output, tokenizer);
      encodedResult[column.first] = std::move(output);
      // Calculate the no of features after encoded
      encodedColumnCount += output.n_cols;
    }
    else if(encodingType == "BagOfWordsEncoding")
    {
      // BagofWords Encoding.
      data::BagOfWordsEncoding<data::SplitByAnyOf::TokenType> encoder;

      encoder.Encode(column.second, output, tokenizer);
      encodedResult[column.first] = std::move(output);
      // Calculate the no of features after encoded
      encodedColumnCount += output.n_cols;
    }
    else
    {
      // Tfidf Encoding.
      const bool smoothIdf = CLI::GetParam<bool>("smooth_idf");
      const string tfidfEncodingType = CLI::GetParam<string>("tfidf_encoding_type");
      if ("RawCount" == tfidfEncodingType)
      {
          data::TfIdfEncoding<data::SplitByAnyOf::TokenType>
            encoder(data::TfIdfEncodingPolicy::TfTypes::RAW_COUNT, smoothIdf);
          encoder.Encode(column.second, output, tokenizer);
          encodedResult[column.first] = std::move(output);
          // Calculate the no of features after encoded
          encodedColumnCount += output.n_cols;
      }
      else if("Binary" == tfidfEncodingType)
      {
        data::TfIdfEncoding<data::SplitByAnyOf::TokenType> 
          encoder(data::TfIdfEncodingPolicy::TfTypes::BINARY, smoothIdf);

        encoder.Encode(column.second, output, tokenizer);
        encodedResult[column.first] = std::move(output);
        // Calculate the no of features after encoded
        encodedColumnCount += output.n_cols;
      }
      else if("SublinearTf" == tfidfEncodingType)
      {
        data::TfIdfEncoding<data::SplitByAnyOf::TokenType>
          encoder(data::TfIdfEncodingPolicy::TfTypes::SUBLINEAR_TF, smoothIdf);
        encoder.Encode(column.second, output, tokenizer);
        encodedResult[column.first] = std::move(output);
        // Calculate the no of features after encoded
        encodedColumnCount += output.n_cols;
      }
      else 
      {
        data::TfIdfEncoding<data::SplitByAnyOf::TokenType>
          encoder(data::TfIdfEncodingPolicy::TfTypes::TERM_FREQUENCY, smoothIdf);
        encoder.Encode(column.second, output, tokenizer);
        encodedResult[column.first] = std::move(output);
        // Calculate the no of features after encoded
        encodedColumnCount += output.n_cols;
      }
    }
  }
  const string outputFilename = CLI::GetParam<string>("preprocess"
      "_dataset");
  WriteOutput(outputFilename, dataset, columnDelimiter,
       dimensions, encodedResult);
}
