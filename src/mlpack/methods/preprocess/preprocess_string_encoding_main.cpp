/**
 * @file preprocess_string_encoding_main.cpp
 * @author Jeffin Sam
 *
 * A IO executable to encode string dataset.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include "mlpack/methods/preprocess/preprocess_string_util.hpp"
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/io.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/core/data/string_encoding.hpp>
#include <mlpack/core/data/tokenizers/split_by_any_of.hpp>
#include <mlpack/core/data/string_encoding_policies/dictionary_encoding_policy.hpp>
#include <mlpack/core/data/string_encoding_policies/bag_of_words_encoding_policy.hpp>
#include <mlpack/core/data/string_encoding_policies/tf_idf_encoding_policy.hpp>


// Program Name.
BINDING_NAME("preprocess_string_encoding");

// Short description.
BINDING_SHORT_DESC(
    "A utility to encode string data. This utility can encode string using "
    "DictionaryEncoding, BagOfWordsEncoding and TfIdfEncoding methods.");

// Long description.
BINDING_LONG_DESC(
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
    PRINT_PARAM_STRING("dimension") + ".");

// Example.
BINDING_EXAMPLE(
    "So, a simple example where we want to encode string dataset " +
    PRINT_DATASET("X") + ", which is having string data in its 3 Column,"
    " using DictionaryEncoding as encoding type."
    "\n\n" +
    PRINT_CALL("preprocess_string", "actual_dataset", "X",
    "preprocess_dataset", "X", "dimension", 3, "encoding_type",
    "DictionaryEncoding"));

// See also...
BINDING_SEE_ALSO("@preprocess_binarize", "#preprocess_binarize");
BINDING_SEE_ALSO("@preprocess_describe", "#preprocess_describe");
BINDING_SEE_ALSO("@preprocess_imputer", "#preprocess_imputer");

PARAM_STRING_IN_REQ("actual_dataset", "File containing the reference dataset.",
    "t");
PARAM_STRING_IN_REQ("preprocess_dataset", "File containing the preprocess or"
    "encoded dataset.", "o");
PARAM_STRING_IN("column_delimiter", "delimeter used to seperate Column in files"
    "example '\\t' for '.tsv' and ',' for '.csv'.", "d", "\t");
PARAM_STRING_IN("delimiter", "A set of chars that is used as delimeter to"
    "tokenize the string dataset ", "D", " ");
PARAM_STRING_IN("encoding_type", "Type of encoding", "e", "DictionaryEncoding")
PARAM_STRING_IN("tfidf_encoding_type", "Type of tfidf encoding", "E",
    "RawCount")
PARAM_FLAG("smooth_idf", "True to have smooth_idf for Tf-Idf.", "s");
PARAM_VECTOR_IN_REQ(std::string, "dimension", "Column which contains the"
    "string data. (1 by default)", "c");

using namespace mlpack;
using namespace mlpack::util;
using namespace arma;
using namespace std;

/**
 * Function used to write back the preproccessed data to a file.
 *
 * @param outputFilename Name of the file to save the data into.
 * @param dataset The actual dataset which was read from file.
 * @param columnDelimiter Delimiter used to separate columns of the file.
 * @param dimensions Dimesnion which has non numeric or string data.
 * @param encodedResult Collection of arma matrices having encoded results.
 */
static void WriteOutput(const string& outputFilename,
                        const vector<vector<string>>& dataset,
                        const string& columnDelimiter,
                        const vector<size_t>& dimensions,
                        const unordered_map<size_t, arma::mat>& encodedResult)
{
  ofstream fout(outputFilename, ios::trunc);
  if (!fout.is_open())
    Log::Fatal << "Unable to open a file for writing output" << endl;
  for (size_t i = 0 ; i < dataset.size(); i++)
  {
    for (size_t j = 0 ; j < dataset[i].size(); j++)
    {
      if (std::find (dimensions.begin(), dimensions.end(), j)  != dimensions.end())
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
  const string filename = IO::GetParam<string>("actual_dataset");
  string columnDelimiter;
  if (IO::HasParam("column_delimiter"))
  {
    columnDelimiter = IO::GetParam<string>("column_delimiter");
    // Allow only 3 delimiters.
    RequireParamValue<string>("column_delimiter", [](const string del)
        { return del == "\t" || del == "," || del == " "; }, true,
        "Delimiter should be either \\t (tab) or , (comma) or ' ' (space) ");
  }
  else
    columnDelimiter = data::ColumnDelimiterType(filename);

  // Parsing the given dimensions.
  vector<string> dimensionsParam = IO::GetParam<vector<string>>("dimension");
  const vector<size_t> dimensions = data::GetColumnIndices(dimensionsParam);
  vector<vector<string>> dataset = data::CreateDataset(filename,
      columnDelimiter[0]);

  for (auto colIndex : dimensions)
  {
    if (colIndex >= dataset.back().size())
    {
      Log::Fatal << "The given index (" << colIndex << ") is out of range!"
                 << endl;
    }
  }

  // Preparing the input dataset on which string manipulation has to be done.
  // vector<vector<string>> nonNumericInput(dimension.size());
  unordered_map<size_t , vector<string>> textInput;

  for (vector<string>& row : dataset)
  {
    for (size_t column : dimensions)
      textInput[column].push_back(move(row[column]));
  }

  unordered_map<size_t , arma::mat> encodedResult;
  const string delimiters = IO::GetParam<string> ("delimiter");
  data::SplitByAnyOf tokenizer(delimiters);


  if (IO::HasParam("encoding_type"))
  {
    RequireParamValue<string>("encoding_type", [](const string et)
        { return et == "DictionaryEncoding" || et == "BagOfWordsEncoding"
        || et == "TfIdfEncoding"; }, true, "Encoding Type should be either"
        " BagOfWordsEncoding or DictionaryEncoding or TfIdfEncoding ");
  }
  const string& encodingType = IO::GetParam<string>("encoding_type");
  arma::mat output;
  using TokenType = data::SplitByAnyOf::TokenType;
  for (auto& column : textInput)
  {
    if (encodingType == "DictionaryEncoding")
    {
        // dictionary Encoding.
      data::DictionaryEncoding<TokenType> encoder;
      encoder.Encode(column.second, output, tokenizer);
      encodedResult[column.first] = std::move(output.t());
    }
    else if (encodingType == "BagOfWordsEncoding")
    {
      // BagofWords Encoding.
      data::BagOfWordsEncoding<TokenType> encoder;

      encoder.Encode(column.second, output, tokenizer);
      encodedResult[column.first] = std::move(output.t());
    }
    else
    {
      // Tfidf Encoding.
      const bool smoothIdf = IO::GetParam<bool>("smooth_idf");
      if (IO::HasParam("tfidf_encoding_type"))
      {
        RequireParamValue<string>("tfidf_encoding_type", [](const string et)
            { return et == "RawCount" || et == "Binary" || et == "SublinearTf"
            || et == "TermFrequency"; }, true, "Tf Idf encoding type should be "
            " either RawCount, Binary, SublinearTf or TermFrequency ");
      }
      const string tfidfEncodingType =
          IO::GetParam<string>("tfidf_encoding_type");
      using TfTypes = data::TfIdfEncodingPolicy::TfTypes;
      if ("RawCount" == tfidfEncodingType)
      {
          data::TfIdfEncoding<TokenType>
            encoder(TfTypes::RAW_COUNT, !smoothIdf);
          encoder.Encode(column.second, output, tokenizer);
          encodedResult[column.first] = std::move(output.t());
      }
      else if ("Binary" == tfidfEncodingType)
      {
        data::TfIdfEncoding<TokenType>
          encoder(TfTypes::BINARY, !smoothIdf);
        encoder.Encode(column.second, output, tokenizer);
        encodedResult[column.first] = std::move(output.t());
      }
      else if ("SublinearTf" == tfidfEncodingType)
      {
        data::TfIdfEncoding<TokenType>
          encoder(TfTypes::SUBLINEAR_TF, !smoothIdf);
        encoder.Encode(column.second, output, tokenizer);
        encodedResult[column.first] = std::move(output.t());
      }
      else
      {
        data::TfIdfEncoding<TokenType>
          encoder(TfTypes::TERM_FREQUENCY,
          !smoothIdf);
        encoder.Encode(column.second, output, tokenizer);
        encodedResult[column.first] = std::move(output.t());
      }
    }
  }
  const string outputFilename = IO::GetParam<string>("preprocess"
      "_dataset");
  WriteOutput(outputFilename, dataset, columnDelimiter,
       dimensions, encodedResult);
}
