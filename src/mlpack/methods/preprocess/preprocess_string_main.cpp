/**
 * @file preprocess_string_main.cpp
 * @author Jeffin Sam
 *
 * A CLI executable to preprocess string dataset.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "mlpack/methods/preprocess/preprocess_string_util.hpp"
#include <mlpack/core/util/mlpack_main.hpp>

PROGRAM_INFO("preprocess_string",
    // Short description.
    "A utility to preprocess string data. This utility can remove stopwords, "
    "punctuation and convert to lowercase and uppercase.",
    // Long description.
    "This utility takes a dataset and the dimension and arguments and "
    "does the preprocessing of string dataset according to arguments given."
    "\n\n"
    "The dataset may be given as the file name and the output may be saved as "
    + PRINT_PARAM_STRING("actual_dataset") + " and " +
    PRINT_PARAM_STRING("preprocess_dataset") + " ."
    "\n\n"
    " Following arguments may be given " + PRINT_PARAM_STRING("lowercase") +
    " to convert the dataset to lower case and " +
    PRINT_PARAM_STRING("punctuation") + "for removing punctuation and "
    + PRINT_PARAM_STRING("stopwords") + "for removing stopwords. Also the "
    "dimension which contains the string dataset " +
    PRINT_PARAM_STRING("dimension") + "."
    "\n\n"
    "So, a simple example where we want to preprocess string the dataset " +
    PRINT_DATASET("X") + ", which is having string data in its 3 Column."
    "\n\n" +
    PRINT_CALL("preprocess_string", "actual_dataset", "X",
        "preprocess_dataset", "X", "lowercase", 1, "stopwords",
        1, "punctuation", 1, "dimension", 3) +
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
PARAM_STRING_IN("stopwordsfile", "File containing stopwords", "S", "");

PARAM_FLAG("lowercase", "convert to lowercase.", "l");
PARAM_FLAG("punctuation", "Remove punctuation.", "p");
PARAM_FLAG("stopwords", "Remove stopwords.", "s");
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
 * @param nonNumericInput The preproccessed data
 * @param columnDelimiter Delimiter used to separate columns of the file.
 * @param dimesnions Dimesnion which we non numeric or of type string.
 */
static void WriteOutput(const string& outputFilename,
                        const vector<vector<string>>& dataset,
                        const unordered_map<size_t, vector<string>>&
                            nonNumericInput,
                        const string& columnDelimiter,
                        const unordered_set<size_t>& dimensions)
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
          fout << nonNumericInput.at(j)[i] << columnDelimiter;
        else
          fout << nonNumericInput.at(j)[i];
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
    columnDelimiter = data::ColumnDelimiterType(filename);
  // Handling Dimension vector
  vector<string> tempDimension =
      CLI::GetParam<vector<string> >("dimension");
  unordered_set<size_t> dimensions = data::GetColumnIndices(tempDimension);
  vector<vector<string>> dataset = data::CreateDataset(filename,
      columnDelimiter[0]);
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
  data::StringCleaning obj;
  if (CLI::HasParam("lowercase"))
  {
    for (auto& column : nonNumericInput)
      obj.LowerCase(column.second);
  }
  if (CLI::HasParam("punctuation"))
  {
    for (auto& column : nonNumericInput)
      obj.RemovePunctuation(column.second);
  }
  if (CLI::HasParam("stopwords"))
  {
    // Not sure how to take input for tokenizer from cli.
    if (!CLI::HasParam("stopwordsfile"))
      Log::Fatal << "Please provide a file for stopwords." << endl;
    // Open an existing file
    const string stopWordFilename = CLI::GetParam<string>("stopwordsfile");
    ifstream stopWordFile(stopWordFilename);
    if (!stopWordFile.is_open())
    {
      Log::Fatal << "Unable to open the file for stopwords." << endl;
    }
    string word;
    deque<string> originalword;
    unordered_set<boost::string_view,
        boost::hash<boost::string_view> >stopwords;
    while (stopWordFile >> word)
    {
      originalword.push_back(word);
      stopwords.insert(originalword.back());
    }
    const string delimiter = CLI::GetParam<string> ("delimiter");
    for (auto& column : nonNumericInput)
      obj.RemoveStopWords(column.second, stopwords,
          data::SplitByChar(delimiter));
  }
  const string outputFilename = CLI::GetParam<string>("preprocess"
      "_dataset");
  WriteOutput(outputFilename, dataset, nonNumericInput, columnDelimiter,
      dimensions);
}
