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
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/core/math/random.hpp>
#include <mlpack/core/util/cli.hpp>
#include <mlpack/core/data/string_cleaning.hpp>
#include <mlpack/core/data/extension.hpp>
#include <mlpack/core/data/tokenizer/split_by_char.hpp>

PROGRAM_INFO("preprocess_string",
    // Short description.
    "A utility to preprocess string data. This utility can remove stopwords, "
    "punctuation and convert to lowercase.",
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
PARAM_STRING_IN("columnDelimiter", "delimeter used to seperate Column in files"
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

static vector<vector<string>> CreateDataset(const string& filename,
                                            char& columnDelimiter)
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
    // delimeter[0] becase the standard function accepts char as input.
    while (getline(streamLine, word, columnDelimiter))
    {
      dataset.back().push_back(word);
    }
  }
  return dataset;
}

static bool IsNumber(const string& column)
{
  for (auto& i : column)
    if (!isdigit(i))
      return false;
  return true;
}

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
        string subStringEnd = tempDimension[i].substr(found+1,
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
  // This is very dangerous, Let's add a check tommorrow.
  string columnDelimiter;
  if (CLI::HasParam("columnDelimiter"))
  {
    columnDelimiter = CLI::GetParam<string>("columnDelimiter");
    // Allow only 3 delimiters.
    RequireParamValue<string>("columnDelimiter", [](const string del)
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
    if (colIndex >= dataset.back().size())
      Log::Fatal << "The index given is out of range, please verify" << endl;
  // Preparing the input dataset on which string manipulation has to be done.
  // vector<vector<string>> nonNumericInput(dimension.size());
  unordered_map<size_t , vector<string>> nonNumericInput;
  for (size_t i = 0; i < dataset.size(); i++)
  {
    for (auto& datasetCol : dimensions)
      nonNumericInput[datasetCol].push_back(move(dataset[i][datasetCol]));
  }
  data::StringCleaning obj;
  if (CLI::HasParam("lowercase"))
  {
    for (auto& datasetCol : dimensions)
      obj.LowerCase(nonNumericInput[datasetCol]);
  }
  if (CLI::HasParam("stopwords"))
  {
    // Not sure how to take input for tokenizer from cli.
    if (!CLI::HasParam("stopwordsfile"))
      Log::Fatal << "Please provide a file for stopwords." << endl;
    // Open an existing file
    const string stopWordFilename =
        CLI::GetParam<string>("stopwordsfile");
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
    const string delimiter = CLI::GetParam<string>
        ("delimiter");
    for (auto& datasetCol : dimensions)
      obj.RemoveStopWords(nonNumericInput[datasetCol], stopwords,
          data::SplitByChar(delimiter));
  }
  if (CLI::HasParam("punctuation"))
  {
    for (auto& datasetCol : dimensions)
      obj.RemovePunctuation(nonNumericInput[datasetCol]);
  }
  const string outputFilename = CLI::GetParam<string>("preprocess"
      "_dataset");
  WriteOutput(outputFilename, dataset, nonNumericInput, columnDelimiter,
      dimensions);
}
