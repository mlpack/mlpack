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

static void mlpackMain()
{
  // Parse command line options.
  // Extracting the filename
  const std::string filename = CLI::GetParam<std::string>("actual_dataset");
  // This is very dangerous, Let's add a check tommorrow.
  std::string column_delimiter;
  if (CLI::HasParam("column_delimiter"))
  {
    column_delimiter = CLI::GetParam<std::string>("column_delimiter");
    // Allow only 3 delimiters.
    RequireParamValue<std::string>("column_delimiter", [](std::string del)
        { return del == "\t" || del == "," || del == " "; }, true,
        "Delimiter should be either \\t (tab) or , (comma) or ' ' (space) ");
  }
  else
  {
      if (data::Extension(filename) == "csv")
      {
        column_delimiter = ",";
        Log::Warn << "Found csv Extension, taking , as column_delimiter. \n";
      }
      else if (data::Extension(filename) == "tsv" ||
          data::Extension(filename) == "txt")
      {
        column_delimiter = "\t";
        Log::Warn << "Found tsv or txt Extension, taking \\t as"
        "column_delimiter. \n";
      }
      else
      {
        Log::Warn << "column_delimiter not specified, taking default value \n";
      }
  }
  // Handling Dimension vector
  std::vector<std::string> temp_dimension =
      CLI::GetParam<std::vector<std::string> >("dimension");
  std::set<size_t>dimension;
  int columnstartindex, columnendindex;
  size_t found;
  for (size_t i = 0; i < temp_dimension.size(); i++)
  {
    found = temp_dimension[i].find('-');
    if ( found != string::npos)
    {
      try
      {
        // Has a range include, something of type a-b.
        columnstartindex = std::stoi(temp_dimension[i].substr(0, found));
        columnendindex = std::stoi(temp_dimension[i].substr(found+1,
            temp_dimension[i].length()));
      }
      catch (const std::exception& e)
      {
        Log::Fatal << "Dimension value not clear, either negatve or can't "
        "parse the range. Usage a-b \n";
      }
      for (int i = columnstartindex; i <= columnendindex; i++)
        dimension.insert(i);
    }
    else
    {
      try
      {
        dimension.insert(std::stoi(temp_dimension[i]));
      }
      catch (const std::exception& e)
      {
        Log::Fatal << "Dimension value not appropriate \n";      
      }
    }
  }

  // Extracting the Contents of file
  // File pointer
  ifstream fin;
  // Open an existing file
  fin.open(filename);
  if (!fin.is_open())
  {
    Log::Fatal << "Unable to open input file \n";
  }
  std::string line, word;
  std::vector<std::vector<std::string>> dataset;
  boost::string_view token, copy;
  while (std::getline(fin, line))
  {
    stringstream streamLine(line);
    dataset.emplace_back(std::vector<std::string>());
    // delimeter[0] becase the standard function accepts char as input.
    while (std::getline(streamLine, word, column_delimiter[0]))
    {
      dataset.back().push_back(word);
    }
  }
  std::set<size_t>::iterator dimensionIterator = dimension.begin();
  // Preparing the input dataset on which string manipulation has to be done.
  std::vector<std::vector<std::string>> input(dimension.size());
  for (size_t i = 0; i < dataset.size(); i++)
  {
    size_t col = 0;
    dimensionIterator = dimension.begin();
    for (size_t j = 0; j < dataset[i].size(); j++)
    {
      if (col < dimension.size())
      {
        if (*dimensionIterator == j)
        {
          // insert into input
          input[col].push_back(dataset[i][j]);
          col++;
          dimensionIterator++;
        }
      }
      else
      {
        break;
      }
    }
  }
  data::StringCleaning obj;
  if (CLI::HasParam("lowercase"))
  {
    for (auto& column_line : input)
      obj.LowerCase(column_line);
  }
  if (CLI::HasParam("stopwords"))
  {
    // Not sure how to take input for tokenizer from cli.
    if (!CLI::HasParam("stopwordsfile"))
    {
      Log::Fatal << "Please provide a file for stopwords.\n";
    }
    ifstream stopwordfile;
    // Open an existing file
    const std::string stopwordfilename =
        CLI::GetParam<std::string>("stopwordsfile");
    stopwordfile.open(stopwordfilename);
    if (!stopwordfile.is_open())
    {
      Log::Fatal << "Unable to open the file for stopwords.\n";
    }
    std::string word;
    std::deque<std::string> originalword;
    std::unordered_set<boost::string_view,
        boost::hash<boost::string_view> >stopwords;
    while (stopwordfile >> word)
    {
      originalword.push_back(word);
      stopwords.insert(originalword.back());
    }
    const std::string delimiter = CLI::GetParam<std::string>
        ("delimiter");
    for (auto& column_line : input)
      obj.RemoveStopWords(column_line, stopwords,
          data::SplitByChar(delimiter));
  }
  if (CLI::HasParam("punctuation"))
  {
    for (auto& column_line : input)
      obj.RemovePunctuation(column_line);
  }
  const std::string outputFilename = CLI::GetParam<std::string>("preprocess"
      "_dataset");
  ofstream fout;
  fout.open(outputFilename, ios::trunc);
  if (!fout.is_open())
  {
    Log::Fatal << "Unable to open a file for writing output.\n";
  }
  dimensionIterator = dimension.begin();
  for (size_t i = 0 ; i < dataset.size(); i++)
  {
    size_t col = 0;
    dimensionIterator = dimension.begin();
    for (size_t j = 0 ; j < dataset[i].size(); j++)
    {
      if (col < dimension.size() && *dimensionIterator == j)
      {
        if (j + 1 < dataset[i].size())
          fout<<input[col][i]<<column_delimiter;
        else
          fout<<input[col][i];
        dimensionIterator++;
        col++;
      }
      else
      {
        if (j + 1 < dataset[i].size())
          fout<<dataset[i][j]<<column_delimiter;
        else
          fout<<dataset[i][j];
      }
    }
    fout<<"\n";
  }
}
