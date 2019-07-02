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
PARAM_STRING_IN_REQ("delimeter", "delimeter used to seperate Column in files"
    "example '\\t' for '.tsv' and ',' for '.csv'.", "d");
PARAM_STRING_IN("strDelimeter", "A set of chars that is used as delimeter to"
    "tokenize the string dataset ", "D", " ");
PARAM_STRING_IN("stopwordsfile", "File containing stopwords", "S", "");

PARAM_FLAG("lowercase", "convert to lowercase.", "l");
PARAM_FLAG("punctuation", "Remove punctuation.", "p");
PARAM_FLAG("stopwords", "Remove stopwords.", "s");
PARAM_VECTOR_IN_REQ(int, "dimension", "Column which contains the string data."
    "(1 by default)", "c");

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
  const std::string delimiter = CLI::GetParam<std::string>("delimeter");
  std::vector<int> dimension = CLI::GetParam<std::vector<int>>("dimension");
  // Sorting neccessary
  std::sort(dimension.begin(), dimension.end());
  // Extracting the Contents of file
  // File pointer
  fstream fin;
  // Open an existing file
  fin.open(filename, ios::in);
  std::string line;
  std::vector<std::string> temp;
  std::vector<std::vector<std::string>> dataset;
  // const size_t Column = CLI::GetParam<int>("dimension");
  data::SplitByChar sobj(delimiter);
  boost::string_view token, copy;
  while (std::getline(fin, line))
  {
    copy = line;
    token = sobj(copy);
    while (!token.empty())
    {
      temp.push_back(std::string(token));
      token = sobj(copy);
    }
    dataset.push_back(temp);
    temp.clear();
  }
  fin.close();
  // Preparing the input dataset on which string manipulation has to be done.
  size_t row = 0;
  std::vector<std::vector<std::string>> input(dimension.size());
  for (size_t i = 0; i < dataset.size(); i++)
  {
    row = 0;
    for (size_t j = 0; j < dataset[i].size(); j++)
    {
      if(row < dimension.size())
      {
        if (dimension[row] == j)
        {
          // insert into input
          input[row].push_back(dataset[i][j]);
          row++;          
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
    for (size_t i = 0; i < input.size(); i++)
      obj.LowerCase(input[i]);
  }
  if (CLI::HasParam("stopwords"))
  {
    // Not sure how to take input for tokenizer from cli.
    if (!CLI::HasParam("stopwordsfile"))
    {
      throw std::runtime_error("Please provide a file for stopwords.");
    }
    fstream fin;
    // Open an existing file
    const std::string stopwordfilename = CLI::GetParam<std::string>
        ("stopwordsfile");
    fin.open(stopwordfilename, ios::in);
    std::string word;
    std::deque<std::string> originalword;
    std::unordered_set<boost::string_view,
        boost::hash<boost::string_view>>stopwords;
    while (fin>>word)
    {
      originalword.push_back(word);
      stopwords.insert(originalword.back());
    }
    fin.close();
    const std::string strDelimiter = CLI::GetParam<std::string>
        ("strDelimeter");
    for (size_t i = 0; i < input.size(); i++)
      obj.RemoveStopWords(input[i], stopwords,
          data::SplitByChar(strDelimiter));
  }
  if (CLI::HasParam("punctuation"))
  {
    for (size_t i = 0; i < input.size(); i++)
      obj.RemovePunctuation(input[i]);
  }
  const std::string filename2 = CLI::GetParam<std::string>("preprocess"
      "_dataset");
  fstream fout;
  fout.open(filename2, ios::out | ios::trunc);
  row = 0;
  // const std::string deli = obj.Delimiter();
  for (size_t i = 0 ; i < dataset.size(); i++)
  {
    row = 0;
    for (size_t j =0 ; j < dataset[i].size(); j++)
    {
      if (row < dimension.size() && dimension[row] == j)
      {
        fout<<input[row][i]<<sobj.Delimiter();
      }
      else
      {
        fout<<dataset[i][j]<<sobj.Delimiter();
      }
    }
    fout<<"\n";
  }
}
