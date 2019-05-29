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

PARAM_INT_IN("lowercase", "convert to lowercase(1 to convert).", "l", 1);
PARAM_INT_IN("punctuation", "Remove punctuation (1 to remove).", "p", 1);
PARAM_INT_IN("stopwords", "Remove stopwords (1 to remove).", "s", 1);
PARAM_INT_IN("dimension", "Column which contains the string data.(1 by "
    "default)", "c", 1);

using namespace mlpack;
using namespace mlpack::util;
using namespace arma;
using namespace std;

static void mlpackMain()
{
  // Parse command line options.
  // Check on label parameters.
  // Check for punctuation.
  RequireParamValue<int>("punctuation",
      [](double x) { return x == 0 || x <= 1; }, true,
      "value for punctuation may be either 0 to not\
       remove punctuation or 1 to remove punctuation 1");
  RequireParamValue<int>("lowercase",
      [](double x) { return x == 0 || x <= 1; }, true,
      "value for lowercase may be either 0 to not\
       remove punctuation or 1 to remove punctuation 1");
  RequireParamValue<int>("stopwords",
      [](double x) { return x == 0 || x <= 1; }, true,
      "value for stopwords may be either 0 to not\
       remove punctuation or 1 to remove punctuation 1");
  // If puntucation is not set, warn the user.
  if (!CLI::HasParam("punctuation"))
  {
    Log::Warn << "You did not specify " << PRINT_PARAM_STRING("punctuation")
        << ", so it will be automatically set to 1." << endl;
  }
  // If lowercase is not set, warn the user.
  if (!CLI::HasParam("lowercase"))
  {
    Log::Warn << "You did not specify " << PRINT_PARAM_STRING("lowercase")
        << ", so it will be automatically set to 1." << endl;
  }
  // If stopwords is not set, warn the user.
  if (!CLI::HasParam("stopwords"))
  {
    Log::Warn << "You did not specify " << PRINT_PARAM_STRING("stopwords")
        << ", so it will be automatically set to 1." << endl;
  }
  // If dimension is not set, warn the user.
  if (!CLI::HasParam("dimension"))
  {
    Log::Warn << "You did not specify " << PRINT_PARAM_STRING("dimension")
        << ", so it will be automatically set to 1." << endl;
  }
  // Ectracting the filename
  const std::string filename = CLI::GetParam<std::string>("actual_dataset");
  // Extracting the Contents of file
  // File pointer
  fstream fin;
  // Open an existing file
  fin.open(filename, ios::in);
  std::string line;
  std::vector<std::string> temp;
  std::vector<std::string> input;
  std::vector<std::vector<std::string>> dataset;
  const size_t Column = CLI::GetParam<int>("dimension");
  while (std::getline(fin, line))
  {
    boost::split(temp, line, boost::is_any_of(std::string(1, ',')));
    input.push_back(temp[Column]);
    dataset.push_back(temp);
    temp.clear();
  }
  fin.close();
  if (CLI::GetParam<int>("lowercase"))
  {
    data::LowerCase(input);
  }
  if (CLI::GetParam<int>("stopwords"))
  {
    // Not sure how to take input for tokenizer from cli.
    // data::RemoveStopWords(input);
  }
  if (CLI::GetParam<int>("punctuation"))
  {
    data::RemovePunctuation(input);
  }
  const std::string filename2 = CLI::GetParam<std::string>("preprocess"
      "_dataset");
  fstream fout;
  fout.open(filename2, ios::out | ios::trunc);
  for (size_t i = 0 ; i < dataset.size(); i++)
  {
    for (size_t j =0 ; j < dataset[i].size(); j++)
    {
      if (j == Column)
      {
        fout<<input[i]<<",";
      }
      else
      {
        fout<<dataset[i][j]<<",";
      }
    }
    fout<<"\n";
  }
}
