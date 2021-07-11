/**
 * @file preprocess_string_util.cpp
 * @author Jeffin Sam
 *
 * A CLI executable to encode string dataset.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include "mlpack/methods/preprocess/preprocess_string_util.hpp"

using namespace mlpack;
using namespace mlpack::util;
using namespace arma;
using namespace std;

namespace mlpack {
namespace data {

/**
 * Function neccessary to create a vector<vector<string>> by readin
 * the contents of a file.
 *
 * @param filename Name of the file whose contents need to be preproccessed.
 * @param columnDelimiter Delimiter used to split the columns of file.
 */
vector<vector<string>> CreateDataset(const string& filename,
                                     char columnDelimiter)
{
  vector<vector<string>> dataset;
  // Extracting the contents of file.
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

/**
 * Function to check whether the given column contains only digits or not.
 *
 * @param column Column index to check.
 */
bool IsNumber(const string& column)
{
  if (column.empty())
    return false;

  for (auto i : column)
  {
    if (!isdigit(i))
      return false;
  }

  return true;
}

/**
 * The function parses the given column indices and ranges.
 *
 * @param dimensions A vector of column indices or column ranges.
 */
vector<size_t> GetColumnIndices(const vector<string>& dimensions)
{
  vector<size_t> result;

  for (const string& elem : dimensions)
  {
    const size_t found = elem.find('-');

    if (found != string::npos)
    {
      // Trying to parse an indices range.
      string firstSubstring = elem.substr(0, found);
      string secondSubstring = elem.substr(found + 1);

      if (!IsNumber(firstSubstring) || !IsNumber(secondSubstring))
      {
        Log::Fatal << "Can't parse column indices range '" << elem << "'!"
                   << endl;
      }

      const size_t lowerBound = stoi(firstSubstring);
      const size_t upperBound = stoi(secondSubstring);

      for (size_t i = lowerBound; i <= upperBound; i++)
        result.push_back(i);
    }
    else
    {
      // Trying to parse a column index.

      if (!IsNumber(elem))
        Log::Fatal << "Can't parse column index '" << elem << "'!" << endl;

      result.push_back(stoi(elem));
    }
  }

  result.erase(unique(result.begin(), result.end()), result.end());

  return result;
}

/**
 * Function to get the type of column delimiter base on file extension.
 *
 * @param filename Name of the input file.
 */
string ColumnDelimiterType(const string& filename)
{
  string columnDelimiter;
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
  return columnDelimiter;
}

} // namespace data
} // namespace mlpack
