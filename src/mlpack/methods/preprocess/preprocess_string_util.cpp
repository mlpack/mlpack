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
 * Function to check whether the given colum has only digits.
 *
 * @param column The column no
 */
bool IsNumber(const string& column)
{
  for (auto i : column)
  {
    if (!isdigit(i))
      return false;
  }
  return true;
}

/**
 * Function used to get the columns which has non numeric dataset.
 *
 * @param tempDimesnion A vector of string passed which has column number or
 *    column ranges.
 */
unordered_set<size_t> GetColumnIndices(const vector<string>& tempDimension)
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
