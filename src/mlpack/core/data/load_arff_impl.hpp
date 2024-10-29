/**
 * @file core/data/load_arff_impl.hpp
 * @author Ryan Curtin
 *
 * Load an ARFF dataset.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_LOAD_ARFF_IMPL_HPP
#define MLPACK_CORE_DATA_LOAD_ARFF_IMPL_HPP

// In case it hasn't been included yet.
#include "load_arff.hpp"
#include "string_algorithms.hpp"
#include "is_naninf.hpp"

namespace mlpack {
namespace data {

template<typename eT, typename PolicyType>
void LoadARFF(const std::string& filename,
              arma::Mat<eT>& matrix,
              DatasetMapper<PolicyType>& info)
{
  // First, open the file.
  std::ifstream ifs;
  ifs.open(filename, std::ios::in | std::ios::binary);

  // if file is not open throw an error (file not found).
  if (!ifs.is_open())
  {
    Log::Fatal << "Cannot open file '" << filename << "'. " << std::endl;
  }

  std::string line;
  size_t dimensionality = 0;
  // We'll store a vector of strings representing categories to be mapped, if
  // needed.
  std::map<size_t, std::vector<std::string>> categoryStrings;
  std::vector<bool> types;
  size_t headerLines = 0;
  while (ifs.good())
  {
    // Read the next line, then strip whitespace from either side.
    std::getline(ifs, line, '\n');
    Trim(line);
    ++headerLines;

    // Is the first character a comment, or is the line empty?
    if (line[0] == '%' || line.empty())
      continue; // Ignore this line.

    // If the first character is @, we are looking at @relation, @attribute, or
    // @data.
    if (line[0] == '@')
    {
      std::vector<std::string> tok = Tokenize(line, ' ', '"');
      std::vector<std::string>::iterator it = tok.begin();

      // Get the annotation we are looking at.
      std::string annotation(*it);
      std::transform(annotation.begin(), annotation.end(), annotation.begin(),
            ::tolower);

      if (annotation == "@relation")
      {
        // We don't actually have anything to do with the name of the dataset.
        continue;
      }
      else if (annotation == "@attribute")
      {
        ++dimensionality;
        // We need to mark this dimension with its according type.
        ++it; // Ignore the dimension name.
        ++it;
        // Collect all of the remaining tokens, which represent the dimension.
        std::string dimType = "";
        while (it != tok.end())
          dimType += *(it++);
        std::string origDimType(dimType); // We may need the original cases.
        std::transform(dimType.begin(), dimType.end(), dimType.begin(),
            ::tolower);

        if (dimType == "numeric" || dimType == "integer" || dimType == "real")
        {
          types.push_back(false); // The feature is numeric.
        }
        else if (dimType == "string")
        {
          types.push_back(true); // The feature is categorical.
        }
        else if (dimType[0] == '{')
        {
          // The feature is categorical, and we have all the types right here.
          // Note that categories are case-sensitive, and so we must use the
          // `origDimType` string here instead (which has not had ::tolower used
          // on it).
          types.push_back(true);
          TrimIf(origDimType,
              [](char c)
              {
                return c == '{' || c == '}' || c == ' ' || c == '\t';
              });

          std::vector<std::string> dimTok = Tokenize(origDimType, ',', '"');
          std::vector<std::string>::iterator it = dimTok.begin();
          std::vector<std::string> categories;

          while (it != dimTok.end())
          {
            std::string category = (*it);
            Trim(category);
            categories.push_back(category);

            ++it;
          }

          categoryStrings[dimensionality - 1] = std::move(categories);
        }
      }
      else if (annotation == "@data")
      {
        // We are in the data section.  So we can move out of this loop.
        break;
      }
      else
      {
        throw std::runtime_error("unknown ARFF annotation '" + (*tok.begin()) +
            "'");
      }
    }
  }

  if (ifs.eof())
    throw std::runtime_error("no @data section found");

  // Reset the DatasetInfo object, if needed.
  if (info.Dimensionality() == 0)
  {
    info = DatasetMapper<PolicyType>(dimensionality);
  }
  else if (info.Dimensionality() != dimensionality)
  {
    std::ostringstream oss;
    oss << "data::LoadARFF(): given DatasetInfo has dimensionality "
        << info.Dimensionality() << ", but data has dimensionality "
        << dimensionality;
    throw std::invalid_argument(oss.str());
  }

  for (size_t i = 0; i < types.size(); ++i)
  {
    if (types[i])
      info.Type(i) = Datatype::categorical;
    else
      info.Type(i) = Datatype::numeric;
  }

  // Make sure all strings are mapped, if we have any.
  using IteratorType =
      std::map<size_t, std::vector<std::string>>::const_iterator;
  for (IteratorType it = categoryStrings.begin(); it != categoryStrings.end();
      ++it)
  {
    for (const std::string& str : (*it).second)
    {
      info.template MapString<eT>(str, (*it).first);
    }
  }

  // We need to find out how many lines of data are in the file.
  std::streampos pos = ifs.tellg();
  size_t row = 0;
  while (ifs.good())
  {
    std::getline(ifs, line, '\n');
    ++row;
  }
  // Uncount the EOF row.
  --row;

  // Since we've hit the EOF, we have to call clear() so we can seek again.
  ifs.clear();
  ifs.seekg(pos);

  // Now, set the size of the matrix.
  matrix.set_size(dimensionality, row);

  // Now we are looking at the @data section.
  row = 0;
  while (ifs.good())
  {
    std::getline(ifs, line, '\n');
    Trim(line);
    // Each line of the @data section must be a CSV (except sparse data, which
    // we will handle later).  So now we can tokenize the
    // CSV and parse it.  The '?' representing a missing value is not allowed,
    // so if that occurs we throw an exception.  We also throw an exception if
    // any piece of data does not match its type (categorical or numeric).

    // If the first character is {, it is sparse data, and we can just say this
    // is not handled for now...
    if (line[0] == '{')
      throw std::runtime_error("cannot yet parse sparse ARFF data");

    // Tokenize the line.
    std::vector<std::string> tok = Tokenize(line, ',', '"');

    size_t col = 0;
    std::stringstream token;
    for (std::vector<std::string>::iterator it = tok.begin(); it != tok.end();
         ++it)
    {
      // Check that we are not too many columns in.
      if (col >= matrix.n_rows)
      {
        std::stringstream error;
        error << "Too many columns in line " << (headerLines + row) << ".";
        throw std::runtime_error(error.str());
      }

      // What should this token be?
      if (info.Type(col) == Datatype::categorical)
      {
        // Strip spaces before mapping.
        std::string token = *it;
        Trim(token);
        const size_t currentNumMappings = info.NumMappings(col);
        const eT result = info.template MapString<eT>(token, col);

        // If the set of categories was pre-specified, then we must crash if
        // this was not one of those categories.
        if (categoryStrings.count(col) > 0 &&
            currentNumMappings < info.NumMappings(col))
        {
          std::stringstream error;
          error << "Parse error at line " << (headerLines + row) << " token "
              << col << ": category \"" << token << "\" not in the set of known"
              << " categories for this dimension (";
          for (size_t i = 0; i < categoryStrings.at(col).size() - 1; ++i)
            error << "\"" << categoryStrings.at(col)[i] << "\", ";
          error << "\"" << categoryStrings.at(col).back() << "\").";
          throw std::runtime_error(error.str());
        }

        // We load transposed.
        matrix(col, row) = result;
      }
      else if (info.Type(col) == Datatype::numeric)
      {
        // Attempt to read as numeric.
        token.clear();
        token.str(*it);

        eT val = eT(0);
        token >> val;

        if (token.fail())
        {
          // Check for NaN or inf.
          if (!IsNaNInf(val, token.str()))
          {
            // Okay, it's not NaN or inf.  If it's '?', we issue a specific
            // error, otherwise we issue a general error.
            std::stringstream error;
            std::string tokenStr = token.str();
            Trim(tokenStr);
            if (tokenStr == "?")
              error << "Missing values ('?') not supported, ";
            else
              error << "Parse error ";
            error << "at line " << (headerLines + row) << " token " << col
                << ": \"" << tokenStr << "\".";
            throw std::runtime_error(error.str());
          }
        }

        // If we made it to here, we have a value.
        matrix(col, row) = val; // We load transposed.
      }

      ++col;
    }
    ++row;
  }
}

} // namespace data
} // namespace mlpack

#endif
