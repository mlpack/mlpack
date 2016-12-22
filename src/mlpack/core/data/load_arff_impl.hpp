/**
 * @file load_arff_impl.hpp
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

#include <boost/algorithm/string.hpp>

namespace mlpack {
namespace data {

template<typename eT, typename PolicyType>
void LoadARFF(const std::string& filename,
              arma::Mat<eT>& matrix,
              DatasetMapper<PolicyType>& info)
{
  // First, open the file.
  std::ifstream ifs;
  ifs.open(filename);

  std::string line;
  size_t dimensionality = 0;
  std::vector<bool> types;
  size_t headerLines = 0;
  while (!ifs.eof())
  {
    // Read the next line, then strip whitespace from either side.
    std::getline(ifs, line, '\n');
    boost::trim(line);
    ++headerLines;

    // Is the first character a comment, or is the line empty?
    if (line[0] == '%' || line.empty())
      continue; // Ignore this line.

    // If the first character is @, we are looking at @relation, @attribute, or
    // @data.
    if (line[0] == '@')
    {
      typedef boost::tokenizer<boost::escaped_list_separator<char>> Tokenizer;
      std::string separators = " \t\%"; // Split on comments too.
      boost::escaped_list_separator<char> sep("\\", separators, "\"{");
      Tokenizer tok(line, sep);
      Tokenizer::iterator it = tok.begin();

      // Get the annotation we are looking at.
      std::string annotation(*it);

      if (*tok.begin() == "@relation")
      {
        // We don't actually have anything to do with the name of the dataset.
        continue;
      }
      else if (*tok.begin() == "@attribute")
      {
        ++dimensionality;
        // We need to mark this dimension with its according type.
        ++it; // Ignore the dimension name.
        std::string dimType = *(++it);
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
          throw std::logic_error("list of ARFF values not yet supported");
        }
      }
      else if (*tok.begin() == "@data")
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

  // We need to find out how many lines of data are in the file.
  std::streampos pos = ifs.tellg();
  size_t row = 0;
  while (!ifs.eof())
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
  while (!ifs.eof())
  {
    std::getline(ifs, line, '\n');
    boost::trim(line);
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
    typedef boost::tokenizer<boost::escaped_list_separator<char>> Tokenizer;
    boost::escaped_list_separator<char> sep("\\", ",", "\"");
    Tokenizer tok(line, sep);

    size_t col = 0;
    std::stringstream token;
    for (Tokenizer::iterator it = tok.begin(); it != tok.end(); ++it)
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
        matrix(col, row) = info.MapString(*it, col); // We load transposed.
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
          if (!arma::diskio::convert_naninf(val, token.str()))
          {
            // Okay, it's not NaN or inf.  If it's '?', we issue a specific
            // error, otherwise we issue a general error.
            std::stringstream error;
            std::string tokenStr = token.str();
            boost::trim(tokenStr);
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
