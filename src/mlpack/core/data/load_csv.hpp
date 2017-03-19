/**
 * @file load_csv.hpp
 * @author ThamNgapWei
 *
 * This is a csv parsers which use to parse the csv file format
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_LOAD_CSV_HPP
#define MLPACK_CORE_DATA_LOAD_CSV_HPP

#include <boost/spirit/include/qi.hpp>
#include <boost/algorithm/string/trim.hpp>

#include <mlpack/core.hpp>
#include <mlpack/core/util/log.hpp>

#include <set>
#include <string>

#include "extension.hpp"
#include "format.hpp"
#include "dataset_mapper.hpp"

namespace mlpack {
namespace data {

/**
 *Load the csv file.This class use boost::spirit
 *to implement the parser, please refer to following link
 *http://theboostcpplibraries.com/boost.spirit for quick review.
 */
class LoadCSV
{
public:
  explicit LoadCSV(std::string file, bool fatal = false);

  template<typename T, typename PolicyType>
  void Load(arma::Mat<T> &inout, DatasetMapper<PolicyType> &infoSet, bool transpose = true)
  {
    if(!CanOpen())
    {
      return;
    }

    if(transpose)
    {
      TranposeParse(inout, infoSet);
    }
    else
    {
      NonTranposeParse(inout, infoSet);
    }
  }

  size_t ColSize();
  size_t RowSize();

  /**
   * Peek at the file to determine the number of rows and columns in the matrix,
   * assuming a non-transposed matrix.  This will also take a first pass over
   * the data for DatasetMapper, if MapPolicy::NeedsFirstPass is true.  The info
   * object will be re-initialized with the correct dimensionality.
   *
   * @param rows Variable to be filled with the number of rows.
   * @param cols Variable to be filled with the number of columns.
   * @param info DatasetMapper object to use for first pass.
   */
  template<typename T, typename MapPolicy>
  void GetMatrixSize(size_t& rows, size_t& cols, DatasetMapper<MapPolicy>& info)
  {
    // Take a pass through the file.  If the DatasetMapper policy requires it,
    // we will pass everything string through MapString().  This might be useful
    // if, e.g., the MapPolicy needs to find which dimensions are numeric or
    // categorical.

    // Reset to the start of the file.
    inFile.clear();
    inFile.seekg(0, std::ios::beg);
    rows = 0;
    cols = 0;

    // First, count the number of rows in the file (this is the dimensionality).
    std::string line;
    while (std::getline(inFile, line))
    {
      ++rows;
    }
    info = DatasetMapper<MapPolicy>(rows);

    // Now, jump back to the beginning of the file.
    inFile.clear();
    inFile.seekg(0, std::ios::beg);
    rows = 0;
    while (std::getline(inFile, line))
    {
      ++rows;

      if (rows == 1)
      {
        // Extract the number of columns.
        auto findColSize = [&cols](iter_type) { ++cols; };
        boost::spirit::qi::phrase_parse(line.begin(), line.end(),
            CreateCharRule()[findColSize] % ",", boost::spirit::ascii::space);
      }

      // I guess this is technically a second pass, but that's ok... still the
      // same idea...
      if (MapPolicy::NeedsFirstPass)
      {
        // In this case we must pass everything we parse to the MapPolicy.
        auto firstPassMap = [&](const iter_type& iter)
        {
          std::string str(iter.begin(), iter.end());
          if (str == "\t")
            str.clear();
          boost::trim(str);

          info.template MapFirstPass<T>(std::move(str), rows - 1);
        };

        // Now parse the line.
        boost::spirit::qi::phrase_parse(line.begin(), line.end(),
            CreateCharRule()[firstPassMap] % ",", boost::spirit::ascii::space);
      }
    }
  }

  template<typename T, typename MapPolicy>
  void GetTransposeMatrixSize(size_t& rows, size_t& cols, DatasetMapper<MapPolicy>& info)
  {
    // Take a pass through the file.  If the DatasetMapper policy requires it,
    // we will pass everything string through MapString().  This might be useful
    // if, e.g., the MapPolicy needs to find which dimensions are numeric or
    // categorical.

    // Reset to the start of the file.
    inFile.clear();
    inFile.seekg(0, std::ios::beg);
    rows = 0;
    cols = 0;

    std::string line;
    while (std::getline(inFile, line))
    {
      ++cols;

      if (cols == 1)
      {
        // Extract the number of dimensions.
        auto findRowSize = [&rows](iter_type) { ++rows; };
        boost::spirit::qi::phrase_parse(line.begin(), line.end(),
            CreateCharRule()[findRowSize] % ",", boost::spirit::ascii::space);

        // Now that we know the dimensionality, initialize the DatasetMapper.
        info = DatasetMapper<MapPolicy>(rows);
      }

      // If we need to do a first pass for the DatasetMapper, do it.
      if (MapPolicy::NeedsFirstPass)
      {
        size_t dim = 0;

        // In this case we must pass everything we parse to the MapPolicy.
        auto firstPassMap = [&](const iter_type& iter)
        {
          std::string str(iter.begin(), iter.end());
          if (str == "\t")
            str.clear();
          boost::trim(str);

          info.template MapFirstPass<T>(std::move(str), dim++);
        };

        // Now parse the line.
        boost::spirit::qi::phrase_parse(line.begin(), line.end(),
            CreateCharRule()[firstPassMap] % ",", boost::spirit::ascii::space);
      }
    }
  }

private:
  using iter_type = boost::iterator_range<std::string::iterator>;

  struct ElemParser
  {
    //return int_parser if the type of T is_integral
    template<typename T>
    static typename std::enable_if<std::is_integral<T>::value,
    boost::spirit::qi::int_parser<T>>::type
    Parser()
    {
      return boost::spirit::qi::int_parser<T>();
    }

    //return real_parser if T is floating_point
    template<typename T>
    static typename std::enable_if<std::is_floating_point<T>::value,
    boost::spirit::qi::real_parser<T>>::type
    Parser()
    {
      return boost::spirit::qi::real_parser<T>();
    }
  };

  bool CanOpen();

  template<typename T, typename PolicyType>
  void NonTranposeParse(arma::Mat<T> &inout, DatasetMapper<PolicyType> &infoSet)
  {
    using namespace boost::spirit;

    // Get the size of the matrix.
    size_t rows, cols;
    GetMatrixSize<T>(rows, cols, infoSet);

    // Set up output matrix.
    inout.set_size(rows, cols);
    size_t row = 0;
    size_t col = 0;

    // Reset file position.
    std::string line;
    inFile.clear();
    inFile.seekg(0, std::ios::beg);

    auto setCharClass = [&](iter_type const &iter)
    {
      std::string str(iter.begin(), iter.end());
      if (str == "\t")
      {
        str.clear();
      }
      boost::trim(str);

      inout(row, col++) = infoSet.template MapString<T>(std::move(str), row);
    };

    auto charRule = CreateCharRule();
    while (std::getline(inFile, line))
    {
      //parse the numbers from a line(ex : 1,2,3,4), if the parser find the number
      //it will execute the setNum function
      const bool canParse = qi::phrase_parse(line.begin(), line.end(),
          charRule[setCharClass] % ",", ascii::space);

      if (!canParse)
      {
        throw std::runtime_error("LoadCSV cannot parse categories");
      }

      ++row; col = 0;
    }
  }

  template<typename T, typename PolicyType>
  void TranposeParse(arma::Mat<T> &inout, DatasetMapper<PolicyType> &infoSet)
  {
    // Get matrix size.  This also initializes infoSet correctly.
    size_t rows, cols;
    GetTransposeMatrixSize<T>(rows, cols, infoSet);

    // Set the matrix size.
    inout.set_size(rows, cols);
    TranposeParseImpl(inout, infoSet);
  }

  template<typename T, typename PolicyType>
  bool TranposeParseImpl(arma::Mat<T>& inout,
                         DatasetMapper<PolicyType>& infoSet)
  {
    using namespace boost::spirit;

    size_t row = 0;
    size_t col = 0;
    std::string line;
    inFile.clear();
    inFile.seekg(0, std::ios::beg);

    auto setCharClass = [&](iter_type const &iter)
    {
      // All parsed values must be mapped.
      std::string str(iter.begin(), iter.end());
      if (str == "\t")
        str.clear();
      boost::trim(str);

      inout(row, col) = infoSet.template MapString<T>(std::move(str), row);
      ++row;
    };

    auto charRule = CreateCharRule();
    while (std::getline(inFile, line))
    {
      row = 0;
      //parse number of characters from a line, it will execute setNum if it is number,
      //else execute setCharClass, "|" means "if not a, then b"
      // Assemble the rule

      const bool canParse = qi::phrase_parse(line.begin(), line.end(),
                                             charRule[setCharClass] % ",",
                                             ascii::space);
      if(!canParse)
      {
        throw std::runtime_error("LoadCSV cannot parse categories");
      }
      ++col;
    }

    return true;
  }

  template<typename T>
  boost::spirit::qi::rule<std::string::iterator, T(), boost::spirit::ascii::space_type>
  CreateNumRule() const
  {
    using namespace boost::spirit;

    //elemParser will generate integer or real parser based on T
    auto elemParser = ElemParser::Parser<T>();
    //qi::skip can specify which characters you want to skip,
    //in this example, elemParser will parse int or double value,
    //we use qi::skip to skip space

    //qi::omit can omit the attributes of spirit, every parser of spirit
    //has attribute(the type will pass into actions(functor))
    //if you do not omit it, the attribute combine with attribute may
    //change the attribute

    //input like 2-200 or 2DM will make the parser fail,
    //so we use "look ahead parser--&" to make sure next
    //character is "," or end of line(eof) or end of file(eoi)
    //looks ahead parser will not consume any input or generate
    //any attribute
    if(extension == "csv" || extension == "txt")
    {
      return elemParser >> &(qi::lit(",") | qi::eol | qi::eoi);
    }
    else
    {
      return elemParser >> &(qi::lit("\t") | qi::eol | qi::eoi);
    }
  }

  boost::spirit::qi::rule<std::string::iterator, iter_type(), boost::spirit::ascii::space_type>
  CreateCharRule() const;

  std::string extension;
  bool fatalIfOpenFail;
  std::string fileName;
  std::ifstream inFile;
};

} // namespace data
} // namespace mlpack

#endif
