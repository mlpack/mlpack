/**
 * @file load_csv.hpp
 * @author ThamNgapWei
 *
 * This is a csv parsers which use to parse the csv file format
 */
#ifndef MLPACK_CORE_DATA_LOAD_CSV_HPP
#define MLPACK_CORE_DATA_LOAD_CSV_HPP

#include <boost/spirit/include/qi.hpp>

#include <mlpack/core.hpp>
#include <mlpack/core/util/log.hpp>

#include <set>
#include <string>

#include "extension.hpp"
#include "format.hpp"
#include "dataset_info.hpp"

namespace mlpack {
namespace data /** Functions to load and save matrices and models. */ {

/**
 *Load the csv file.This class use boost::spirit
 *to implement the parser, please refer to following link
 *http://theboostcpplibraries.com/boost.spirit for quick review.
 */
class LoadCSV
{
public:
  explicit LoadCSV(std::string file, bool fatal = false)  :
    extension(Extension(file)),
    fatalIfOpenFail(fatal),
    fileName(std::move(file)),
    inFile(fileName)
  {
    CanOpen();
  }

  template<typename T>
  void Load(arma::Mat<T> &inout, DatasetInfo &infoSet, bool transpose = true)
  {
    if(!CanOpen())
    {
      return;
    }

    //please refer to the comments of ColSize if you do not familiar
    //with boost::spirit yet
    if(transpose)
    {
      TranposeParse(inout, infoSet);
    }
    else
    {
      NonTranposeParse(inout, infoSet);
    }
  }

  size_t ColSize()
  {
    //boost tokenizer or strtok can do the same thing, I use
    //spirit at here because I think this is a nice example
    using namespace boost::spirit;
    using bsi_type = boost::spirit::istream_iterator;
    using iter_type = boost::iterator_range<bsi_type>;

    inFile.clear();
    inFile.seekg(0, std::ios::beg);
    //spirit::qi requires iterators to be atleast forward iterators,
    //but std::istream_iterator is input iteraotr, so we use
    //boost::spirit::istream_iterator to overcome this problem
    bsi_type begin(inFile);
    bsi_type end;
    size_t col = 0;

    //the parser of boost spirit can work with "actions"(functor)
    //when the parser find match target, this functor will be executed
    auto findColSize = [&col](iter_type){ ++col; };

    //qi::char_ bite an character
    //qi::char_(",\r\n") only bite a "," or "\r" or "\n" character
    //* means the parser(ex : qi::char_) can bite [0, any size] of characters
    //~ means negate, so ~qi::char_(",\r\n") means I want to bite anything except of ",\r\n"
    //parse % "," means you want to parse string like "1,2,3,apple"(noticed it without last comma)

    //qi::raw restrict the automatic conversion of boost::spirit, without it, spirit parser
    //will try to convert the string to std::string, this may cause memory allocation(if small string
    //optimization fail).
    //After we wrap the parser with qi::raw, the attribute(the data accepted by functor) will
    //become boost::iterator_range, this could save a tons of memory allocations
    qi::parse(begin, end, qi::raw[*~qi::char_(",\r\n")][findColSize] % ",");

    return col;
  }

  size_t RowSize()
  {
    inFile.clear();
    inFile.seekg(0, std::ios::beg);
    size_t row = 0;
    std::string line;
    while(std::getline(inFile, line))
    {
      ++row;
    }

    return row;
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

  bool CanOpen()
  {
    if(!inFile.is_open())
    {
      if(fatalIfOpenFail)
      {
        Log::Fatal << "Cannot open file '" << fileName << "'. " << std::endl;
      }
      else
      {
        Log::Warn << "Cannot open file '" << fileName << "'; load failed."
                  << std::endl;
      }
      return false;
    }
    inFile.unsetf(std::ios::skipws);

    return true;
  }

  template<typename T>
  void NonTranposeParse(arma::Mat<T> &inout, DatasetInfo &infoSet)
  {
    using namespace boost::spirit;

    size_t row = 0;
    size_t col = 0;
    infoSet = DatasetInfo(RowSize());
    std::string line;
    inout.set_size(infoSet.Dimensionality(), ColSize());
    inFile.clear();
    inFile.seekg(0, std::ios::beg);

    auto setNum = [&](T val)
    {
      inout(row, col++) = val;
    };
    auto setCharClass = [&](iter_type const &iter)
    {
      std::string str(iter.begin(), iter.end());
      if(str == "\t")
      {
        str.clear();
      }
      inout(row, col++) =
          static_cast<T>(infoSet.MapString(std::move(str),
                                           row));
    };

    auto numRule = CreateNumRule<T>();
    auto charRule = CreateCharRule();
    while(std::getline(inFile, line))
    {
      auto begin = line.begin();
      //parse the numbers from a line(ex : 1,2,3,4), if the parser find the number
      //it will execute the setNum function
      qi::phrase_parse(begin, line.end(), numRule[setNum] % ",", ascii::space);
      if(col != inout.n_cols)
      {
        begin = line.begin();
        col = 0;
        const bool canParse = qi::phrase_parse(begin, line.end(),
                                               charRule[setCharClass] % ",",
                                               ascii::space);
        if(!canParse)
        {
          throw std::runtime_error("LoadCSV cannot parse categories");
          break;
        }
      }
      ++row; col = 0;
    }
  }

  template<typename T>
  void TranposeParse(arma::Mat<T> &inout, DatasetInfo &infoSet)
  {
    infoSet = DatasetInfo(ColSize());
    inout.set_size(infoSet.Dimensionality(), RowSize());
    size_t parseTime = 0;
    std::set<size_t> mapCols;
    while(!TranposeParseImpl(inout, infoSet, mapCols))
    {
      //avoid infinite loop
      ++parseTime;
      infoSet = DatasetInfo(inout.n_rows);
      if(parseTime == inout.n_rows)
      {
        return;
      }
    }
  }

  template<typename T>
  bool TranposeParseImpl(arma::Mat<T> &inout, DatasetInfo &infoSet,
                         std::set<size_t> &mapCols)
  {
    using namespace boost::spirit;

    size_t row = 0;
    size_t col = 0;
    size_t progress = 0;
    std::string line;
    inFile.clear();
    inFile.seekg(0, std::ios::beg);
    auto setNum = [&](T val)
    {
      if(mapCols.find(progress) != std::end(mapCols))
      {
        inout(row, col) =
            static_cast<T>(infoSet.MapString(std::to_string(val),
                                             progress));
      }
      else
      {
        inout(row, col) = val;
      }
      ++progress; ++row;
    };
    auto setCharClass = [&](iter_type const &iter)
    {
      if(mapCols.find(progress) != std::end(mapCols))
      {
        std::string str(iter.begin(), iter.end());
        if(str == "\t")
        {
          str.clear();
        }
        inout(row, col) =
            static_cast<T>(infoSet.MapString(std::move(str),
                                             progress));
      }
      else
      {
        mapCols.insert(progress);
      }
      ++progress; ++row;
    };

    auto numRule = CreateNumRule<T>();
    auto charRule = CreateCharRule();
    while(std::getline(inFile, line))
    {
      auto begin = line.begin();
      row = 0;
      progress = 0;
      const size_t oldSize = mapCols.size();
      //parse number of characters from a line, it will execute setNum if it is number,
      //else execute setCharClass, "|" means "if not a, then b"
      const bool canParse = qi::phrase_parse(begin, line.end(),
                                             (numRule[setNum] | charRule[setCharClass]) % ",",
                                             ascii::space);
      if(!canParse)
      {
        throw std::runtime_error("LoadCSV cannot parse categories");
      }
      if(mapCols.size() > oldSize)
      {
        return false;
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
    //but we do not want space to intefere it, so we skip it by qi::skip

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
  CreateCharRule() const
  {
    using namespace boost::spirit;

    if(extension == "csv" || extension == "txt")
    {
      return qi::raw[*~qi::char_(",\r\n")];
    }
    else
    {
      return qi::raw[*~qi::char_("\t\r\n")];
    }
  }

  std::string extension;
  bool fatalIfOpenFail;
  std::string fileName;
  std::ifstream inFile;
};

} // namespace data
} // namespace mlpack

#endif
