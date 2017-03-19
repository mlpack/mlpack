#include "load_csv.hpp"

namespace mlpack {
namespace data {

LoadCSV::LoadCSV(std::string file, bool fatal) :
  extension(Extension(file)),
  fatalIfOpenFail(fatal),
  fileName(std::move(file)),
  inFile(fileName)
{
  CanOpen();
}

bool LoadCSV::CanOpen()
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

size_t LoadCSV::ColSize()
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

size_t LoadCSV::RowSize()
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

boost::spirit::qi::rule<std::string::iterator, LoadCSV::iter_type(), boost::spirit::ascii::space_type>
LoadCSV::CreateCharRule() const
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

} // namespace data
} // namespace mlpack
