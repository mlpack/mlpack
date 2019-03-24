/**
 * @file load_csv.cpp
 * @author Tham Ngap Wei
 * @author Mehul Kumar Nirala
 *
 * A CSV reader that uses boost::spirit.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "load_csv.hpp"

using namespace boost::spirit;

namespace mlpack {
namespace data {

LoadCSV::LoadCSV(const std::string& file) :
  extension(Extension(file)),
  filename(file),
  inFile(file)
{
  // Attempt to open stream.
  CheckOpen();

  //! Spirit rule for parsing quoted string.
  boost::spirit::qi::rule<std::string::iterator, iter_type()> quotedRule;
  // Match quoted strings as: "string" or 'string'
  quotedRule = qi::raw[(qi::char_("'") >> *((qi::char_ - "'") |
                               "'" >> qi::char_("'")) >> "'") |
                       (qi::char_('"') >> *((qi::char_ - '"') |
                               '"' >> qi::char_('"')) >> '"') ];

  // Set rules.
  if (extension == "csv")
  {
    // Match all characters that are not ',', '\r', or '\n'.
    stringRule = quotedRule.copy() | qi::raw[*~qi::char_(",\r\n")];
  }
  else if (extension == "txt")
  {
    // Match all characters that are not ' ', ',', '\r', or '\n'.
    stringRule = quotedRule.copy() | qi::raw[*~qi::char_(" ,\r\n")];
  }
  else // TSV.
  {
    // Match all characters that are not '\t', '\r', or '\n'.
    stringRule = quotedRule.copy() | qi::raw[*~qi::char_("\t\r\n")];
  }

  if (extension == "csv")
  {
    // Extract a single comma as the delimiter, catching whitespace on either
    // side.
    delimiterRule = qi::raw[(*qi::char_(" ") >> qi::char_(",") >>
        *qi::char_(" "))];
  }
  else if (extension == "txt")
  {
    // This one is a little more difficult, we need to catch any number of
    // spaces more than one.
    delimiterRule = qi::raw[+qi::char_(" ")];
  }
  else // TSV.
  {
    // Catch a tab character, possibly with whitespace on either side.
    delimiterRule = qi::raw[(*qi::char_(" ") >> qi::char_("\t") >>
        *qi::char_(" "))];
  }
}

void LoadCSV::CheckOpen()
{
  if (!inFile.is_open())
  {
    std::ostringstream oss;
    oss << "Cannot open file '" << filename << "'. " << std::endl;
    throw std::runtime_error(oss.str());
  }

  inFile.unsetf(std::ios::skipws);
}

} // namespace data
} // namespace mlpack
