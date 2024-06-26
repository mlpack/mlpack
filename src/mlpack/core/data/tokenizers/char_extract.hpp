/**
 * @file core/data/tokenizers/char_extract.hpp
 * @author Jeffin Sam
 * @author Mikhail Lozhnikov
 *
 * Definition of the CharExtract class which tokenizes a string into
 * characters.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_TOKENIZERS_CHAR_EXTRACT_HPP
#define MLPACK_CORE_DATA_TOKENIZERS_CHAR_EXTRACT_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace data {

/**
 * The class is used to split a string into characters.
 */
class CharExtract
{
 public:
  //! The type of the token which the tokenizer extracts.
  using TokenType = int;

  /**
   * The function extracts the first character from the given string view and
   * removes it from the view. Each charecter is casted to unsigned char i.e. 
   * it belongs to [0, 255]. The functon returns EOF provided that the input 
   * string is empty.
   *
   * @param str String view to retrieve the next token from.
   */
  int operator()(std::string_view& str) const
  {
    if (str.empty())
      return EOF;

    const int retval = static_cast<unsigned char>(str[0]);

    str.remove_prefix(1);

    return retval;
  }

  /**
   * The function returns true if the given token is equal to EOF.
   *
   * @param token The given token.
   */
  static bool IsTokenEmpty(const int token)
  {
    return token == EOF;
  }
};

} // namespace data
} // namespace mlpack

#endif
