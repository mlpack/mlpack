/**
 * @file split_by_any_of.hpp
 * @author Jeffin Sam
 * @author Mikhail Lozhnikov
 *
 * Definition of the SplitByAnyOf class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_TOKENIZERS_SPLIT_BY_ANY_OF_HPP
#define MLPACK_CORE_DATA_TOKENIZERS_SPLIT_BY_ANY_OF_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/boost_backport/boost_backport_string_view.hpp>
#include <array>

namespace mlpack {
namespace data {

/**
 * The SplitByAnyOf class tokenizes a string using a set of delimiters.
 */
class SplitByAnyOf
{
 public:
  //! The type of the token which the tokenizer extracts.
  using TokenType = boost::string_view;

  //! A convenient alias for the mask type.
  using MaskType = std::array<bool, 1 << CHAR_BIT>;

  /**
   * Construct the object from the given delimiters.
   *
   * @param delimiters The given delimiters.
   */
  SplitByAnyOf(const boost::string_view delimiters)
  {
    mask.fill(false);

    for (char symbol : delimiters)
      mask[static_cast<unsigned char>(symbol)] = true;
  }

  /**
   * The function extracts the first token from the given string view and
   * then removes the prefix containing the token from the view.
   *
   * @param str String view to retrieve the token from.
   */
  boost::string_view operator()(boost::string_view& str, int ngram = 1) const
  {
    boost::string_view retval;
    while (retval.empty())
    {
      const std::size_t pos = FindFirstDelimiter(str);
      const std::size_t nthpos = FindNthDelimiter(str, ngram);
      if (pos == str.npos || nthpos == str.npos)
      {
        str.clear();
        retval = str;
        return retval;
      }
        retval = str.substr(0, nthpos);
      str.remove_prefix(pos + 1);
    }
    return retval;
  }

  /**
   * The function returns true if the given token is empty.
   *
   * @param token The given token.
   */
  static bool IsTokenEmpty(const boost::string_view token)
  {
    return token.empty();
  }

  //! Return the mask.
  const MaskType& Mask() const { return mask; }
  //! Modify the mask.
  MaskType& Mask() { return mask; }

 private:
  /**
   * The function finds the first character in the given string view equal to 
   * any of the delimiters and returns the position of the character or 
   * boost::string_view::npos if no such character is found.
   *
   * @param str String where to find the character.
   */
  size_t FindNthDelimiter(const boost::string_view str, int ntimes) const
  {
    int nPos = 0;
    for (size_t pos = 0; pos < str.size(); pos++)
    {
      if (mask[static_cast<unsigned char>(str[pos])])
      { 
        nPos++;
        if (nPos == ntimes)
          return pos;
      }
    }
    return str.npos;
  }

  size_t FindFirstDelimiter(const boost::string_view str) const
  {
    for (size_t pos = 0; pos < str.size(); pos++)
    {
      if (mask[static_cast<unsigned char>(str[pos])])
        return pos;
    }
    return str.npos;
  }

 private:
  //! The mask that corresponds to the delimiters.
  MaskType mask;
};

} // namespace data
} // namespace mlpack

#endif
